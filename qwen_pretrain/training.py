"""Training orchestration and optimizer-update loop."""
import json
import math
from pathlib import Path
import time

from .checkpoints import (CheckpointStore, Progress, checkpoint_files, load_checkpoint,
                          load_model_weights, optimizer_lr, resolve_checkpoint, saved_schedule)
from .config import TrainConfig
from .io import MetricsWriter, directory_lock, write_json
from .milestones import MilestoneStore
from .optim import build_optimizer, load_optimizer_state
from .runtime import StopController, preserve_rng, restore_rng, seed_everything
from .schedules import cosine_lr, make_schedule, target_step


def _resolve_paths(overrides, mode):
    values = dict(overrides)
    for key in ('run_dir', 'out_dir'):
        if key in values and values[key] is not None and (not isinstance(values[key], str) or not values[key]):
            raise ValueError(f'{key} must be a nonempty path string')
    if values.get('run_dir'):
        root = Path(values['run_dir']).expanduser().resolve()
        if mode == 'resume' and not root.is_dir():
            raise FileNotFoundError(f'Run directory not found: {root}')
        # Existing flat checkpoint directories can be resumed without moving files.
        flat_files = checkpoint_files(root)
        nested_files = checkpoint_files(root / 'checkpoints')
        if mode in ('train', 'extend') and flat_files:
            raise ValueError('Run directory already contains flat checkpoints; resume it or choose a new directory')
        if mode == 'resume' and flat_files and nested_files:
            raise ValueError('Run directory has both flat and nested checkpoints; select the intended checkpoint directory explicitly')
        flat = mode == 'resume' and bool(flat_files)
        values['out_dir'] = str(root if flat else root / 'checkpoints')
        values['run_dir'] = str(root)
        if mode in ('train', 'extend'):
            values.setdefault('milestone_dir', str(root / 'milestones'))
    else:
        values['run_dir'] = None
        values['out_dir'] = str(Path(values.get('out_dir', 'checkpoints')).expanduser().resolve())
        root = Path(values['out_dir'])
    return values, root


def _validate_resume(config, state, dataset):
    saved = state['args']
    defaults = TrainConfig()
    for key in ('seq_len', 'batch_size', 'grad_accum', 'seed', 'no_checkpointing'):
        if saved.get(key, getattr(defaults, key)) != getattr(config, key):
            raise ValueError(f'Checkpoint mismatch for {key}: {saved.get(key)} != {getattr(config, key)}')
    if Path(saved['model_dir']).expanduser().resolve() != Path(config.model_dir).resolve():
        raise ValueError('model_dir changed; resume requires the original model configuration/tokenizer')
    previous = state.get('dataset_info')
    changed = (Path(saved['data_dir']).expanduser().resolve() != Path(config.data_dir).resolve()
               or (previous is not None and previous != dataset.info))
    if changed and not config.allow_data_change:
        raise ValueError('Dataset path/metadata/size changed; pass --allow-data-change after checking the tokenizer')
    if previous:
        for key in ('vocab_size', 'eos_token_id'):
            if previous['meta'].get(key) != dataset.meta.get(key):
                raise ValueError(f'Dataset tokenizer metadata changed: {key}')
    if changed:
        print('[resume] continuing on the explicitly selected new dataset', flush=True)


def _schedule(config, progress, state):
    existing = saved_schedule(state) if state is not None else None
    if config.max_steps is None:
        config.max_steps = existing['end_step'] if existing else 153
    if config.target_tokens is not None:
        if config.target_tokens <= progress.tokens_processed:
            if state is None or config.new_lr_schedule:
                raise ValueError('The new training target must exceed current tokens_processed')
            return existing, True
        config.max_steps = target_step(config.target_tokens, progress.tokens_processed,
                                      progress.step, config.tokens_per_step)
    if config.max_steps <= progress.step:
        if state is not None and config.max_steps == progress.step and not config.new_lr_schedule:
            return existing, True
        raise ValueError('Training endpoint must exceed the checkpoint step')
    if config.new_lr_schedule:
        current_lr = optimizer_lr(state)
        peak = current_lr if config.continuation_lr is None else config.continuation_lr
        return make_schedule(progress.step, config.max_steps, peak, config.min_lr,
                             config.continuation_warmup_steps, current_lr), False
    if existing:
        if config.max_steps != existing['end_step']:
            raise ValueError('Endpoint changed; use the extend command to create a new LR stage')
        print('[resume] continuing the saved learning-rate schedule', flush=True)
        return existing, False
    return make_schedule(progress.step, config.max_steps, config.lr, config.min_lr, config.warmup_steps), False


def run_training(overrides, mode='train'):
    """Resolve a stage before allocating a model; all writes happen under one lock."""
    if mode not in ('train', 'extend', 'resume', 'legacy'):
        raise ValueError(f'Unknown training mode: {mode}')
    values, root = _resolve_paths(overrides, mode)
    if mode == 'extend':
        values['new_lr_schedule'] = True
        if not values.get('resume'):
            raise ValueError('extend requires --checkpoint')
        if not any(values.get(key) is not None for key in ('target_tokens', 'max_steps')):
            raise ValueError('extend requires an explicit target_tokens or max_steps')
    elif mode == 'resume':
        values.setdefault('resume', 'auto')
    elif mode == 'train' and (values.get('resume') or values.get('new_lr_schedule')):
        raise ValueError('Use resume or extend to load a full training checkpoint')

    directory = Path(values['out_dir'])
    with directory_lock(directory, '.train.lock'):
        candidates = checkpoint_files(directory)
        extending = bool(values.get('new_lr_schedule'))
        if extending and candidates:
            raise ValueError('Extension output already has checkpoints. Use resume to continue it, or choose a new run directory')
        state = None
        if values.get('resume'):
            source = resolve_checkpoint(values['resume'], directory)
            state = load_checkpoint(source)
            values['resume'] = str(source)
            if candidates and int(candidates[0].stem[5:]) > int(state['step']):
                raise ValueError('Output has newer checkpoints; resume auto or choose a new run directory for a branch')
        elif candidates:
            raise ValueError('Output already has checkpoints; use resume or choose a new run directory')

        # An explicit endpoint replaces an inherited endpoint, never combines them.
        if 'max_steps' in values and 'target_tokens' not in values:
            values['target_tokens'] = None
        if 'target_tokens' in values and 'max_steps' not in values:
            values['max_steps'] = None
        config = TrainConfig.from_overrides(values, state['args'] if state is not None else None)
        config.validate()
        for key in ('model_dir', 'data_dir', 'out_dir', 'milestone_dir'):
            setattr(config, key, str(Path(getattr(config, key)).expanduser().resolve()))
        progress = Progress.from_checkpoint(state) if state is not None else Progress()
        weights = None
        if config.init_from:
            if Path(config.init_from).suffix != '.safetensors':
                raise ValueError('init_from requires a model-only milestone .safetensors')
            weights, info = load_model_weights(config.init_from)
            if not {'step', 'tokens_processed'} <= info.keys():
                raise ValueError('Milestone lacks step/tokens_processed metadata')
            progress = Progress(int(info['step']), int(info['tokens_processed']))
            if progress.step < 0 or progress.tokens_processed < 0:
                raise ValueError('Milestone progress must be nonnegative')
            print('[init] weights only; optimizer and random state start fresh', flush=True)
        schedule, completed = _schedule(config, progress, state)

        from .data.tokens import BatchSampler, TokenDataset
        dataset = TokenDataset(config.data_dir, config.seq_len)
        if state is not None:
            _validate_resume(config, state, dataset)
        milestones = MilestoneStore(config.milestone_dir, config.milestone_tokens, config.model_dir)
        # Completed runs normally need no GPU/model. Missing export repair is the
        # only exception because historical weights must come from this checkpoint.
        repair_needed = any(
            (milestones.path_for(value).exists() and not milestones.path_for(value).with_suffix('.json').exists())
            or (max(0, progress.tokens_processed - config.tokens_per_step) < value <= progress.tokens_processed
                and not milestones.path_for(value).exists())
            for value in config.milestone_tokens)
        if completed and not repair_needed:
            print(f'[done] checkpoint already reached its target: step={progress.step} tokens={progress.tokens_processed:,}', flush=True)
            return

        import torch
        from .modeling import build_model, require_device
        require_device('cuda', training=True)
        seed_everything(config.seed)
        model = build_model(config.model_dir, 'cuda', training=True,
                            gradient_checkpointing=not config.no_checkpointing,
                            weights=state['model'] if state is not None else weights,
                            param_dtype=config.param_dtype)
        del weights
        if model.config.vocab_size < dataset.meta['vocab_size']:
            raise ValueError('Model vocabulary is smaller than the prepared tokenizer vocabulary')
        optimizer = build_optimizer(model, config)
        sampler = BatchSampler(dataset, config, 'cuda')
        if state is not None:
            load_optimizer_state(optimizer, model, state['optimizer'])
            sampler.restore(state)
            restore_rng(state)
            print(f'[resume] step={progress.step} tokens={progress.tokens_processed:,} lr={optimizer_lr(state):.8g}', flush=True)
        if state is not None or config.init_from:
            milestones.recover(model, progress, config.tokens_per_step)
        state = None  # Release CPU model/optimizer tensors before training.
        if completed:
            print('[done] target reached; missing milestone export repaired', flush=True)
            return

        print(f'GPU: {torch.cuda.get_device_name()} | model params: {sum(value.numel() for value in model.parameters()):,} '
              f'| param_dtype={config.param_dtype}', flush=True)
        print(f'[schedule] {json.dumps(schedule)}', flush=True)
        print(f'[training] tokens/step={config.tokens_per_step:,} final_step={config.max_steps} '
              f'final_tokens={progress.tokens_processed + (config.max_steps - progress.step) * config.tokens_per_step:,}', flush=True)
        write_json(root / 'run.json', {'config': config.to_dict(), 'schedule': schedule,
                                      'start_step': progress.step, 'start_tokens': progress.tokens_processed})
        trainer = Trainer(config, model, optimizer, sampler, schedule, progress,
                          CheckpointStore(directory, config.keep_checkpoints), milestones,
                          MetricsWriter(root / 'metrics.jsonl'))
        # Persist the stage before its first update. Even interruption during
        # initial warmup can now resume without creating another LR schedule.
        if not checkpoint_files(directory):
            for group in optimizer.param_groups:
                group['lr'] = cosine_lr(progress.step, schedule)
            trainer.save()
        with StopController() as stop:
            trainer.run(stop)


class Trainer:
    def __init__(self, config, model, optimizer, sampler, schedule, progress,
                 checkpoints, milestones, metrics):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.sampler = sampler
        self.schedule = schedule
        self.progress = progress
        self.checkpoints = checkpoints
        self.milestones = milestones
        self.metrics = metrics

    def save(self):
        self.checkpoints.save(self.progress, self.model, self.optimizer, self.config,
                              self.sampler, self.schedule, self.sampler.dataset.info)

    def update(self, lr):
        import torch
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        # Accumulate on the GPU; one host sync per update instead of two per micro-batch.
        loss_sum = torch.zeros((), device='cuda', dtype=torch.float32)
        for _ in range(self.config.grad_accum):
            tokens = self.sampler.batch('train')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Hugging Face causal LMs shift labels internally.
                loss = self.model(input_ids=tokens, labels=tokens, use_cache=False).loss
            (loss / self.config.grad_accum).backward()
            loss_sum += loss.detach().float() / self.config.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        loss_value, grad_norm = torch.stack([loss_sum, grad_norm.float()]).tolist()
        # Checked before optimizer.step(): a non-finite loss or gradient never updates weights.
        if not math.isfinite(loss_value):
            raise FloatingPointError('Non-finite loss; the abnormal update will not be saved')
        if not math.isfinite(grad_norm):
            raise FloatingPointError('Non-finite gradient norm; the abnormal update will not be saved')
        self.optimizer.step()
        return loss_value, grad_norm

    def evaluate(self):
        import numpy as np
        import torch
        losses = []
        fixed_rng = np.random.default_rng(self.config.seed + 200)
        self.model.eval()
        try:
            with torch.no_grad(), preserve_rng():
                for _ in range(self.config.eval_batches):
                    tokens = self.sampler.batch('val', fixed_rng)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = self.model(input_ids=tokens, labels=tokens, use_cache=False).loss
                    losses.append(loss.detach().float().item())
        finally:
            self.model.train()
        result = sum(losses) / len(losses)
        if not math.isfinite(result):
            raise FloatingPointError('Non-finite validation loss; the last checkpoint remains available')
        return result

    def run(self, stop):
        import torch
        config, progress = self.config, self.progress
        start_step = progress.step
        started = time.monotonic()
        for step in range(start_step + 1, config.max_steps + 1):
            if stop.requested:
                self.save()
                break
            lr = cosine_lr(step, self.schedule)
            loss, grad_norm = self.update(lr)
            previous_tokens = progress.tokens_processed
            progress.step = step
            progress.tokens_processed += config.tokens_per_step
            if step % config.log_every == 0 or step == start_step + 1:
                elapsed = max(1e-5, time.monotonic() - started)
                speed = (step - start_step) * config.tokens_per_step / elapsed
                print(f'step {step}/{config.max_steps} tokens={progress.tokens_processed:,} '
                      f'loss={loss:.4f} grad_norm={grad_norm:.3f} lr={lr:.2e} avg_tok/s={speed:.0f} '
                      f'peak_vram={torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)
                self.metrics.write('train', step=step, tokens=progress.tokens_processed,
                                   loss=loss, grad_norm=grad_norm, lr=lr, avg_tokens_per_second=speed)
            due = self.milestones.due(previous_tokens, progress.tokens_processed)
            evaluate = bool(config.eval_every and step % config.eval_every == 0)
            saved = bool((config.save_every and step % config.save_every == 0)
                         or step == config.max_steps or stop.requested or due or evaluate)
            if saved:
                self.save()
            for target in due:
                self.milestones.save(self.model, progress, target)
            if evaluate and not stop.requested:
                value = self.evaluate()
                print(f'[eval] step={step} val_loss={value:.4f}', flush=True)
                self.metrics.write('eval', step=step, tokens=progress.tokens_processed, loss=value)
            if stop.requested:
                if not saved:
                    self.save()
                break
        if stop.requested:
            print(f'[interrupted] saved completed step {progress.step}; resume this run directory', flush=True)
        else:
            print('Done.', flush=True)
