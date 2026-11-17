"""Full training checkpoints; legacy dictionary keys remain readable."""
from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write

FORMAT_VERSION = 2


def checkpoint_files(directory):
    return sorted((item for item in Path(directory).glob('step_*.pt')
                   if item.is_file() and item.stem[5:].isdigit()),
                  key=lambda item: int(item.stem[5:]), reverse=True)


def resolve_checkpoint(value, directory):
    if value == 'auto':
        candidates = checkpoint_files(directory)
        if not candidates:
            raise FileNotFoundError(f'No step_*.pt checkpoints in {directory}')
        return candidates[0]
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    if path.suffix != '.pt':
        raise ValueError('Training resume/extension requires a full .pt checkpoint')
    return path


def load_checkpoint(path):
    """Load only trusted training files: AdamW/RNG recovery requires pickle."""
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    required = {'step', 'model', 'optimizer', 'args', 'train_rng', 'eval_rng', 'torch_rng', 'cuda_rng'}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError(f'Not a complete training checkpoint: {path}')
    if state.get('format_version', 1) not in (1, FORMAT_VERSION):
        raise ValueError(f'Unsupported checkpoint format: {state["format_version"]}')
    return state


@dataclass
class Progress:
    step: int = 0
    tokens_processed: int = 0

    @classmethod
    def from_checkpoint(cls, state):
        saved = state['args']
        step = int(state['step'])
        tokens = int(state.get('tokens_processed',
                     step * saved['seq_len'] * saved['batch_size'] * saved['grad_accum']))
        if step < 0 or tokens < 0:
            raise ValueError('Checkpoint progress must be nonnegative')
        return cls(step, tokens)


def saved_schedule(state):
    if state.get('lr_schedule') is not None:
        return dict(state['lr_schedule'])
    # The original script did not persist a scheduler object.
    saved = state['args']
    return dict(start_step=0, end_step=saved['max_steps'], peak_lr=saved['lr'],
                min_lr=saved['min_lr'], warmup_steps=saved['warmup_steps'], start_lr=0.)


def optimizer_lr(state):
    groups = state['optimizer']['param_groups']
    if not groups or any(group['lr'] != groups[0]['lr'] for group in groups):
        raise ValueError('Expected a single learning rate across optimizer parameter groups')
    return float(groups[0]['lr'])


class CheckpointStore:
    def __init__(self, directory, keep=2):
        self.directory = Path(directory)
        self.keep = keep

    def save(self, progress, model, optimizer, config, sampler, schedule, dataset_info):
        import torch
        from .runtime import capture_rng
        state = {
            'format_version': FORMAT_VERSION,
            'step': progress.step,
            'tokens_processed': progress.tokens_processed,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': config.to_dict(),
            'lr_schedule': schedule,
            'dataset_info': dataset_info,
            **sampler.rng_state(),
            **capture_rng(),
        }
        path = self.directory / f'step_{progress.step:06d}.pt'
        atomic_write(path, lambda temporary: torch.save(state, temporary))
        print(f'[checkpoint] saved {path}', flush=True)
        # Prune only after publication. Incomplete writes leave older files intact.
        for old in checkpoint_files(self.directory)[self.keep:]:
            try:
                old.unlink()
                print(f'[checkpoint] removed {old}', flush=True)
            except OSError as exc:
                print(f'[checkpoint] warning: could not remove {old}: {exc}', flush=True)
        return path


def load_model_weights(path):
    """Read inference weights without restoring optimizer or random-state objects."""
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework='pt', device='cpu') as snapshot:
            info = snapshot.metadata() or {}
        return load_file(str(path), device='cpu'), info
    if path.suffix != '.pt':
        raise ValueError('Expected a .pt checkpoint or .safetensors milestone')
    import torch
    # No unsafe fallback: inference never needs arbitrary pickle execution.
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(state, dict) or 'model' not in state:
        raise ValueError('Checkpoint must contain a model state_dict')
    info = {key: state[key] for key in ('step', 'tokens_processed') if key in state}
    return state['model'], info
