"""Permanent model-only snapshots for evaluation and generation."""
from pathlib import Path

from .io import atomic_write, write_json


class MilestoneStore:
    def __init__(self, directory, targets, model_dir):
        self.directory = Path(directory)
        self.targets = targets
        self.model_dir = model_dir
        if targets:
            self.directory.mkdir(parents=True, exist_ok=True)

    def due(self, previous_tokens, current_tokens):
        return [value for value in self.targets if previous_tokens < value <= current_tokens]

    def path_for(self, target):
        return self.directory / f'tokens_{target:010d}.safetensors'

    def save(self, model, progress, target):
        from safetensors import safe_open
        from safetensors.torch import save_file
        path = self.path_for(target)
        sidecar = path.with_suffix('.json')
        if path.exists() and sidecar.exists():
            return
        if path.exists():
            with safe_open(str(path), framework='pt', device='cpu') as snapshot:
                info = snapshot.metadata() or {}
            step, tokens = int(info['step']), int(info['tokens_processed'])
        else:
            import torch
            # Separate copies support tied embeddings without shared-storage errors.
            # Snapshots stay BF16 (inference precision) even when training uses FP32 masters.
            weights = {key: value.detach().to(device='cpu', copy=True,
                                              dtype=torch.bfloat16 if value.is_floating_point() else value.dtype
                                              ).contiguous()
                       for key, value in model.state_dict().items()}
            step, tokens = progress.step, progress.tokens_processed
            atomic_write(path, lambda temporary: save_file(weights, str(temporary),
                         metadata={'step': str(step), 'tokens_processed': str(tokens)}))
        write_json(sidecar, {'step': step, 'target_tokens': target,
                            'tokens_processed': tokens, 'model_dir': self.model_dir})
        print(f'[milestone] saved {path} (step {step})', flush=True)

    def recover(self, model, progress, tokens_per_step):
        for target in self.targets:
            path = self.path_for(target)
            missing_sidecar = path.exists() and not path.with_suffix('.json').exists()
            last_update = max(0, progress.tokens_processed - tokens_per_step) < target <= progress.tokens_processed
            if missing_sidecar or last_update:
                self.save(model, progress, target)
