"""Resolved training settings and strict JSON/CLI configuration handling."""
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path


def token_count(value):
    """Accept exact counts or decimal K/M/B suffixes, e.g. 5B or 1.5B."""
    try:
        text = str(value).strip().replace('_', '').upper()
        multiplier = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}.get(text[-1:], 1)
        number = Decimal(text[:-1] if multiplier != 1 else text) * multiplier
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            raise ValueError
        return int(number)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f'Invalid token count: {value!r}; use a positive integer or e.g. 5B') from exc


def milestone_counts(value):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(',') if part.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError('milestone_tokens must be a list or a comma-separated string')
    return sorted({token_count(part) for part in value})


def read_config(path, allowed):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Config must be a JSON object')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown config fields: {", ".join(sorted(unknown))}')
    return value


@dataclass
class TrainConfig:
    model_dir: str = 'model'
    data_dir: str = 'prepared'
    out_dir: str = 'checkpoints'
    run_dir: str | None = None
    seq_len: int = 2048
    batch_size: int = 2
    grad_accum: int = 16
    max_steps: int | None = None
    target_tokens: int | None = None
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    eval_every: int = 50
    eval_batches: int = 10
    save_every: int = 50
    keep_checkpoints: int = 2
    log_every: int = 1
    seed: int = 42
    resume: str = ''
    init_from: str = ''
    new_lr_schedule: bool = False
    continuation_lr: float | None = None
    continuation_warmup_steps: int = 0
    no_checkpointing: bool = False
    milestone_dir: str = 'milestones'
    milestone_tokens: str | list = '50000000,100000000,250000000,500000000,1000000000'
    allow_data_change: bool = False
    # Checkpoints from before this field resume in FP32 (BF16 weights upcast losslessly).
    param_dtype: str = 'float32'

    @classmethod
    def from_overrides(cls, overrides, saved=None):
        names = {item.name for item in fields(cls)}
        unknown = set(overrides) - names
        if unknown:
            raise ValueError(f'Unknown training settings: {", ".join(sorted(unknown))}')
        values = {key: value for key, value in (saved or {}).items() if key in names}
        # Commands choose the source and stage explicitly; never inherit them.
        for key in ('resume', 'init_from', 'new_lr_schedule', 'allow_data_change',
                    'continuation_lr', 'continuation_warmup_steps'):
            values.pop(key, None)
        values.update(overrides)
        config = cls(**values)
        if config.target_tokens is not None:
            config.target_tokens = token_count(config.target_tokens)
        config.milestone_tokens = milestone_counts(config.milestone_tokens)
        return config

    @property
    def tokens_per_step(self):
        return self.seq_len * self.batch_size * self.grad_accum

    def to_dict(self):
        # Keep args compatible with the original script's comma-separated list.
        values = asdict(self)
        values['milestone_tokens'] = ','.join(map(str, self.milestone_tokens))
        return values

    def validate(self):
        for key in ('seq_len', 'batch_size', 'grad_accum', 'eval_batches', 'keep_checkpoints', 'log_every'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.seq_len < 2:
            raise ValueError('seq_len must be >= 2 for shifted causal-language-model labels')
        for key in ('warmup_steps', 'continuation_warmup_steps', 'eval_every', 'save_every', 'seed'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.seed >= 2**32:
            raise ValueError('seed must be less than 2**32')
        if self.max_steps is not None and (isinstance(self.max_steps, bool)
                or not isinstance(self.max_steps, int) or self.max_steps < 1):
            raise ValueError('max_steps must be a positive integer')
        for key in ('lr', 'min_lr', 'weight_decay', 'continuation_lr'):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('model_dir', 'data_dir', 'out_dir', 'milestone_dir'):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{key} must be a nonempty path string')
        for key in ('new_lr_schedule', 'no_checkpointing', 'allow_data_change'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'{key} must be a JSON boolean')
        for key in ('resume', 'init_from'):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f'{key} must be a path string')
        if self.param_dtype not in ('float32', 'bfloat16'):
            raise ValueError('param_dtype must be float32 or bfloat16')
        if self.resume and self.init_from:
            raise ValueError('resume and init_from are mutually exclusive')
        if self.new_lr_schedule and not self.resume:
            raise ValueError('A continuation LR stage requires a full checkpoint')
        if (self.continuation_lr is not None or self.continuation_warmup_steps) and not self.new_lr_schedule:
            raise ValueError('Continuation LR settings require an explicit extension stage')
        if self.continuation_lr is not None and self.continuation_warmup_steps < 1:
            raise ValueError('An explicit continuation_lr requires positive continuation_warmup_steps')
