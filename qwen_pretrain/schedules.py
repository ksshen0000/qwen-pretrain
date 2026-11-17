"""Dependency-free learning-rate and token accounting for pretraining."""
import math


def cosine_lr(step, schedule):
    """A schedule is independent of the current process's start step."""
    offset = step - schedule['start_step']
    warmup = schedule['warmup_steps']
    if warmup and offset <= warmup:
        fraction = max(0, offset) / warmup
        return schedule['start_lr'] + (schedule['peak_lr'] - schedule['start_lr']) * fraction
    duration = schedule['end_step'] - schedule['start_step'] - warmup
    progress = min(1., max(0., (offset - warmup) / max(1, duration)))
    return schedule['min_lr'] + .5 * (schedule['peak_lr'] - schedule['min_lr']) * (1 + math.cos(math.pi * progress))


def make_schedule(start_step, end_step, peak_lr, min_lr, warmup_steps=0, start_lr=0.):
    if start_step < 0:
        raise ValueError('Schedule start step must be nonnegative')
    if end_step <= start_step:
        raise ValueError('Training end step must be greater than checkpoint step')
    if not 0 <= warmup_steps < end_step - start_step:
        raise ValueError('Warmup must be shorter than the remaining training stage')
    if not all(math.isfinite(v) for v in (peak_lr, min_lr, start_lr)):
        raise ValueError('Learning rates must be finite')
    if not 0 <= min_lr <= peak_lr or start_lr < 0:
        raise ValueError('Require 0 <= min_lr <= peak_lr and start_lr >= 0')
    return dict(start_step=start_step, end_step=end_step, peak_lr=peak_lr,
                min_lr=min_lr, warmup_steps=warmup_steps, start_lr=start_lr)


def target_step(target_tokens, tokens_processed, start_step, tokens_per_step):
    if tokens_per_step <= 0 or min(tokens_processed, start_step) < 0:
        raise ValueError('Require positive tokens_per_step and nonnegative checkpoint progress')
    if target_tokens <= tokens_processed:
        raise ValueError('Target tokens must exceed checkpoint tokens_processed')
    return start_step + (target_tokens - tokens_processed + tokens_per_step - 1) // tokens_per_step
