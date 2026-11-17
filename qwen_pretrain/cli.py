"""Unified commands. Heavy ML dependencies are imported only by an operation."""
import argparse
import json
from pathlib import Path

from .config import read_config, token_count


def _training_arguments(parser, mode):
    parser.add_argument('--config', type=Path, help='Flat JSON config; explicit CLI flags take precedence')
    parser.add_argument('--run-dir', help='Run directory containing checkpoints/, milestones/ and logs')
    if mode != 'train':
        parser.add_argument('--checkpoint', dest='resume', help='Full .pt source; resume defaults to auto')
    for flag, value_type in [('model-dir', str), ('data-dir', str), ('eval-every', int),
                             ('eval-batches', int), ('save-every', int), ('keep-checkpoints', int),
                             ('log-every', int), ('milestone-dir', str), ('milestone-tokens', str),
                             ('param-dtype', str)]:
        parser.add_argument(f'--{flag}', type=value_type)
    if mode != 'train':
        parser.add_argument('--allow-data-change', action=argparse.BooleanOptionalAction, default=None)
    if mode == 'resume':
        return
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument('--target-tokens', type=token_count, help='Cumulative token target, e.g. 5B')
    endpoint.add_argument('--max-steps', type=int, help='Cumulative optimizer-update target')
    parser.add_argument('--min-lr', type=float)
    if mode == 'extend':
        parser.add_argument('--peak-lr', dest='continuation_lr', type=float,
                            help='New peak; omitted means checkpoint current LR')
        parser.add_argument('--warmup-steps', dest='continuation_warmup_steps', type=int,
                            help='Steps to linearly transition from checkpoint LR to the new peak')
    else:
        parser.add_argument('--lr', type=float)
        parser.add_argument('--warmup-steps', type=int)
        parser.add_argument('--weight-decay', type=float)
        parser.add_argument('--seq-len', type=int)
        parser.add_argument('--batch-size', type=int)
        parser.add_argument('--grad-accum', type=int)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--no-checkpointing', action='store_true', default=None)


def _run_training(args, parser, mode):
    from .training import run_training
    allowed = {action.dest for action in parser._actions} - {'help', 'config'}
    aliases = {'checkpoint': 'resume'} if mode != 'train' else {}
    if mode == 'extend':
        aliases.update(peak_lr='continuation_lr', warmup_steps='continuation_warmup_steps')
    values = read_config(args.config, allowed | set(aliases))
    for public, internal in aliases.items():
        if public in values:
            if internal in values:
                raise ValueError(f'Config cannot specify both {public} and {internal}')
            values[internal] = values.pop(public)
    explicit = {key: value for key, value in vars(args).items()
                if key in allowed and value is not None}
    if explicit.get('target_tokens') is not None:
        values.pop('max_steps', None)
    elif explicit.get('max_steps') is not None:
        values.pop('target_tokens', None)
    values.update(explicit)
    if not values.get('run_dir'):
        raise ValueError('--run-dir is required (CLI or config)')
    if mode != 'resume' and values.get('target_tokens') is not None and values.get('max_steps') is not None:
        raise ValueError('Specify target_tokens or max_steps, not both')
    run_training(values, mode=mode)


def _inspect(argv):
    parser = argparse.ArgumentParser(prog='qwen-pretrain inspect')
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args(argv)
    from .checkpoints import Progress, load_checkpoint, optimizer_lr, saved_schedule
    state = load_checkpoint(args.checkpoint)
    progress = Progress.from_checkpoint(state)
    print(json.dumps({'format_version': state.get('format_version', 1),
                      'step': progress.step, 'tokens_processed': progress.tokens_processed,
                      'current_lr': optimizer_lr(state), 'lr_schedule': saved_schedule(state),
                      'args': state['args']}, indent=2, ensure_ascii=False))


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='qwen-pretrain', description='Single-GPU Qwen pretraining tools')
    commands = parser.add_subparsers(dest='command', required=True)
    training = {}
    for name, help_text in [('train', 'Start from random model initialization'),
                             ('extend', 'Continue a full checkpoint with a new endpoint and LR stage'),
                             ('resume', 'Resume a run with its saved endpoint and LR schedule')]:
        training[name] = commands.add_parser(name, help=help_text)
        _training_arguments(training[name], name)
    # Forward these complete argument lists to their operation's parser. This
    # keeps --help useful without importing Torch/Transformers/Arrow.
    for name, help_text in [('prepare', 'Prepare packed train/validation token files'),
                            ('generate', 'Generate from .pt or milestone .safetensors'),
                            ('evaluate', 'Evaluate permanent model milestones'),
                            ('inspect', 'Print metadata from a trusted full checkpoint')]:
        commands.add_parser(name, help=help_text, add_help=False)
    if argv and argv[0] in ('prepare', 'generate', 'evaluate', 'inspect'):
        name, rest = argv[0], argv[1:]
        try:
            if name == 'prepare':
                from .data.prepare import main as operation
            elif name == 'generate':
                from .generation import main as operation
            elif name == 'evaluate':
                from .evaluation import main as operation
            else:
                operation = _inspect
            operation(rest)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        return
    args = parser.parse_args(argv)
    try:
        _run_training(args, training[args.command], args.command)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
