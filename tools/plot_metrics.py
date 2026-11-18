#!/usr/bin/env python3
"""Plot a run's metrics.jsonl: loss, learning rate, gradient norm and throughput.

    pip install matplotlib
    python tools/plot_metrics.py runs/5b/metrics.jsonl --output logs/metrics.png
"""
import argparse
import json
from pathlib import Path

INK, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TRAIN, EVAL = '#2a78d6', '#eb6834'  # Categorical slots 1-2 (validated pair).


def read_metrics(path):
    train, evals = [], []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                (evals if record.get('event') == 'eval' else train).append(record)
    if not train:
        raise SystemExit(f'No training records in {path}')
    return train, evals


def ema(values, alpha):
    out, average = [], None
    for value in values:
        average = value if average is None else alpha * value + (1 - alpha) * average
        out.append(average)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path, default=Path('metrics.png'))
    parser.add_argument('--smoothing', type=float, default=0.02, help='EMA weight of the newest point')
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    train, evals = read_metrics(args.metrics)
    x = [r['tokens'] / 1e9 for r in train]
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK_MUTED,
                         'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
                         'axes.edgecolor': GRID, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    first, last = train[0], train[-1]
    figure.suptitle(f'Pretraining metrics · steps {first["step"]:,}–{last["step"]:,} · '
                    f'{first["tokens"] / 1e9:.2f}B–{last["tokens"] / 1e9:.2f}B tokens',
                    x=0.01, ha='left', fontsize=13, color=INK)

    def style(axis, title):
        axis.set_title(title, loc='left', fontsize=11, color=INK)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)

    def faint_and_smooth(axis, values, color):
        axis.plot(x, values, color=color, linewidth=0.6, alpha=0.18)
        smooth = ema(values, args.smoothing)
        axis.plot(x, smooth, color=color, linewidth=2)
        return smooth

    loss = axes[0][0]
    smooth = faint_and_smooth(loss, [r['loss'] for r in train], TRAIN)
    loss.plot([], [], color=TRAIN, linewidth=2, label='Train (EMA)')
    if evals:
        ex, ey = [r['tokens'] / 1e9 for r in evals], [r['loss'] for r in evals]
        loss.plot(ex, ey, color=EVAL, linewidth=2, marker='o', markersize=4,
                  markeredgecolor=SURFACE, markeredgewidth=1, label='Validation')
        loss.annotate(f'val {ey[-1]:.3f}', (ex[-1], ey[-1]), xytext=(6, 8),
                      textcoords='offset points', color=INK, fontsize=9)
    loss.annotate(f'train {smooth[-1]:.3f}', (x[-1], smooth[-1]), xytext=(6, -14),
                  textcoords='offset points', color=INK, fontsize=9)
    raw = [r['loss'] for r in train]
    loss.set_ylim(min(raw) - 0.05, sorted(raw)[int(len(raw) * 0.995)] + 0.1)  # Ignore rare spikes.
    loss.legend(frameon=False, loc='upper right')
    style(loss, 'Loss')

    lr = axes[0][1]
    lr.plot(x, [r['lr'] for r in train], color=TRAIN, linewidth=2)
    lr.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1e}' if value else '0'))
    style(lr, 'Learning rate')

    grad = axes[1][0]
    norms = [r['grad_norm'] for r in train if 'grad_norm' in r]
    if norms:
        x_grad = [r['tokens'] / 1e9 for r in train if 'grad_norm' in r]
        grad.plot(x_grad, norms, color=TRAIN, linewidth=0.6, alpha=0.18)
        grad.plot(x_grad, ema(norms, args.smoothing), color=TRAIN, linewidth=2)
        grad.set_ylim(0, max(1.05, sorted(norms)[int(len(norms) * 0.995)] * 1.1))
        grad.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle='--')
        grad.annotate('clip threshold 1.0', (x_grad[0], 1.0), xytext=(4, 4),
                      textcoords='offset points', color=INK_MUTED, fontsize=9)
    style(grad, 'Gradient norm (before clipping)')

    speed = axes[1][1]
    rates = [(r['tokens'] / 1e9, r['avg_tokens_per_second'] / 1e3) for r in train if 'avg_tokens_per_second' in r]
    if rates:
        speed.plot(*zip(*rates), color=TRAIN, linewidth=2)
        speed.set_ylim(0, max(rate for _, rate in rates) * 1.15)
    speed.set_ylabel('thousand tokens / s')
    style(speed, 'Average throughput since (re)start')

    for axis in axes[1]:
        axis.set_xlabel('Cumulative tokens (billions)')
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
