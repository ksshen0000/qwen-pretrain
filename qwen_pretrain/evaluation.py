"""Evaluate milestones using fixed validation windows, probes and greedy samples."""
import argparse
from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path

from .benchmarks import ProbeScorer
from .checkpoints import load_model_weights
from .generation import generate_text
from .io import atomic_write
from .modeling import build_model
from .prompts import FACTS, GRAMMAR, PROMPTS

FIELDS = ['milestone_tokens', 'step', 'actual_tokens', 'val_loss', 'ppl',
          'fact_accuracy', 'grammar_accuracy', 'samples_file']


def write_report(directory, rows):
    def write_csv(temporary):
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(directory / 'evaluation.csv', write_csv)
    lines = ['| Target Tokens | Step | Actual Tokens | Val Loss | PPL | Facts | Grammar |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['milestone_tokens']:,} | {row['step']:,} | {row['actual_tokens']:,} | "
                     f"{row['val_loss']:.4f} | {row['ppl']:.2f} | {row['fact_accuracy']:.1%} | "
                     f"{row['grammar_accuracy']:.1%} |")
    atomic_write(directory / 'evaluation.md', lambda temporary: temporary.write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='model')
    parser.add_argument('--data-dir', default='prepared')
    parser.add_argument('--milestone-dir', default='milestones')
    parser.add_argument('--output-dir', type=Path, default=Path('eval_results'))
    parser.add_argument('--eval-batches', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    if min(args.eval_batches, args.batch_size, args.max_new_tokens) < 1 or args.seq_len < 2:
        parser.error('Batch counts and generation length must be positive; seq-len must be >= 2')
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from .data.tokens import dataset_metadata
    snapshots = sorted(Path(args.milestone_dir).glob('tokens_*.safetensors'),
                       key=lambda item: int(item.stem.removeprefix('tokens_')))
    if not snapshots:
        parser.error('No milestone snapshots found')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = build_model(args.model_dir, args.device)
    meta = dataset_metadata(args.data_dir)['meta']
    val = np.memmap(Path(args.data_dir) / 'val.bin', mode='r', dtype=meta['dtype'])
    if len(val) <= args.seq_len:
        parser.error('Validation dataset too short')
    rng = np.random.default_rng(20261111)
    starts = rng.integers(0, len(val) - args.seq_len, size=(args.eval_batches, args.batch_size))
    scorer = ProbeScorer(model, tokenizer, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for snapshot in snapshots:
        print(f'[eval] loading {snapshot}', flush=True)
        weights, info = load_model_weights(snapshot)
        model.load_state_dict(weights, strict=True)
        del weights
        sidecar = snapshot.with_suffix('.json')
        if sidecar.is_file():
            info.update(json.loads(sidecar.read_text(encoding='utf-8')))
        # Old or interrupted snapshots still carry step/tokens in the file itself.
        target = int(snapshot.stem.removeprefix('tokens_'))
        losses = []
        with torch.inference_mode():
            for batch_starts in starts:
                tokens = np.stack([val[index:index + args.seq_len].astype(np.int64) for index in batch_starts])
                tokens = torch.from_numpy(tokens).to(args.device)
                autocast = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
                with autocast:
                    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
                losses.append(loss.float().item())
        val_loss = sum(losses) / len(losses)
        if not math.isfinite(val_loss):
            raise ValueError(f'Non-finite validation loss for {snapshot}')
        factual = sum(max(range(len(choices)), key=lambda index: scorer.continuation(prefix, choices[index])) == correct
                      for prefix, choices, correct in FACTS)
        grammar = sum(scorer.sentence(good) > scorer.sentence(bad) for good, bad in GRAMMAR)
        samples_file = args.output_dir / f'{snapshot.stem}_samples.txt'
        lines = []
        for prompt in PROMPTS:
            continuation = generate_text(model, tokenizer, prompt, args.device, args.max_new_tokens)
            lines.append(f'PROMPT: {prompt}\nOUTPUT: {continuation}\n')
        atomic_write(samples_file, lambda temporary: temporary.write_text('\n'.join(lines), encoding='utf-8'))
        row = dict(milestone_tokens=int(info.get('target_tokens', target)), step=int(info['step']),
                   actual_tokens=int(info['tokens_processed']), val_loss=round(val_loss, 5),
                   ppl=round(math.exp(min(val_loss, 50)), 3), fact_accuracy=round(factual / len(FACTS), 3),
                   grammar_accuracy=round(grammar / len(GRAMMAR), 3), samples_file=str(samples_file))
        rows.append(row)
        print(row, flush=True)
        write_report(args.output_dir, rows)
    print(f'[done] CSV: {args.output_dir / "evaluation.csv"}', flush=True)
    print('Note: 10 fact and 10 grammar items are illustrative probes, not a comprehensive benchmark.', flush=True)
