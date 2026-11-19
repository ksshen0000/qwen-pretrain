"""Argument parsing for data preparation, independent of ML imports."""
import argparse
import os
from pathlib import Path

from ..config import token_count


def build_parser():
    ap = argparse.ArgumentParser(description='Prepare packed token data from Parquet')
    ap.add_argument('--parquet', type=Path, nargs='+', required=True)
    ap.add_argument('--model-dir', type=Path, default=Path('model'))
    ap.add_argument('--output-dir', type=Path, default=Path('prepared_parallel'))
    ap.add_argument('--max-tokens', type=token_count, default=10_000_000)
    ap.add_argument('--continue-from', type=Path,
                    help='Original prepared dataset: verify/skip its documents and copy its validation set')
    ap.add_argument('--additional-train-tokens', type=token_count,
                    help='Requested NEW training tokens; requires --continue-from; replaces --max-tokens')
    ap.add_argument('--val-fraction', type=float, default=0.01)
    ap.add_argument('--seq-len', type=int, default=2048,
                    help='Training window; each split must hold at least seq-len + 1 tokens')
    ap.add_argument('--batch-rows', type=int, default=128)
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 2),
                    help='number of CPU worker processes (default: min(8, CPUs))')
    ap.add_argument('--inflight', type=int, default=0,
                    help='max pending batches (default: 2 x workers)')
    ap.add_argument('--progress-every', type=int, default=5_000_000)
    ap.add_argument('--overwrite', action='store_true',
                    help='replace prior completed/partial output files')

    return ap
