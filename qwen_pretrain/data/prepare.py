"""Bounded parallel tokenization with ordered writes and verified continuation."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import time

from ..io import directory_lock, write_json
from .prepare_cli import build_parser
from .replay import ReplayVerifier
from .tokenize import doc_batches, init_worker, tokenize_docs


@dataclass
class PrepareConfig:
    parquet: list[Path]
    model_dir: Path = Path('model')
    output_dir: Path = Path('prepared_parallel')
    max_tokens: int = 10_000_000
    continue_from: Path | None = None
    additional_train_tokens: int | None = None
    val_fraction: float = .01
    seq_len: int = 2048
    batch_rows: int = 128
    workers: int = 8
    inflight: int = 0
    progress_every: int = 5_000_000
    overwrite: bool = False

    @property
    def min_tokens(self):
        return self.seq_len + 1

    def validate(self):
        if self.max_tokens <= 0 or not 0 < self.val_fraction < 1:
            raise ValueError('max-tokens must be positive; val-fraction must be between 0 and 1')
        if bool(self.continue_from) != (self.additional_train_tokens is not None):
            raise ValueError('continue-from and additional-train-tokens must be used together')
        if self.seq_len < 2:
            raise ValueError('seq-len must be >= 2')
        if self.additional_train_tokens is not None and self.additional_train_tokens < self.min_tokens:
            raise ValueError(f'additional-train-tokens must be >= {self.min_tokens} (seq-len + 1)')
        if min(self.workers, self.batch_rows, self.progress_every) < 1 or self.inflight < 0:
            raise ValueError('workers/batch-rows/progress-every must be positive; inflight must be nonnegative')
        if len({item.resolve() for item in self.parquet}) != len(self.parquet):
            raise ValueError('The same Parquet file was supplied more than once, possibly through symlinks')
        if self.continue_from and self.output_dir.resolve() == self.continue_from.resolve():
            raise ValueError('Output must differ from the original dataset, including symlink targets')


class DatasetWriter:
    """Publish meta.json last, so interrupted preparation never looks complete."""
    def __init__(self, directory, overwrite, original=None):
        self.directory = directory
        self.parts = {split: directory / f'{split}.bin.part' for split in ('train', 'val')}
        managed = [directory / name for name in ('train.bin', 'val.bin', 'meta.json', 'meta.json.part')]
        managed.extend(self.parts.values())
        for item in managed:
            if item.is_symlink():
                raise ValueError('Output files must not be symlinks; choose a new output directory')
            if item.exists() and original and any(item.samefile(original / f'{split}.bin') for split in ('train', 'val')):
                raise ValueError('Output must not reuse original binary files, including hard links')
        present = [item for item in managed if item.exists()]
        if present and not overwrite:
            raise ValueError('Output exists; choose a new output directory or explicitly pass --overwrite')
        if overwrite:
            # Remove publication marker first, before touching any binary files.
            (directory / 'meta.json').unlink(missing_ok=True)
            for item in managed:
                item.unlink(missing_ok=True)

    def publish(self, metadata):
        for split, part in self.parts.items():
            os.replace(part, self.directory / f'{split}.bin')
        write_json(self.directory / 'meta.json', metadata)


def prepare(config):
    config.validate()
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    for item in config.parquet:
        if not item.is_file() or 'text' not in pq.ParquetFile(item).schema.names:
            raise ValueError(f'Missing Parquet file or text column: {item}')
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
    eos, vocab_size = tokenizer.eos_token_id, len(tokenizer)
    if eos is None:
        raise ValueError('Tokenizer has no EOS token')
    del tokenizer
    dtype = np.dtype(np.uint32 if vocab_size > np.iinfo(np.uint16).max else np.uint16)
    replay = (ReplayVerifier(config.continue_from, config.parquet, dtype, vocab_size, eos, config.min_tokens)
              if config.continue_from else None)
    val_budget = replay.meta['val_tokens'] if replay else max(config.min_tokens, int(config.max_tokens * config.val_fraction))
    train_budget = config.additional_train_tokens if replay else config.max_tokens - val_budget
    if train_budget < config.min_tokens:
        raise ValueError(f'Token budget too small for {config.seq_len}-token training windows')
    with directory_lock(config.output_dir, '.prepare.lock'):
        writer = DatasetWriter(config.output_dir, config.overwrite, replay.directory if replay else None)
        counts = _write_tokens(config, writer, replay, dtype, train_budget, val_budget)
        metadata = {'dtype': str(dtype), 'vocab_size': vocab_size, 'eos_token_id': int(eos),
                    'train_tokens': counts['train'], 'val_tokens': counts['val'],
                    'source_files': [str(item) for item in config.parquet],
                    'source_paths_resolved': [str(item.resolve()) for item in config.parquet],
                    'requested_train_tokens': train_budget,
                    'preparation_complete': counts['train'] == train_budget}
        if replay:
            metadata.update(replay.metadata())
        writer.publish(metadata)
        print('[done] ' + json.dumps(metadata, ensure_ascii=False), flush=True)
        if counts['train'] < train_budget:
            print(f'[warning] source exhausted: train={counts["train"]:,} < {train_budget:,}; '
                  'dataset published with its actual size; further training will resample it', flush=True)


def _write_tokens(config, writer, replay, dtype, train_budget, val_budget):
    import numpy as np
    counts = {'train': 0, 'val': 0}
    inflight = config.inflight or config.workers * 2
    next_report = config.progress_every
    started = time.monotonic()
    replay_budget = replay.budget if replay else 0
    print(f'[start] workers={config.workers} inflight={inflight} target_train={train_budget:,} '
          f'val={val_budget:,} replay={replay_budget:,} dtype={dtype}', flush=True)
    try:
        with writer.parts['train'].open('wb') as train_stream, writer.parts['val'].open('wb') as val_stream:
            if replay:
                with (replay.directory / 'val.bin').open('rb') as original:
                    shutil.copyfileobj(original, val_stream)
                counts['val'] = val_budget
            with ProcessPoolExecutor(max_workers=config.workers, initializer=init_worker,
                                     initargs=(str(config.model_dir),)) as pool:
                batches = iter(doc_batches(config.parquet, config.batch_rows))
                pending = deque()
                exhausted = finished = False
                while not finished:
                    while len(pending) < inflight and not exhausted:
                        try:
                            docs = next(batches)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.append(pool.submit(tokenize_docs, docs))
                    if not pending:
                        break
                    for ids in pending.popleft().result():
                        if not ids or (replay and replay.skip(ids)):
                            continue
                        split = 'val' if counts['val'] < val_budget else 'train'
                        available = (val_budget if split == 'val' else train_budget) - counts[split]
                        array = np.asarray(ids[:available], dtype=dtype)
                        (val_stream if split == 'val' else train_stream).write(array.tobytes())
                        counts[split] += len(array)
                        if counts['val'] >= val_budget and counts['train'] >= train_budget:
                            finished = True
                            break
                    verified = sum(replay.counts.values()) if replay else 0
                    work_done = sum(counts.values()) + verified
                    if work_done >= next_report or finished:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(f'[progress] new_train={counts["train"]:,}/{train_budget:,} '
                              f'replayed={verified:,}/{replay_budget:,} '
                              f'avg={work_done/elapsed:,.0f} tok/s elapsed={elapsed/60:.1f} min', flush=True)
                        next_report = work_done + config.progress_every
                for future in pending:
                    future.cancel()
            for stream in (train_stream, val_stream):
                stream.flush()
                os.fsync(stream.fileno())
        if replay and not replay.complete:
            raise ValueError('Source exhausted before all original tokens were verified; output not published')
        if min(counts.values()) < config.min_tokens:
            raise ValueError(f'Not enough data: actual new training tokens={counts["train"]:,}')
        return counts
    except BaseException:
        print('[interrupted] Incomplete preparation retained; restart using a new output or --overwrite. '
              'This command does not resume tokenization.', flush=True)
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prepare(PrepareConfig(**vars(args)))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
