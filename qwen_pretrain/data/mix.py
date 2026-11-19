"""Concatenate prepared datasets into one mixture (standard library only).

Training samples windows uniformly from train.bin, so each source's share of
the mixed train tokens is its sampling probability. Set the shares with each
source's `prepare --max-tokens`, then mix:

    python -m qwen_pretrain mix --source prepared_edu prepared_wiki --output-dir prepared_mix
"""
import argparse
import os
from pathlib import Path
import shutil

from ..io import directory_lock, write_json
from .tokens import dataset_metadata

_TOKENIZER_KEYS = ('dtype', 'vocab_size', 'eos_token_id')


def _read_sources(sources):
    infos = []
    for directory in sources:
        info = dataset_metadata(directory)  # Also checks .bin sizes against meta.json.
        if not info['meta'].get('preparation_complete', True):
            raise ValueError(f'{directory} is an incomplete preparation (source exhausted early)')
        infos.append(info)
    first = infos[0]['meta']
    for directory, info in zip(sources, infos):
        for key in _TOKENIZER_KEYS:
            if info['meta'].get(key) != first.get(key):
                raise ValueError(f'{directory} differs in {key}: {info["meta"].get(key)} != {first.get(key)}')
    return infos


def mix(sources, output_dir):
    sources = [Path(item) for item in sources]
    output_dir = Path(output_dir)
    if len(sources) < 2:
        raise ValueError('Mixing needs at least two --source directories')
    resolved = [item.resolve() for item in sources]
    if len(set(resolved)) != len(resolved):
        raise ValueError('The same source was supplied more than once')
    if output_dir.resolve() in resolved:
        raise ValueError('Output must differ from every source')
    infos = _read_sources(sources)
    names = ('train.bin', 'val.bin', 'meta.json')
    with directory_lock(output_dir, '.mix.lock'):
        if any((output_dir / name).exists() for name in names):
            raise ValueError(f'{output_dir} already holds a dataset; choose a new output directory')
        width = int(infos[0]['meta']['dtype'][4:]) // 8
        totals = {}
        for split in ('train', 'val'):
            part = output_dir / f'{split}.bin.part'
            with part.open('wb') as stream:
                for directory in sources:
                    with (directory / f'{split}.bin').open('rb') as source:
                        shutil.copyfileobj(source, stream, 16 << 20)
                stream.flush()
                os.fsync(stream.fileno())
            totals[split] = sum(info[f'{split}_bytes'] for info in infos) // width
            if part.stat().st_size != totals[split] * width:
                raise ValueError(f'{split}.bin.part has an unexpected size; output not published')
        components = [{'path': str(path), 'train_tokens': info['meta']['train_tokens'],
                       'val_tokens': info['meta']['val_tokens'],
                       'train_share': round(info['meta']['train_tokens'] / totals['train'], 6),
                       'source_files': info['meta'].get('source_files', [])}
                      for path, info in zip(resolved, infos)]
        for split in ('train', 'val'):
            os.replace(output_dir / f'{split}.bin.part', output_dir / f'{split}.bin')
        meta = {key: infos[0]['meta'][key] for key in _TOKENIZER_KEYS}
        meta.update(train_tokens=totals['train'], val_tokens=totals['val'],
                    preparation_complete=True, mixed_from=components)
        write_json(output_dir / 'meta.json', meta)  # Published last: marks the mixture complete.
    for item in components:
        print(f'[mix] {item["train_share"]:7.2%}  train={item["train_tokens"]:,}  '
              f'val={item["val_tokens"]:,}  {item["path"]}', flush=True)
    print(f'[done] {output_dir}: train={totals["train"]:,} val={totals["val"]:,}', flush=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(prog='qwen-pretrain mix', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, nargs='+', required=True, help='Prepared dataset directories')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        mix(args.source, args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
