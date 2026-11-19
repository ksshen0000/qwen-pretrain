"""Read-only packed token arrays and the original random-window sampler."""
import json
from pathlib import Path


def dataset_metadata(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
    sizes = {'uint16': 2, 'uint32': 4}
    if meta.get('dtype') not in sizes:
        raise ValueError('Prepared dtype must be uint16 or uint32')
    info = {'meta': meta}
    for split in ('train', 'val'):
        size = (directory / f'{split}.bin').stat().st_size
        expected = meta.get(f'{split}_tokens')
        if size % sizes[meta['dtype']] or (expected is not None and size != expected * sizes[meta['dtype']]):
            raise ValueError(f'{split}.bin size disagrees with meta.json')
        info[f'{split}_bytes'] = size
    return info


class TokenDataset:
    def __init__(self, directory, seq_len):
        import numpy as np
        self.directory = Path(directory)
        self.info = dataset_metadata(directory)
        self.meta = self.info['meta']
        self.arrays = {split: np.memmap(self.directory / f'{split}.bin', mode='r', dtype=self.meta['dtype'])
                       for split in ('train', 'val')}
        if min(map(len, self.arrays.values())) < seq_len + 1:
            raise ValueError('Data too small for the requested sequence length')


class BatchSampler:
    def __init__(self, dataset, config, device):
        import numpy as np
        self.dataset = dataset
        self.config = config
        self.device = device
        self.train_rng = np.random.default_rng(config.seed + 100)
        self.eval_rng = np.random.default_rng(config.seed + 200)

    def batch(self, split, rng=None):
        import numpy as np
        import torch
        raw = self.dataset.arrays[split]
        rng = rng if rng is not None else (self.train_rng if split == 'train' else self.eval_rng)
        # Preserve the old bounds and RNG consumption for checkpoint compatibility.
        starts = rng.integers(0, len(raw) - self.config.seq_len, size=self.config.batch_size)
        tokens = np.stack([raw[index:index + self.config.seq_len].astype(np.int64) for index in starts])
        return torch.from_numpy(tokens).to(self.device, non_blocking=True)

    def rng_state(self):
        return {'train_rng': self.train_rng.bit_generator.state, 'eval_rng': self.eval_rng.bit_generator.state}

    def restore(self, state):
        self.train_rng.bit_generator.state = state['train_rng']
        self.eval_rng.bit_generator.state = state['eval_rng']
