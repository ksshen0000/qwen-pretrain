"""Verify and exclude documents used by an original prepared dataset."""
from pathlib import Path


class ReplayVerifier:
    def __init__(self, directory, sources, dtype, vocab_size, eos_token_id, min_tokens):
        import numpy as np
        from .tokens import dataset_metadata
        self.directory = Path(directory).expanduser().resolve()
        self.meta = dataset_metadata(self.directory)['meta']
        if self.meta.get('continued_from'):
            raise ValueError('continue-from requires an original dataset, not an already continued dataset')
        if (np.dtype(self.meta['dtype']) != np.dtype(dtype)
                or self.meta['vocab_size'] != vocab_size or self.meta['eos_token_id'] != eos_token_id):
            raise ValueError('Original dtype/vocabulary/EOS do not match the selected tokenizer')
        previous = self.meta.get('source_paths_resolved', self.meta['source_files'])
        previous = [Path(item).resolve() for item in previous]
        if not previous or [Path(item).resolve() for item in sources][:len(previous)] != previous:
            raise ValueError('Parquet inputs must begin with the original source files in their original order')
        self.data = {split: np.memmap(self.directory / f'{split}.bin', mode='r', dtype=dtype)
                     for split in ('train', 'val')}
        if min(map(len, self.data.values())) < min_tokens:
            raise ValueError(f'Original data is too short for {min_tokens - 1}-token windows')
        self.counts = {'train': 0, 'val': 0}
        self.dtype = dtype

    @property
    def budget(self):
        return sum(map(len, self.data.values()))

    @property
    def complete(self):
        return all(self.counts[split] == len(self.data[split]) for split in ('val', 'train'))

    def skip(self, ids):
        """Verify one complete original document, then discard its entire suffix."""
        import numpy as np
        if self.complete:
            return False
        split = 'val' if self.counts['val'] < len(self.data['val']) else 'train'
        offset = self.counts[split]
        available = len(self.data[split]) - offset
        array = np.asarray(ids[:available], dtype=self.dtype)
        if not np.array_equal(array, self.data[split][offset:offset + len(array)]):
            raise ValueError(f'Original {split} token mismatch in document starting at token {offset:,}; '
                             'check tokenizer, source order and original splitting logic')
        self.counts[split] += len(array)
        return True

    def metadata(self):
        return {'continued_from': str(self.directory),
                'skipped_original_tokens': dict(self.counts),
                'validation_copied_from': str(self.directory / 'val.bin')}
