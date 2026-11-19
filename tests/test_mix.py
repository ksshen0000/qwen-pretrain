"""Mixing prepared datasets (standard library only)."""
from array import array
import json
from pathlib import Path
import tempfile
import unittest

from qwen_pretrain.data.mix import mix


def prepared(directory, train, val, vocab_size=151669, complete=True):
    directory.mkdir(parents=True)
    (directory / 'train.bin').write_bytes(array('I', train).tobytes())
    (directory / 'val.bin').write_bytes(array('I', val).tobytes())
    meta = {'dtype': 'uint32', 'vocab_size': vocab_size, 'eos_token_id': 151643,
            'train_tokens': len(train), 'val_tokens': len(val), 'preparation_complete': complete,
            'source_files': [f'{directory.name}.parquet']}
    (directory / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return directory


def tokens(path):
    values = array('I')
    values.frombytes(path.read_bytes())
    return list(values)


class MixTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def test_concatenates_splits_and_records_shares(self):
        a = prepared(self.root / 'a', [1, 2, 3], [7])
        b = prepared(self.root / 'b', [4], [8, 9])
        meta = mix([a, b], self.root / 'out')
        out = self.root / 'out'
        self.assertEqual(tokens(out / 'train.bin'), [1, 2, 3, 4])
        self.assertEqual(tokens(out / 'val.bin'), [7, 8, 9])
        self.assertEqual((meta['train_tokens'], meta['val_tokens']), (4, 3))
        self.assertEqual([item['train_share'] for item in meta['mixed_from']], [0.75, 0.25])
        self.assertEqual(json.loads((out / 'meta.json').read_text()), meta)
        self.assertFalse(list(out.glob('*.part')))

    def test_rejects_tokenizer_mismatch_incomplete_and_existing_output(self):
        a = prepared(self.root / 'a', [1, 2], [3])
        with self.assertRaisesRegex(ValueError, 'vocab_size'):
            mix([a, prepared(self.root / 'b', [1], [2], vocab_size=10)], self.root / 'out1')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            mix([a, prepared(self.root / 'c', [1], [2], complete=False)], self.root / 'out2')
        d = prepared(self.root / 'd', [5], [6])
        mix([a, d], self.root / 'out3')
        with self.assertRaisesRegex(ValueError, 'already holds'):
            mix([a, d], self.root / 'out3')
        with self.assertRaisesRegex(ValueError, 'differ from every source'):
            mix([a, d], a)


if __name__ == '__main__':
    unittest.main()
