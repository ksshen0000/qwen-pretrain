"""Hugging Face export round trip on a tiny Qwen3 model (CPU, offline)."""
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import transformers
except ImportError:  # Pure-Python test environments skip these.
    torch = None


def tiny_model_dir(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config
    Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                tie_word_embeddings=True).save_pretrained(directory)
    vocab = {word: index for index, word in enumerate(['<unk>', '<eos>', 'a', 'b', 'c'])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                            eos_token='<eos>').save_pretrained(directory)


@unittest.skipIf(torch is None, 'torch/transformers are not installed')
class ExportTests(unittest.TestCase):
    def test_milestone_exports_and_reloads_identically(self):
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from qwen_pretrain.export import export
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            tiny_model_dir(root / 'model')
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root / 'model'))
            weights = {key: value.detach().to(torch.bfloat16).clone().contiguous()
                       for key, value in model.state_dict().items()}
            milestone = root / 'tokens_0000001000.safetensors'
            save_file(weights, str(milestone), metadata={'step': '7', 'tokens_processed': '1000'})

            output = root / 'exported'
            export(milestone, root / 'model', output)

            reloaded = AutoModelForCausalLM.from_pretrained(output)
            self.assertEqual(reloaded.dtype, torch.bfloat16)
            for key, value in reloaded.state_dict().items():
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
            self.assertIs(reloaded.lm_head.weight, reloaded.model.embed_tokens.weight)
            self.assertEqual(AutoTokenizer.from_pretrained(output)('a b c')['input_ids'], [2, 3, 4])
            info = json.loads((output / 'training_info.json').read_text())
            self.assertEqual((info['step'], info['tokens_processed']), (7, 1000))
            self.assertFalse(output.with_name('exported.tmp').exists())
            with self.assertRaises(ValueError):  # Never overwrite a finished export.
                export(milestone, root / 'model', output)


if __name__ == '__main__':
    unittest.main()
