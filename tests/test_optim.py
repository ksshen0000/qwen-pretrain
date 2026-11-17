"""Optimizer-state migration: run with `python -m unittest discover -s tests -t .`."""
import copy
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:  # Pure-Python test environments skip these.
    torch = None

from qwen_pretrain.optim import build_optimizer, load_optimizer_state


def tiny_model(dtype):
    torch.manual_seed(0)
    embedding = torch.nn.Embedding(11, 8)
    model = torch.nn.Sequential(embedding, torch.nn.LayerNorm(8), torch.nn.Linear(8, 11))
    model[2].weight = embedding.weight  # Tied, like Qwen3-0.6B.
    return model.to(dtype)


def take_step(model, optimizer):
    tokens = torch.tensor([[1, 2, 3, 4]])
    optimizer.zero_grad()
    model(tokens).float().square().mean().backward()
    optimizer.step()


@unittest.skipIf(torch is None, 'torch is not installed')
class OptimizerStateTests(unittest.TestCase):
    config = SimpleNamespace(lr=3e-4, weight_decay=0.1)

    def test_legacy_single_group_bf16_state_migrates_into_fp32_groups(self):
        old_model = tiny_model(torch.bfloat16)
        # The original script: one group over model.parameters(), BF16 parameters.
        old = torch.optim.AdamW(old_model.parameters(), lr=1.23e-4, weight_decay=0.1,
                                betas=(0.9, 0.95), eps=1e-8)
        take_step(old_model, old)
        saved = copy.deepcopy(old.state_dict())
        expected = {name: old.state[parameter] for name, parameter in old_model.named_parameters()}

        model = tiny_model(torch.float32)
        model.load_state_dict(old_model.state_dict())
        optimizer = build_optimizer(model, self.config, fused=False)
        load_optimizer_state(optimizer, model, saved)

        self.assertEqual([group['weight_decay'] for group in optimizer.param_groups], [0.1, 0.])
        self.assertEqual({group['lr'] for group in optimizer.param_groups}, {1.23e-4})
        for name, parameter in model.named_parameters():
            state = optimizer.state[parameter]
            self.assertEqual(state['exp_avg'].dtype, torch.float32, name)
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(state[key], expected[name][key].float(), rtol=0, atol=0)
            self.assertEqual(float(state['step']), float(expected[name]['step']))
        take_step(model, optimizer)  # The migrated state must be usable.

    def test_grouped_state_round_trips(self):
        model = tiny_model(torch.float32)
        optimizer = build_optimizer(model, self.config, fused=False)
        take_step(model, optimizer)
        saved = copy.deepcopy(optimizer.state_dict())
        restored = build_optimizer(model, self.config, fused=False)
        load_optimizer_state(restored, model, saved)
        for parameter in model.parameters():
            torch.testing.assert_close(restored.state[parameter]['exp_avg_sq'],
                                       optimizer.state[parameter]['exp_avg_sq'], rtol=0, atol=0)

    def test_parameter_count_mismatch_fails(self):
        model = tiny_model(torch.float32)
        optimizer = build_optimizer(model, self.config, fused=False)
        saved = optimizer.state_dict()
        saved['param_groups'][0]['params'] = saved['param_groups'][0]['params'][:-1]
        with self.assertRaises(ValueError):
            load_optimizer_state(optimizer, model, saved)


if __name__ == '__main__':
    unittest.main()
