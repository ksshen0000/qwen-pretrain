import unittest

from qwen_pretrain.schedules import cosine_lr, make_schedule, target_step


class ScheduleTests(unittest.TestCase):
    def test_fresh_warmup_and_decay(self):
        schedule = make_schedule(0, 100, 3e-4, 3e-5, 10)
        self.assertAlmostEqual(cosine_lr(1, schedule), 3e-5)
        self.assertAlmostEqual(cosine_lr(10, schedule), 3e-4)
        self.assertAlmostEqual(cosine_lr(100, schedule), 3e-5)

    def test_continuation_does_not_jump_to_old_peak(self):
        schedule = make_schedule(15259, 76294, 3e-5, 1e-5, start_lr=3e-5)
        self.assertAlmostEqual(cosine_lr(15259, schedule), 3e-5)
        self.assertLess(cosine_lr(15260, schedule), 3e-5)
        self.assertAlmostEqual(cosine_lr(76294, schedule), 1e-5)

    def test_rewarm_starts_from_actual_checkpoint_lr(self):
        schedule = make_schedule(15000, 76000, 1e-4, 1e-5, 200, 3e-5)
        self.assertAlmostEqual(cosine_lr(15000, schedule), 3e-5)
        self.assertAlmostEqual(cosine_lr(15100, schedule), 6.5e-5)
        self.assertAlmostEqual(cosine_lr(15200, schedule), 1e-4)
        self.assertAlmostEqual(cosine_lr(76000, schedule), 1e-5)

    def test_serialized_schedule_does_not_restart_on_resume(self):
        import json
        schedule = make_schedule(15000, 76000, 1e-4, 1e-5, 200, 3e-5)
        restored = json.loads(json.dumps(schedule))
        for step in (15001, 15100, 30000, 76000):
            self.assertEqual(cosine_lr(step, restored), cosine_lr(step, schedule))

    def test_cumulative_target_rounds_up_from_actual_tokens(self):
        start_step = 15259
        actual_tokens = start_step * 65536
        final_step = target_step(5_000_000_000, actual_tokens, start_step, 65536)
        self.assertEqual(final_step, 76294)
        self.assertGreaterEqual(final_step * 65536, 5_000_000_000)
        self.assertLess(final_step * 65536 - 5_000_000_000, 65536)

    def test_weights_only_with_changed_batch_counts_from_metadata(self):
        self.assertEqual(target_step(1100, 1000, 20, 64), 22)

    def test_invalid_endpoints_and_warmup_fail(self):
        for kwargs in (dict(end_step=10), dict(warmup_steps=90),
                       dict(min_lr=1.), dict(peak_lr=float('nan'))):
            params = dict(start_step=10, end_step=100, peak_lr=3e-4, min_lr=3e-5)
            params.update(kwargs)
            with self.assertRaises(ValueError):
                make_schedule(**params)
        with self.assertRaises(ValueError):
            target_step(1000, 1000, 20, 64)


if __name__ == '__main__':
    unittest.main()
