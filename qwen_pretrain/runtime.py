"""Random-state persistence and graceful stops at optimizer-update boundaries."""
from contextlib import contextmanager
import random
import signal


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def capture_rng():
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {
        'python_rng': random.getstate(),
        'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch
    if 'python_rng' in state:
        random.setstate(state['python_rng'])
    if 'numpy_rng' in state:
        value = state['numpy_rng']
        np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch_rng'])
    if state.get('cuda_rng') and torch.cuda.is_available():
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Visible CUDA device count differs from the saved RNG state')
        torch.cuda.set_rng_state_all(state['cuda_rng'])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


class StopController:
    """Signal handlers only set a flag: never save a half-finished update."""
    def __init__(self):
        self.signal_number = None
        self.previous = {}

    @property
    def requested(self):
        return self.signal_number is not None

    def _request(self, number, frame):
        self.signal_number = number

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self.previous[number] = signal.signal(number, self._request)
        return self

    def __exit__(self, *exc):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
