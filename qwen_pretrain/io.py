"""Durable writes, process locks and append-only metrics (standard library only)."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'))


@contextmanager
def directory_lock(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another process is using {directory} ({name})') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    # Never unlink a lock: waiters might still refer to its inode.


class MetricsWriter:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, event, **values):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': event, **values}, allow_nan=False) + '\n')
