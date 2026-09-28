"""Thread-safety and hang-recovery tests for the persistent ExifTool singleton.

Covers review findings F26 (concurrent loader threads must not consume each
other's response — cross-photo EXIF corruption) and F27 (the response loop
must not hang forever on a dead or stalled process).
"""

import collections
import os
import threading
import time

import pytest

import exiftool.exiftool_batch as exiftool_batch
from exiftool.exiftool_batch import ExifToolBatch, parse_exif_data

_MODEL_BY_PATH = {'A': 'CAM_A', 'B': 'CAM_B'}


class _CrossStdout:
    """In-memory stdout: readline() pops from a single shared line queue."""

    def __init__(self):
        self._lines = collections.deque()
        self._cv = threading.Condition()

    def push(self, lines):
        with self._cv:
            self._lines.extend(lines)
            self._cv.notify_all()

    def readline(self):
        with self._cv:
            while not self._lines:
                self._cv.wait()
            return self._lines.popleft()


class _CrossStdin:
    """Fake stdin that enqueues the response for the requested path.

    The first command to arrive enqueues its response, signals ``event`` and
    then sleeps ``delay``. That widens the window in which a second, unlocked
    thread can barge in and read the first thread's response (the F26 race).
    """

    def __init__(self, stdout, event, delay):
        self._stdout = stdout
        self._event = event
        self._delay = delay
        self._first = True
        self._flag_lock = threading.Lock()

    def write(self, data):
        if '-execute' not in data:
            return
        parts = data.split('\n')
        path = parts[parts.index('-execute') - 1]
        model = _MODEL_BY_PATH[path]
        with self._flag_lock:
            first = self._first
            self._first = False
        self._stdout.push([
            f'[{{"SourceFile": "{path}", "Model": "{model}"}}]\n',
            '{ready}\n',
        ])
        if first:
            self._event.set()
            time.sleep(self._delay)

    def flush(self):
        pass


class _FakeProc:
    def __init__(self, stdin, stdout):
        self.stdin = stdin
        self.stdout = stdout

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class _NullStdin:
    def write(self, data):
        pass

    def flush(self):
        pass


class _PipeProc:
    """Process backed by a real OS pipe so kill() forces an stdout EOF."""

    def __init__(self, stdout, stdin, write_fd=None):
        self.stdout = stdout
        self.stdin = stdin
        self._write_fd = write_fd
        self._killed = False

    def poll(self):
        return 1 if self._killed else None

    def kill(self):
        self._killed = True
        if self._write_fd is not None:
            try:
                os.close(self._write_fd)
            except OSError:
                pass

    def wait(self, timeout=None):
        return 1


def _make_instance(monkeypatch):
    monkeypatch.setattr(ExifToolBatch, '_start_process', lambda self: None)
    return ExifToolBatch()


def test_unexecutable_exiftool_is_treated_as_optional(monkeypatch):
    """A PATH hit that is a directory raises EACCES, not FileNotFoundError."""
    monkeypatch.setattr(
        exiftool_batch.subprocess,
        'Popen',
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, 'denied')),
    )

    inst = ExifToolBatch()

    assert inst.process is None


def test_batch_skips_subprocess_when_optional_exiftool_is_unavailable(monkeypatch):
    inst = _make_instance(monkeypatch)
    monkeypatch.setattr(
        exiftool_batch.subprocess,
        'run',
        lambda *args, **kwargs: pytest.fail('unavailable exiftool must not be retried'),
    )

    assert inst.get_metadata_batch(['/photos/a.jpg']) == {}


def test_get_metadata_serializes_concurrent_threads(monkeypatch):
    """F26: two interleaved threads must each get their own photo's metadata."""
    stdout = _CrossStdout()
    first_write = threading.Event()
    stdin = _CrossStdin(stdout, first_write, delay=0.3)
    inst = _make_instance(monkeypatch)
    inst.process = _FakeProc(stdin, stdout)

    results = {}

    def worker(path):
        results[path] = inst.get_metadata(path)

    t_a = threading.Thread(target=worker, args=('A',))
    t_a.start()
    assert first_write.wait(timeout=5)
    t_b = threading.Thread(target=worker, args=('B',))
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert results['A'].get('Model') == 'CAM_A'
    assert results['B'].get('Model') == 'CAM_B'


def test_get_metadata_returns_on_dead_process(monkeypatch):
    """F27: an EOF (dead process) must return promptly, not busy-loop."""
    r_fd, w_fd = os.pipe()
    stdout = os.fdopen(r_fd, 'r')
    os.close(w_fd)  # immediate EOF on the read end
    inst = _make_instance(monkeypatch)
    inst.process = _PipeProc(stdout, _NullStdin())

    result = {}
    done = threading.Event()

    def run():
        result['v'] = inst.get_metadata('X')
        done.set()

    threading.Thread(target=run, daemon=True).start()
    try:
        assert done.wait(timeout=5), "get_metadata hung on a dead process"
        assert result['v'] == {}
    finally:
        stdout.close()


def test_get_metadata_returns_on_stalled_process(monkeypatch):
    """F27: a stalled process (no output) must be bounded by the deadline."""
    monkeypatch.setattr(exiftool_batch, 'STAY_OPEN_TIMEOUT_SECONDS', 0.5, raising=False)
    r_fd, w_fd = os.pipe()
    stdout = os.fdopen(r_fd, 'r')
    inst = _make_instance(monkeypatch)
    inst.process = _PipeProc(stdout, _NullStdin(), write_fd=w_fd)

    result = {}
    done = threading.Event()

    def run():
        result['v'] = inst.get_metadata('X')
        done.set()

    threading.Thread(target=run, daemon=True).start()
    try:
        assert done.wait(timeout=5), "get_metadata hung on a stalled process"
        assert result['v'] == {}
    finally:
        stdout.close()
        try:
            os.close(w_fd)
        except OSError:
            pass


class TestParseExifDataIso:
    """#142: ISO must land as int, since photos.iso is INTEGER-affinity and a
    fractional REAL (e.g. an Immich-sourced exposure index) 500s the gallery's
    response model.
    """

    def test_fractional_iso_rounds_to_int(self):
        result = parse_exif_data({'ISO': 63.4525478595867})
        assert result['iso'] == 63
        assert isinstance(result['iso'], int)

    def test_integer_float_iso_becomes_int(self):
        result = parse_exif_data({'ISO': 400})
        assert result['iso'] == 400
        assert isinstance(result['iso'], int)

    def test_numeric_string_iso_becomes_int(self):
        result = parse_exif_data({'ISO': '400'})
        assert result['iso'] == 400
        assert isinstance(result['iso'], int)

    def test_oversized_iso_is_none_rather_than_an_unbindable_int(self):
        """A REAL survives an INTEGER column two ways: a fractional part, or
        not fitting int64. ``round(1e20)`` is a Python int sqlite3 refuses to
        bind (``OverflowError``), and this INSERT is inside a 50-photo batch
        with no guard above it -- so an unbounded round would trade one odd
        value for a lost batch and a dead scan.
        """
        result = parse_exif_data({'ISO': 1e20})
        assert result['iso'] is None

    def test_unparseable_iso_is_none(self):
        result = parse_exif_data({'ISO': 'Auto'})
        assert result['iso'] is None

    def test_none_iso_is_none(self):
        result = parse_exif_data({'ISO': None})
        assert result['iso'] is None

    def test_absent_iso_is_none(self):
        result = parse_exif_data({})
        assert result['iso'] is None

    def test_f_stop_and_focal_length_stay_float(self):
        result = parse_exif_data({'Aperture': 2.8, 'FocalLength': 50})
        assert result['f_stop'] == 2.8
        assert isinstance(result['f_stop'], float)
        assert isinstance(result['focal_length'], float)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
