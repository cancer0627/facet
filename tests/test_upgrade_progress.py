"""验证升级进度跨进程传递，以及异常时保留实际计数。"""

import json
import os
import subprocess
import sys

import pytest

from processing.upgrade_progress import (
    PROGRESS_ENV, PhotoProgress, UpgradeProgressMonitor,
)


def test_child_progress_is_visible_while_output_goes_to_log(tmp_path):
    state = tmp_path / 'progress.json'
    observed = []
    with (tmp_path / 'detail.log').open('w') as log:
        with UpgradeProgressMonitor(state, 2, 'Quality') as monitor:
            original_update = monitor.bar.update

            def record_update(count):
                original_update(count)
                observed.append(monitor.bar.n)

            monitor.bar.update = record_update
            result = subprocess.run(
                [sys.executable, '-c',
                 'import time\n'
                 'from processing.upgrade_progress import PhotoProgress\n'
                 'with PhotoProgress(total=2, desc="Model A") as p:\n'
                 '    p.update()\n'
                 '    time.sleep(0.5)\n'
                 '    p.update()\n'],
                env=dict(os.environ, **{PROGRESS_ENV: str(state)}),
                stdout=log, stderr=subprocess.STDOUT,
            )
    assert result.returncode == 0
    assert 1 in observed
    assert observed[-1] == 2


def test_interrupted_iteration_does_not_claim_completion(tmp_path, monkeypatch):
    state = tmp_path / 'progress.json'
    monkeypatch.setenv(PROGRESS_ENV, str(state))
    with pytest.raises(RuntimeError):
        with PhotoProgress(total=2, desc='Quality') as progress:
            progress.update()
            raise RuntimeError('failed batch')
    assert json.loads(state.read_text())['current'] == 1


def test_monitor_resets_for_next_model(tmp_path, capsys):
    state = tmp_path / 'progress.json'
    with UpgradeProgressMonitor(state, 2, 'Quality') as monitor:
        for desc, current, total in [('Model A', 2, 2), ('Model B', 0, 1)]:
            state.write_text(json.dumps(dict(desc=desc, current=current, total=total)))
            monitor._refresh()
        assert monitor.bar.n == 0
        assert monitor.bar.total == 1
    assert 'Model B' in capsys.readouterr().err
