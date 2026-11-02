"""验证配置目录能进入现有扫描流程，测试不加载模型或写入图片库。"""

import json
import sys
from types import SimpleNamespace

import pytest

import facet


@pytest.fixture
def scan_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(facet, '_run_scan', lambda args, resumed: calls.append(args))
    monkeypatch.setattr(facet, '_acquire_library_lock',
                        lambda *_args: SimpleNamespace(release=lambda: None))
    return calls


@pytest.mark.parametrize('source', ['argument', 'environment'])
def test_scan_config_uses_selected_config(tmp_path, monkeypatch, scan_calls, source):
    config = tmp_path / 'custom.json'
    directories = ['/photos/含 空格', '/photos/second']
    config.write_text(json.dumps({'viewer': {'scan_directories': directories + directories[:1]}}))
    argv = ['facet.py', '--scan-config']
    if source == 'argument':
        monkeypatch.delenv('FACET_CONFIG', raising=False)
        argv += ['--config', str(config)]
    else:
        monkeypatch.setenv('FACET_CONFIG', str(config))
    monkeypatch.setattr(sys, 'argv', argv)
    facet.main()
    assert len(scan_calls) == 1
    assert scan_calls[0].photo_paths == directories
    assert scan_calls[0].force is False


@pytest.mark.parametrize('directories', [[], '/photos', [None], [' ']])
def test_scan_config_rejects_empty_or_invalid_list(tmp_path, monkeypatch, scan_calls, directories):
    config = tmp_path / 'invalid.json'
    config.write_text(json.dumps({'viewer': {'scan_directories': directories}}))
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--scan-config', '--config', str(config)])
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code == 2
    assert not scan_calls


@pytest.mark.parametrize('extra', [['/explicit'], ['--resume'], ['--retry-failed']])
def test_scan_config_rejects_other_directory_sources(monkeypatch, scan_calls, extra):
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--scan-config', *extra])
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code == 2
    assert not scan_calls
