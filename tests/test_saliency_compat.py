"""Verify saliency dependency isolation, mask transfer, and worker shutdown."""
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from models import saliency_compat as compat
from models.saliency_scorer import SaliencyScorer


@pytest.mark.parametrize('torch_version,transformers_version,expected', [
    ('2.2.2', '5.15.1', True), ('2.5.0', '5.15.1', False),
    ('2.2.2', '4.57.6', False),
])
def test_select_compatibility_only_when_needed(monkeypatch, torch_version, transformers_version, expected):
    monkeypatch.delenv('FACET_SALIENCY_WORKER', raising=False)
    monkeypatch.setattr(compat.importlib.metadata, 'version',
                        lambda package: torch_version if package == 'torch' else transformers_version)
    assert compat.needs_compat_worker() is expected
    monkeypatch.setenv('FACET_SALIENCY_WORKER', '1')
    assert not compat.needs_compat_worker()


def test_scorer_routes_masks_and_unloads_worker():
    with mock.patch.object(compat, 'needs_compat_worker', return_value=True), \
            mock.patch.object(compat, 'SaliencyWorker') as factory:
        worker = factory.return_value
        scorer = SaliencyScorer(device='cpu')
        scorer.load()
        scorer.load()
        factory.assert_called_once()
        image = Image.new('RGB', (8, 8))
        binary = np.ones((8, 8), dtype=np.uint8) * 255
        worker.predict.return_value = [binary]
        assert np.array_equal(scorer.get_saliency_mask(image), binary)
        soft = np.full((8, 8), .5, dtype=np.float32)
        worker.predict.return_value = [soft]
        assert np.array_equal(scorer.get_saliency_soft(image), soft)
        scorer.unload()
        worker.close.assert_called_once()
        assert not scorer._loaded


def test_worker_protocol_and_cleanup(monkeypatch):
    original_popen = subprocess.Popen
    captured = []
    server = '''
import sys,json,numpy as np
from PIL import Image
print(json.dumps(dict(ok=True)),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    arrays=[]
    for path in request['paths']:
        with Image.open(path) as image:
            w,h=image.size
        arrays.append(np.full((h,w),.5 if request['soft'] else 255))
    np.savez(request['output'],*arrays)
    print(json.dumps(dict(ok=True)),flush=True)
'''

    def spawn(command, **kwargs):
        captured.append((command, kwargs['env']))
        return original_popen([sys.executable, '-c', server], **kwargs)

    monkeypatch.setattr(compat, 'compat_available', lambda: True)
    monkeypatch.setattr(compat.subprocess, 'Popen', spawn)
    worker = compat.SaliencyWorker({})
    try:
        masks = worker.predict([Image.new('RGB', (8, 6)), Image.new('RGB', (5, 4))])
        assert [m.shape for m in masks] == [(6, 8), (4, 5)]
        assert np.all(masks[0] == 255)
        soft = worker.predict([Image.new('RGB', (3, 2))], soft=True)
        assert np.all(soft[0] == .5)
        assert captured[0][1]['PYTHONPATH'].split(compat.os.pathsep)[0] == str(compat.compat_directory())
    finally:
        worker.close()
    assert worker.process.poll() == 0


def test_compatibility_requires_explicit_install(monkeypatch):
    monkeypatch.setattr(compat, 'compat_available', lambda: False)
    with pytest.raises(ImportError, match='install_saliency_compat.py'):
        compat.SaliencyWorker({})


@pytest.mark.parametrize('bbox', [None, [0, 0, 1, 1]])
def test_saliency_command_writes_target_without_upgrade_branch(tmp_path, monkeypatch, bbox):
    import io
    import facet
    from db import init_database, get_connection

    db = str(tmp_path / 'saliency.db')
    init_database(db)
    image = io.BytesIO()
    Image.new('RGB', (32, 32)).save(image, format='JPEG')
    with get_connection(db) as conn:
        for path in ('/target.jpg', '/other.jpg'):
            conn.execute('INSERT INTO photos (path,filename,thumbnail) VALUES (?,?,?)',
                         (path, path, image.getvalue()))
        conn.commit()
    monkeypatch.setattr(compat, 'needs_compat_worker', lambda: False)
    loaded = []
    monkeypatch.setattr(SaliencyScorer, 'load', lambda self: loaded.append(True))
    monkeypatch.setattr(SaliencyScorer, 'score_batch', lambda self, images, cv_images: [
        dict(subject_sharpness=7, subject_bbox=bbox) for _ in images])
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--recompute-saliency', '--db', db,
                                     '--upgrade-photo', '/target.jpg'])
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code in (None, 0)
    with pytest.raises(SystemExit):
        facet.main()
    assert len(loaded) == 1  # No subject detected still counts as processed; reruns do not reload the model.
    with get_connection(db) as conn:
        assert conn.execute("SELECT subject_sharpness FROM photos WHERE path='/target.jpg'").fetchone()[0] == 7
        assert conn.execute("SELECT subject_sharpness FROM photos WHERE path='/other.jpg'").fetchone()[0] is None


def test_cpu_recompute_commits_each_photo_before_next_inference(tmp_path, monkeypatch):
    """下一张中断时，已完成的照片仍能被其他连接读到，重跑会跳过它。"""
    import io
    import facet
    from db import init_database, get_connection

    db = str(tmp_path / 'resume.db')
    init_database(db)
    image = io.BytesIO()
    Image.new('RGB', (32, 32)).save(image, format='JPEG')
    with get_connection(db) as conn:
        for path in ('/first.jpg', '/second.jpg'):
            conn.execute('INSERT INTO photos (path,filename,thumbnail) VALUES (?,?,?)',
                         (path, path, image.getvalue()))
        conn.commit()
    monkeypatch.setattr(compat, 'needs_compat_worker', lambda: False)
    monkeypatch.setattr(SaliencyScorer, 'load', lambda self: None)
    calls = []

    def score(self, images, cv_images):
        assert len(images) == 1
        if calls:
            with get_connection(db) as conn:
                assert conn.execute('SELECT COUNT(*) FROM photos WHERE subject_sharpness=7').fetchone()[0] == 1
            raise RuntimeError('interrupted second photo')
        calls.append(True)
        return [dict(subject_sharpness=7)]

    monkeypatch.setattr(SaliencyScorer, 'score_batch', score)
    monkeypatch.setenv('FACET_DEVICE', 'cpu')
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--recompute-saliency', '--db', db])
    with pytest.raises(RuntimeError, match='interrupted second photo'):
        facet.main()
