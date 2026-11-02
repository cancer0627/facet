"""验证单张及整批升级范围、按模型调度和失败后继续。"""
import io
import json

import pytest
from PIL import Image

from db import get_connection, init_database


def _all_photo_work(*_args):
    return {flag: {'/a.jpg', '/b.jpg'} for flag in (
        '--recompute-iqa', '--recompute-saliency', '--recompute-composition-cpu',
        '--recompute-blinks', '--recompute-eyes-expression', '--recompute-face-signals',
        '--recompute-average',
    )}


@pytest.fixture
def library(tmp_path):
    db = str(tmp_path / 'scope.db')
    init_database(db)
    image = io.BytesIO()
    Image.new('RGB', (64, 64), 'gray').save(image, format='JPEG')
    with get_connection(db) as conn:
        for path in ('/a.jpg', '/b.jpg'):
            conn.execute('INSERT INTO photos (path, filename, thumbnail, face_count, is_blink, aggregate, exposure_score) '
                         'VALUES (?, ?, ?, 1, 1, 3, 5)', (path, path, image.getvalue()))
        conn.commit()
    return db


def test_composition_blinks_and_aggregates_stay_on_target(library):
    pytest.importorskip('cv2')
    from processing.scorer import Facet
    scorer = Facet(db_path=library, lightweight=True)
    with get_connection(library) as conn:
        before = dict(conn.execute("SELECT * FROM photos WHERE path = '/b.jpg'").fetchone())
    scorer.recompute_composition_scores(photo_path='/a.jpg')
    scorer.recompute_blink_detection(photo_path='/a.jpg')
    scorer.update_all_aggregates(photo_path='/a.jpg')
    with get_connection(library) as conn:
        after = dict(conn.execute("SELECT * FROM photos WHERE path = '/b.jpg'").fetchone())
        target = conn.execute("SELECT comp_score, is_blink FROM photos WHERE path = '/a.jpg'").fetchone()
    assert before == after
    assert target['comp_score'] is not None
    assert target['is_blink'] == 0


def test_iqa_writes_only_target(library, monkeypatch):
    pytest.importorskip('torch')
    from processing.scorer import Facet

    class Scorer:
        def __init__(self, **kwargs):
            pass

        def load(self):
            pass

        def unload(self):
            pass

        def score_batch(self, images):
            return [7.0] * len(images)

    monkeypatch.setattr('models.pyiqa_scorer.PyIQAScorer', Scorer)
    scorer = Facet(db_path=library, lightweight=True)
    scorer.recompute_iqa_from_thumbnails(photo_path='/a.jpg')
    with get_connection(library) as conn:
        target = conn.execute("SELECT aesthetic_iaa FROM photos WHERE path = '/a.jpg'").fetchone()[0]
        other = conn.execute("SELECT aesthetic_iaa FROM photos WHERE path = '/b.jpg'").fetchone()[0]
    assert target == 7.0
    assert other is None


@pytest.mark.parametrize('quiet', [False, True])
@pytest.mark.parametrize('failed_flag', [
    '--extract-gps', '--recompute-iqa', '--recompute-saliency', '--generate-critiques',
])
def test_failed_stage_continues_later_stages(library, monkeypatch, failed_flag, quiet, capsys):
    import subprocess
    import sys
    import facet
    monkeypatch.setattr(facet, '_upgrade_photo_work', _all_photo_work)

    calls = []

    def run(cmd, **kwargs):
        assert ('stdout' in kwargs) == quiet
        if '--upgrade-photos-file' in cmd:
            assert 'FACET_UPGRADE_PROGRESS_FILE' in kwargs['env']
        calls.append(cmd)
        failed = failed_flag in cmd
        return subprocess.CompletedProcess(cmd, 1 if failed else 0)

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--upgrade-db', '--db', library,
                                    '--generate-critiques', 'zh'] + (['--quiet'] if quiet else []))
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code == 1
    stages = [cmd for cmd in calls if '--upgrade-photos-file' in cmd]
    assert len(stages) == 8
    assert '--recompute-iqa' in stages[0]
    assert '--recompute-average' in stages[-2]
    assert '--generate-critiques' in stages[-1]

    output = capsys.readouterr().err
    assert '8/8' in output
    if quiet:
        assert 'Detailed log: ' in output


@pytest.mark.parametrize('setting', [None, False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_upgrade_critique_switch(library, tmp_path, monkeypatch, setting, explicit):
    import subprocess
    import sys
    import facet
    monkeypatch.setattr(facet, '_upgrade_photo_work', _all_photo_work)

    config_path = tmp_path / 'scoring.json'
    config = {} if setting is None else {'critique': {'generate_on_upgrade': setting}}
    config_path.write_text(json.dumps(config))
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, 'run', run)
    argv = ['facet.py', '--upgrade-db', '--db', library, '--config', str(config_path)]
    if explicit:
        argv += ['--generate-critiques', 'zh']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code == 0
    photo_calls = [cmd for cmd in calls if '--upgrade-photos-file' in cmd]
    # Legacy configuration cannot enable generation without the explicit flag.
    expected_steps = 8 if explicit else 7
    assert len(photo_calls) == expected_steps
    assert '--recompute-average' in photo_calls[6]
    critiques = [cmd for cmd in photo_calls if '--generate-critiques' in cmd]
    assert len(critiques) == int(explicit)
    if explicit:
        assert critiques[0][critiques[0].index('--generate-critiques') + 1] == 'zh'


def test_completed_scores_skip_workers_and_config_change_only_refreshes_aggregate(library, monkeypatch):
    import subprocess
    import sys
    import facet
    from config import ScoringConfig

    config = ScoringConfig(None, validate=False)
    with get_connection(library) as conn:
        conn.execute('UPDATE photos SET face_count = 0, aesthetic_iaa = 7, liqe_score = 6, '
                     'qrealign_score = 5, subject_sharpness = 5, comp_score = 6, '
                     'power_point_score = 6, leading_lines_score = 5, config_version = ?',
                     (config.version_hash,))
        conn.commit()
    # No face / no subject detected: NULL face scores and subject_bbox are valid.
    assert all(not paths for paths in facet._upgrade_photo_work(library, config).values())
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--upgrade-db', '--db', library])
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code == 0
    assert not any('--upgrade-photos-file' in cmd for cmd in calls)
    with get_connection(library) as conn:
        conn.execute("UPDATE photos SET config_version = 'old' WHERE path = '/b.jpg'")
        conn.commit()
    plan = facet._upgrade_photo_work(library, config)
    assert plan['--recompute-average'] == {'/b.jpg'}
    assert all(not paths for flag, paths in plan.items() if flag != '--recompute-average')


def test_missing_score_refreshes_aggregate_even_when_config_matches(library, monkeypatch):
    import subprocess
    import sys
    import facet

    plan = {flag: set() for flag in _all_photo_work()}
    plan['--recompute-iqa'] = {'/b.jpg'}
    monkeypatch.setattr(facet, '_upgrade_photo_work', lambda *_args: plan)
    calls = []
    scopes = {}

    def run(cmd, **kwargs):
        calls.append(cmd)
        if '--upgrade-photos-file' in cmd:
            with open(cmd[-1]) as scope_file:
                scopes[cmd[-1]] = json.load(scope_file)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--upgrade-db', '--db', library])
    with pytest.raises(SystemExit):
        facet.main()
    photo_calls = [cmd for cmd in calls if '--upgrade-photos-file' in cmd]
    assert [cmd[-3] for cmd in photo_calls] == ['--recompute-iqa', '--recompute-average']
    assert all(scopes[cmd[-1]] == ['/b.jpg'] for cmd in photo_calls)


def test_unavailable_saliency_dependency_is_reported_once(library, monkeypatch):
    import subprocess
    import sys
    import facet

    monkeypatch.setattr(facet, '_upgrade_photo_work', _all_photo_work)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 3 if '--recompute-saliency' in cmd else 0)

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--upgrade-db', '--db', library])
    with pytest.raises(SystemExit) as result:
        facet.main()
    assert result.value.code == 1
    assert len([cmd for cmd in calls if '--recompute-saliency' in cmd]) == 1
    assert any('--recompute-average' in cmd for cmd in calls)


def test_iqa_model_loaded_once_for_whole_scope(library, monkeypatch):
    """即使显存充足，升级仍逐模型处理整批，保留范围外和已有评分。"""
    from models.model_manager import ModelManager
    from processing.scorer import Facet

    events = []

    class Scorer:
        def __init__(self, model_name):
            self.name = model_name

        def load(self):
            events.append(('load', self.name))

        def unload(self):
            events.append(('unload', self.name))

        def score_batch(self, images):
            events.append(('score', self.name, len(images)))
            return [7.0] * len(images)

    monkeypatch.setattr('models.pyiqa_scorer.PyIQAScorer', Scorer)
    monkeypatch.setattr(ModelManager, 'detect_vram', staticmethod(lambda: 80.0))
    with get_connection(library) as conn:
        conn.execute("INSERT INTO photos (path, filename, thumbnail, face_count) "
                     "SELECT '/c.jpg', 'c.jpg', thumbnail, 1 FROM photos LIMIT 1")
        conn.execute("UPDATE photos SET aesthetic_iaa = 9 WHERE path = '/a.jpg'")
        conn.commit()
    scorer = Facet(db_path=library, lightweight=True)
    scorer.recompute_iqa_from_thumbnails(
        photo_path=['/a.jpg', '/b.jpg'], sequential_models=True,
    )
    for name, column in scorer._IQA_MODELS:
        size = 1 if column == 'aesthetic_iaa' else 2
        assert events[:3] == [('load', name), ('score', name, size), ('unload', name)]
        events = events[3:]
    assert not events
    with get_connection(library) as conn:
        rows = {row['path']: dict(row) for row in conn.execute('SELECT * FROM photos')}
    assert rows['/a.jpg']['aesthetic_iaa'] == 9
    assert rows['/b.jpg']['aesthetic_iaa'] == 7
    assert rows['/c.jpg']['aesthetic_iaa'] is None
    assert rows['/c.jpg']['liqe_score'] is None


@pytest.mark.parametrize('batch_size', [1, 64])
def test_iqa_progress_waits_for_batch_inference(library, tmp_path, monkeypatch, batch_size):
    from processing.scorer import Facet
    from processing.upgrade_progress import PROGRESS_ENV

    state = tmp_path / 'progress.json'
    monkeypatch.setenv(PROGRESS_ENV, str(state))
    counts_during_inference = []

    class Scorer:
        def __init__(self, **kwargs):
            pass

        def load(self):
            pass

        def unload(self):
            pass

        def score_batch(self, images):
            event = json.loads(state.read_text())
            counts_during_inference.append(event['current'])
            assert event['current'] < event['total']
            return [7.0] * len(images)

    monkeypatch.setattr('models.pyiqa_scorer.PyIQAScorer', Scorer)
    scorer = Facet(db_path=library, lightweight=True)
    scorer._run_iqa_pass(
        ['topiq_iaa'], batch_size,
        {'updated': {'aesthetic_iaa': 0}, 'errors': 0},
    )
    assert counts_during_inference
    assert json.loads(state.read_text())['current'] == 2


@pytest.mark.parametrize('scope', [['/a.jpg'], []])
def test_batch_scope_keeps_other_photos_unchanged(library, scope):
    from processing.scorer import Facet
    scorer = Facet(db_path=library, lightweight=True)
    with get_connection(library) as conn:
        before = dict(conn.execute("SELECT * FROM photos WHERE path = '/b.jpg'").fetchone())
    scorer.recompute_composition_scores(photo_path=scope)
    scorer.recompute_blink_detection(photo_path=scope)
    scorer.recompute_eyes_expression(photo_path=scope)
    scorer.recompute_face_signals(photo_path=scope)
    scorer.update_all_aggregates(photo_path=scope)
    with get_connection(library) as conn:
        after = dict(conn.execute("SELECT * FROM photos WHERE path = '/b.jpg'").fetchone())
    assert before == after


def test_interrupted_batch_keeps_aggregates_pending(library, tmp_path, monkeypatch):
    import sys
    import facet
    from config import ScoringConfig
    from processing.scorer import Facet

    config = ScoringConfig(None, validate=False)
    with get_connection(library) as conn:
        conn.execute('UPDATE photos SET config_version = ?', (config.version_hash,))
        conn.commit()
    scope_file = tmp_path / 'scope.json'
    scope_file.write_text(json.dumps(['/a.jpg']))

    def interrupted_score(self, **kwargs):
        assert kwargs == {'photo_path': ['/a.jpg'], 'sequential_models': True}
        # 模拟模型已提交结果、调度器尚未刷新综合分时中断。
        with get_connection(library) as conn:
            conn.execute("UPDATE photos SET aesthetic_iaa = 7, liqe_score = 6, "
                         "face_quality_iqa = 8, qrealign_score = 5 WHERE path = '/a.jpg'")
            conn.commit()
        raise KeyboardInterrupt

    monkeypatch.setattr(Facet, 'recompute_iqa_from_thumbnails', interrupted_score)
    monkeypatch.setattr(sys, 'argv', ['facet.py', '--recompute-iqa', '--db', library,
                                    '--upgrade-photos-file', str(scope_file)])
    with pytest.raises(KeyboardInterrupt):
        facet.main()
    work = facet._upgrade_photo_work(library, config)
    assert work['--recompute-average'] == {'/a.jpg'}
    assert '/a.jpg' not in work['--recompute-iqa']
    assert '/b.jpg' in work['--recompute-iqa']
