"""大象手选范围、权限、独立结果，以及头眼优先和不确定不评分的验证。"""

import copy
import json
import os
import time
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from db.schema import init_database
from models.elephant_detector import ElephantDetector
from models.elephant_focus import FOCUS_CHECK_VERSION, assess_result, head_eye_evidence, score_elephants
from models.elephant_landmarks import LANDMARK_MODEL_ID, ElephantLandmarkLoadError
from processing.elephant_detection import connect, get_job, run_job, source_signature, result_version


def measured_result(eye_score=8, head_score=9):
    return {'status': 'scored', 'width': 1000, 'height': 1000, 'count': 1, 'best_index': 0,
            'elephants': [{'bbox': [0, 0, .9, .9], 'focus_verified': True,
                           'focus_check_version': FOCUS_CHECK_VERSION,
                           'focus_evidence': {'reason': None, 'landmarks': {'model': LANDMARK_MODEL_ID},
                                              'eyes': [{'eye_score': eye_score, 'head_score': head_score}]}}]}


def test_elephant_model_selects_only_elephant_class():
    assert ElephantDetector.species == 'elephant'


def test_blurred_eye_cannot_be_compensated_by_sharp_head_or_body():
    result = assess_result(measured_result(2, 9.3))
    assert result['focus_score'] == 2
    assert result['assessment'] == 'fail'
    assert result['assessment_reason'] == 'head_eye_blurred'


def test_body_clipping_does_not_disqualify_verified_elephant_head_and_eyes():
    result = assess_result(measured_result())
    assert result['focus_score'] == 8 and result['assessment'] == 'pass'


@pytest.mark.parametrize('flag', ['stale', 'outdated'])
def test_changed_original_or_rule_never_displays_old_score(flag):
    result = measured_result()
    result[flag] = True
    result = assess_result(result)
    assert result['score'] is None and result['elephants'][0]['focus_score'] is None
    assert result['assessment_reason'] == flag


def test_missing_or_invalid_measurement_is_not_filled_with_body_score():
    result = measured_result()
    result['score'] = result['elephants'][0]['sharpness'] = 9.3
    result['elephants'][0]['focus_evidence']['eyes'][0]['eye_score'] = None
    assert assess_result(result)['score'] is None
    result['elephants'][0]['focus_evidence']['eyes'][0]['eye_score'] = float('nan')
    assert assess_result(result)['score'] is None


def pose(points=None, confidence=None):
    points = points if points is not None else [[.5, .5]] * 17
    confidence = confidence if confidence is not None else [1] * 17
    return {'points': copy.deepcopy(points), 'views': [copy.deepcopy(points), copy.deepcopy(points)],
            'visibility': confidence, 'view_visibility': [confidence, confidence], 'model': LANDMARK_MODEL_ID}


def known_animal():
    return {'bbox': [0, 0, 1, 1], 'mask_outline': [[[0, 0], [1, 0], [1, 1], [0, 1]]]}


def test_unstable_landmarks_do_not_trigger_sharpness_measurement():
    landmarks = pose()
    landmarks['points'][3] = [.8, .8]
    landmarks['views'][1][0] = [0, 0]
    landmarks['visibility'][1] = 0
    detector = Mock()
    detector.detect.return_value = landmarks
    with patch('models.elephant_focus._region_score') as score:
        result = head_eye_evidence(Image.new('RGB', (1000, 1000)), known_animal(), detector)
    assert result['reason'] == 'head_unlocalized'
    score.assert_not_called()


def test_independent_local_eye_verification_and_lower_region_score():
    points = [[.5, .5] for _ in range(17)]
    points[3] = [.8, .8]
    confidence = [1] * 17
    confidence[1] = 0  # 仅一只眼可见，另一只眼不参与。
    coarse, local = pose(points, confidence), pose(confidence=confidence)
    detector = Mock()
    detector.detect.side_effect = [coarse, local]
    with patch('models.elephant_focus._region_score', side_effect=[2.5, 9.3]) as score:
        result = head_eye_evidence(Image.new('RGB', (1000, 1000)), known_animal(), detector)
    assert result['score'] == 2.5 and result['reason'] is None
    assert len(result['eyes']) == 1 and score.call_count == 2
    assert result['eyes'][0]['eye_radius'] < result['eyes'][0]['head_radius']


def test_unconfirmed_eye_is_not_replaced_by_body_or_background_detail():
    points = [[.5, .5] for _ in range(17)]
    points[3] = [.8, .8]
    confidence = [1] * 17
    confidence[1] = 0
    local = pose(confidence=confidence)
    local['views'][1][0] = [0, 0]
    detector = Mock()
    detector.detect.side_effect = [pose(points, confidence), local]
    with patch('models.elephant_focus._region_score') as score:
        result = head_eye_evidence(Image.new('RGB', (1000, 1000)), known_animal(), detector)
    assert result['reason'] == 'eye_unverified' and 'score' not in result
    score.assert_not_called()


def test_size_filter_and_no_elephant_never_load_head_eye_model():
    detector = Mock()
    image = Image.new('RGB', (1000, 1000), 'gray')
    assert score_elephants(image, [], detector=detector)['status'] == 'no_elephant'
    detections = [{'box': [100, 100, 200, 200], 'confidence': .9, '_mask': np.ones((64, 64), dtype=bool)}]
    result = score_elephants(image, detections, min_area_percent=3, detector=detector)
    assert result['score'] is None and result['excluded_small_count'] == 1
    detector.detect.assert_not_called()


@pytest.fixture
def library(tmp_path, monkeypatch):
    from api.routers import elephants, tasks
    db = str(tmp_path / 'elephants.db')
    init_database(db)
    paths = []
    for index in range(3):
        path = str(tmp_path / f'photo-{index}.jpg')
        Image.new('RGB', (200, 200), 'gray').save(path)
        paths.append(path)
    with connect(db) as conn:
        conn.executemany('INSERT INTO photos (path,filename,category,aggregate,star_rating,bird_sharpness) '
                         'VALUES (?,?,?,?,?,?)', [(p, p.rsplit('/', 1)[-1], 'wildlife', 7.5, 4, 6.5) for p in paths])

    @contextmanager
    def get_db():
        with connect(db) as conn:
            yield conn

    monkeypatch.setattr(elephants, 'DEFAULT_DB_PATH', db)
    monkeypatch.setattr(elephants, 'get_db', get_db)
    monkeypatch.setattr(elephants, 'get_visibility_clause', lambda user_id: ('1=1', []))
    monkeypatch.setattr(elephants, '_processes', {})
    monkeypatch.setattr(tasks, 'DEFAULT_DB_PATH', db)
    monkeypatch.setattr(tasks, 'get_db', get_db)
    monkeypatch.setattr('facet.library_job_holder', lambda db: None)
    monkeypatch.setattr('api.routers.scan._recompute_conflict_detail', lambda: None)
    return db, paths


def seed_job(db, paths, job_id='test-job', owner='test', cancel=False):
    with connect(db) as conn:
        conn.execute('INSERT INTO elephant_detection_jobs '
                     '(id,user_id,status,phase,total,paths_json,options_json,created_at,updated_at,pid,cancel_requested) '
                     'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                     (job_id, owner, 'queued', 'preparing', len(paths), json.dumps(paths),
                      json.dumps({'tiled': False, 'recompute': False, 'min_area_percent': 1}),
                      time.time(), time.time(), os.getpid(), int(cancel)))


class FakeDetector:
    def __init__(self, **kwargs):
        pass

    def load(self):
        pass

    def detect(self, image):
        return []


def test_worker_processes_only_selection_preserves_category_rating_and_bird_result(library):
    db, paths = library
    seed_job(db, paths[:1])
    run_job(db, 'test-job', FakeDetector, Image.open, str)
    assert get_job(db, 'test-job')['succeeded'] == 1
    with connect(db) as conn:
        rows = conn.execute('SELECT category,aggregate,star_rating,bird_sharpness,elephant_status FROM photos ORDER BY path').fetchall()
    assert [row['elephant_status'] for row in rows] == ['no_elephant', None, None]
    assert all(tuple(row)[:4] == ('wildlife', 7.5, 4, 6.5) for row in rows)
    seed_job(db, paths[:1], 'cached')
    run_job(db, 'cached', Mock(side_effect=AssertionError('cached model')), Image.open, str)
    assert get_job(db, 'cached')['skipped'] == 1
    with open(paths[0], 'ab') as file:
        file.write(b'changed')
    seed_job(db, paths[:1], 'changed')
    run_job(db, 'changed', FakeDetector, Image.open, str)
    assert get_job(db, 'changed')['succeeded'] == 1


def test_cancelled_job_leaves_unprocessed_photos_untouched(library):
    db, paths = library
    seed_job(db, paths, cancel=True)
    run_job(db, 'test-job', Mock(side_effect=AssertionError('cancelled model')), Image.open, str)
    job = get_job(db, 'test-job')
    assert job['status'] == 'cancelled' and job['completed'] == 0


def test_model_load_failure_stops_without_overwriting_photo(library):
    db, paths = library
    seed_job(db, paths)
    with patch.object(FakeDetector, 'load', side_effect=RuntimeError('offline')):
        run_job(db, 'test-job', FakeDetector, Image.open, str)
    assert get_job(db, 'test-job')['status'] == 'failed'
    with connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM photos WHERE elephant_status IS NOT NULL').fetchone()[0] == 0


def test_start_uses_exact_selection_and_deduplicates(library, edition_client):
    db, paths = library
    process = Mock(pid=os.getpid())
    process.poll.return_value = None
    with patch('api.routers.elephants.subprocess.Popen', return_value=process) as spawn, patch('api.routers.elephants.threading.Thread'):
        response = edition_client.post('/api/elephants/jobs', json={'paths': [paths[0], paths[1], paths[0]]})
    assert response.status_code == 200, response.text
    job = get_job(db, response.json()['id'])
    assert json.loads(job['paths_json']) == paths[:2]
    assert 'processing.elephant_detection' in spawn.call_args.args[0]


@pytest.mark.parametrize('threshold', [-1, 101, 'nan', 'inf'])
def test_invalid_threshold_never_starts_worker(library, edition_client, threshold):
    _, paths = library
    with patch('api.routers.elephants.subprocess.Popen') as spawn:
        response = edition_client.post('/api/elephants/jobs', json={'paths': paths[:1], 'min_area_percent': threshold})
    assert response.status_code == 422
    spawn.assert_not_called()


def test_job_owner_is_required_for_status_cancellation_and_gallery_scope(library, edition_client):
    from api.routers.gallery import _build_gallery_where
    db, paths = library
    seed_job(db, paths, owner='another')
    assert edition_client.get('/api/elephants/jobs/test-job').status_code == 404
    assert edition_client.post('/api/elephants/jobs/test-job/cancel', json={'confirm': True}).status_code == 404
    assert get_job(db, 'test-job')['cancel_requested'] == 0
    with connect(db) as conn:
        clauses, params = _build_gallery_where({'elephant_job': 'test-job'}, conn, user_id='test')
        assert conn.execute('SELECT path FROM photos WHERE '+' AND '.join(clauses), params).fetchall() == []


def test_background_task_uses_elephant_kind_and_can_cancel_owned_job(library, edition_client):
    db, paths = library
    seed_job(db, paths)
    response = edition_client.get('/api/tasks')
    assert response.status_code == 200, response.text
    task = next(task for task in response.json()['tasks'] if task['id'] == 'test-job')
    assert task['kind'] == 'elephants' and task['can_terminate']
    response = edition_client.post('/api/tasks/test-job/terminate', json={'confirm': True})
    assert response.status_code == 200, response.text
    assert get_job(db, 'test-job')['cancel_requested'] == 1


def test_photo_and_preview_reject_changed_original(library, edition_client, monkeypatch):
    from processing.elephant_detection import save_result
    db, paths = library
    monkeypatch.setattr('api.path_validation.resolve_photo_disk_path', lambda path: path)
    result = measured_result()
    save_result(db, paths[0], result, source_signature(paths[0]), result_version(False))
    response = edition_client.get('/api/elephants/photo', params={'path': paths[0]})
    assert response.status_code == 200 and response.json()['focus_score'] == 8
    assert edition_client.get('/api/elephants/preview', params={'path': paths[0]}).status_code == 200
    with open(paths[0], 'ab') as file:
        file.write(b'changed')
    response = edition_client.get('/api/elephants/photo', params={'path': paths[0]})
    assert response.json()['stale'] and response.json()['focus_score'] is None
    assert edition_client.get('/api/elephants/preview', params={'path': paths[0]}).status_code == 409


def test_real_eye_blur_lowers_score_even_with_sharp_surrounding_skin():
    from PIL import ImageFilter
    yy, xx = np.indices((1000, 1000))
    pixels = (((xx // 16) % 2) * 200 + 20).astype('uint8')
    image = Image.fromarray(pixels).convert('RGB')
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(4)))
    region = (xx - 500) ** 2 + (yy - 500) ** 2 <= 60 ** 2
    mixed = np.asarray(image).copy()
    mixed[region] = blurred[region]
    points = [[.5, .5] for _ in range(17)]
    points[3] = [.8, .8]
    confidence = [1] * 17
    confidence[1] = 0

    def measure(original):
        detector = Mock()
        detector.detect.side_effect = [pose(points, confidence), pose(confidence=confidence)]
        return head_eye_evidence(original, known_animal(), detector)

    sharp = measure(image)
    soft_eye = measure(Image.fromarray(mixed))
    assert sharp['score'] > 8
    assert soft_eye['score'] < 3
    assert soft_eye['eyes'][0]['head_score'] > 8


def test_head_eye_model_failure_preserves_existing_result_and_stops(library):
    from processing.elephant_detection import save_result
    db, paths = library
    seed_job(db, paths)
    save_result(db, paths[0], measured_result(), source_signature(paths[0]), 'old-version')
    with patch('processing.elephant_detection.score_elephants', side_effect=ElephantLandmarkLoadError('offline')):
        run_job(db, 'test-job', FakeDetector, Image.open, str)
    assert get_job(db, 'test-job')['status'] == 'failed'
    with connect(db) as conn:
        row = conn.execute('SELECT elephant_status,elephant_version FROM photos WHERE path=?', (paths[0],)).fetchone()
    assert tuple(row) == ('scored', 'old-version')
