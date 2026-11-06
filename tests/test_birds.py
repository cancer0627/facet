"""鸟类定位、局部评分、选中范围、任务恢复与预览的功能验证。"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from io import BytesIO
from unittest import mock

import numpy as np
import pytest
from PIL import Image, ImageFilter

from db.schema import init_database
from models.bird_detector import (
    BirdDetector, MODEL_ID, detection_windows, guard_frame_edges, local_sharpness, merge_detections,
    score_detections, select_eligible_birds,
)
from models.bird_instances import BirdInstanceSegmenter, instance_windows, merge_instances
from models.bird_focus import FOCUS_CHECK_VERSION
from processing.bird_detection import connect, get_job, result_version, run_job, source_signature


def checker_image(size=160):
    yy, xx = np.indices((size, size))
    return Image.fromarray((((xx // 5 + yy // 5) % 2) * 255).astype('uint8')).convert('RGB')


def masked_detection(box, confidence):
    # 数学样例的目标完整落在画面内，不把测试图边缘误当成鸟体边缘。
    x0, y0, x1, y1 = box
    return {'box': [x0 + 4, y0 + 4, x1 - 4, y1 - 4], 'confidence': confidence,
            '_mask': np.ones((64, 64), dtype=bool)}


def mixed_size_details():
    return {'width': 4000, 'height': 4000, 'score_basis': 'bird_interior', 'birds': [
        {'bbox': [.1, .1, .12, .125], 'sharpness': 9.3, 'width': 80, 'height': 100, 'reason': None},
        {'bbox': [.2, .2, .4, .35], 'sharpness': 7.68, 'width': 800, 'height': 600, 'reason': None},
    ]}


def test_tiny_sharp_bird_does_not_outvote_a_larger_bird_and_keeps_its_raw_score():
    details = mixed_size_details()
    result = select_eligible_birds(details, 1)
    assert result['best_index'] == 1 and result['score'] == 7.68
    assert result['birds'][0]['sharpness'] == 9.3
    assert result['birds'][0]['area_percent'] == .05
    assert result['birds'][0]['exclusion_reason'] == 'below_min_area'
    assert result['eligible_count'] == 1 and result['excluded_small_count'] == 1
    assert 'eligible' not in details['birds'][0]


def test_size_threshold_can_be_removed_or_changed_without_destroying_bird_scores():
    result = select_eligible_birds(mixed_size_details(), 1)
    unlimited = select_eligible_birds(result, 0)
    assert unlimited['best_index'] == 0 and unlimited['score'] == 9.3
    assert unlimited['excluded_small_count'] == 0
    too_large = select_eligible_birds(unlimited, 4)
    assert too_large['status'] == 'unjudgeable' and too_large['score'] is None
    assert too_large['count'] == 2 and too_large['excluded_small_count'] == 2
    assert select_eligible_birds(too_large, 1)['score'] == 7.68


@pytest.mark.parametrize('threshold', [-1, 101, float('nan'), float('inf')])
def test_invalid_size_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        select_eligible_birds(mixed_size_details(), threshold)


def test_sharp_crop_ranks_above_blurred_crop():
    image = checker_image()
    assert local_sharpness(image) > local_sharpness(image.filter(ImageFilter.GaussianBlur(3)))
    assert local_sharpness(Image.new('RGB', (20, 100))) is None


def test_edge_width_score_separates_focus_and_is_not_a_contrast_amount_score():
    scores = []
    for sigma in (0, 1, 2, 4):
        array = np.zeros((160, 160), dtype='uint8')
        array[:, 80:] = 200
        image = Image.fromarray(array).convert('RGB').filter(ImageFilter.GaussianBlur(sigma))
        scores.append(local_sharpness(image))
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 8 and scores[-1] < 3
    low_contrast = np.zeros((160, 160), dtype='uint8')
    low_contrast[:, 80:] = 40
    image = Image.fromarray(low_contrast).convert('RGB').filter(ImageFilter.GaussianBlur(1))
    assert abs(local_sharpness(image) - scores[1]) < 1.0


def test_random_noise_is_not_a_sharp_bird():
    rng = np.random.default_rng(42)
    noise = np.clip(128 + rng.normal(0, 10, (256, 256)), 0, 255).astype('uint8')
    assert local_sharpness(Image.fromarray(noise).convert('RGB')) < 3


def test_sharp_background_cannot_raise_the_same_masked_blurred_subject_score():
    array = np.zeros((512, 512), dtype='uint8')
    array[:, 256:] = 200
    blurred = Image.fromarray(array).convert('RGB').filter(ImageFilter.GaussianBlur(4))
    with_background = np.asarray(blurred).copy()
    yy, xx = np.indices((512, 512))
    outside = (xx < 100) | (xx > 412)
    texture = np.repeat(((((xx // 12 + yy // 12) % 2) * 200).astype('uint8'))[:, :, None], 3, axis=2)
    with_background[outside] = texture[outside]
    changed = Image.fromarray(with_background)
    mask = np.zeros((512, 512), dtype=bool)
    mask[32:480, 120:392] = True
    assert local_sharpness(changed) > local_sharpness(blurred) + 5
    assert local_sharpness(changed, mask) == local_sharpness(blurred, mask)
    assert local_sharpness(changed, mask) < 3


def test_mask_boundary_itself_is_not_evidence_of_clear_feathers():
    flat = Image.new('RGB', (160, 160), (100, 100, 100))
    mask = np.zeros((160, 160), dtype=bool)
    mask[40:120, 40:120] = True
    assert local_sharpness(flat, mask) is None


def test_coarse_segmentation_is_not_upsampled_into_a_reliable_score():
    detection = masked_detection([0, 0, 160, 160], .99)
    detection['_mask'] = np.ones((5, 8), dtype=bool)
    result = score_detections(checker_image(), [detection])
    assert result['score'] is None
    assert result['birds'][0]['reason'] == 'low_mask_resolution'


def test_unverified_detection_has_no_score_even_when_its_background_is_sharp():
    result = score_detections(checker_image(), [{'box': [0, 0, 160, 160], 'confidence': .99}])
    assert result['status'] == 'unjudgeable'
    assert result['score'] is None
    assert result['birds'][0]['reason'] == 'no_mask'


def test_original_frame_wing_fragment_cannot_win_even_with_clear_internal_detail():
    image = checker_image(320)
    with mock.patch('models.bird_detector.local_sharpness', return_value=9.3):
        result = score_detections(image, [
            {'box': [200, 100, 320, 150], 'confidence': .99, '_mask': np.ones((64, 64), dtype=bool)},
            masked_detection([50, 100, 150, 250], .8),
        ])
    assert result['birds'][0]['sharpness'] == 9.3
    assert result['birds'][0]['exclusion_reason'] == 'frame_edge'
    assert result['best_index'] == 1


@pytest.mark.parametrize('bbox', [
    [10 / 7952, .2, .3, .4], [.2, 10 / 5304, .4, .4],
    [.7, .2, 1 - 10 / 7952, .4], [.1678822938, .7986425339, .4610160966, .9981146305],
])
def test_large_original_edge_tolerance_uses_detection_scale_on_every_side(bbox):
    details = {'width': 7952, 'height': 5304, 'birds': [
        {'bbox': bbox, 'sharpness': 9.3, 'width': 2331, 'height': 1058, 'focus_verified': True},
        {'bbox': [.3, .5, .43, .74], 'sharpness': 2., 'width': 1034, 'height': 1273,
         'focus_verified': True},
    ]}
    result = guard_frame_edges(details, 1)
    assert result['score'] == 2. and result['best_index'] == 1
    assert result['birds'][0]['sharpness'] == 9.3
    assert result['birds'][0]['exclusion_reason'] == 'frame_edge'
    assert not result['birds'][0]['eligible']
    assert 'frame_edge' not in details['birds'][0]
    # 降低大小门槛不能让残缺主体重新决定照片分数。
    assert select_eligible_birds(result, 0)['score'] == 2.
    assert guard_frame_edges(result, 0)['score'] == 2.


def test_explicit_local_detection_scale_keeps_an_inset_complete_candidate():
    details = {'width': 7952, 'height': 5304, 'birds': [
        {'bbox': [.2, .2, .4, 1 - 20 / 5304], 'sharpness': 9.3, 'width': 1590, 'height': 4223,
         'frame_margin': [6., 6.]},
    ]}
    result = guard_frame_edges(details)
    assert result['score'] == 9.3 and result['birds'][0]['frame_edge'] is False
    details['birds'][0]['frame_margin'] = [6., 24.]
    result = guard_frame_edges(details)
    assert result['score'] is None and result['status'] == 'unjudgeable'
    assert result['birds'][0]['sharpness'] == 9.3


def test_segmentation_carries_actual_resize_scale_to_scoring():
    import torch

    detector = BirdInstanceSegmenter(None)
    detector.bird_label = 16
    detector.model = mock.Mock(return_value=[{
        'labels': torch.tensor([16]), 'scores': torch.tensor([.9]),
        'boxes': torch.tensor([[100., 100., 300., 255.]]),
        'masks': torch.ones((1, 1, 256, 768)),
    }])
    image = Image.new('RGB', (6000, 2000))
    detections = detector._detect_window(image, (0, 0, 6000, 2000))
    assert detections[0]['frame_margin'] == [3 * 6000 / 768, 3 * 2000 / 256]
    with mock.patch('models.bird_detector.local_sharpness', return_value=9.3):
        result = score_detections(image, detections)
    assert result['birds'][0]['frame_margin'] == detections[0]['frame_margin']
    assert result['score'] is None
    assert result['birds'][0]['exclusion_reason'] == 'frame_edge'


def test_mask_deduplication_keeps_crossing_birds_and_rejects_a_wing_fragment():
    a = masked_detection([0, 0, 200, 200], .7)
    a['truncated'] = False
    b = masked_detection([0, 0, 200, 200], .9)
    b['truncated'] = False
    a['_mask'][:, 32:] = False
    b['_mask'][:, :32] = False
    assert len(merge_instances([a, b])) == 2
    fragment = masked_detection([0, 0, 60, 100], .99)
    fragment['truncated'] = False
    kept = merge_instances([fragment, a])
    assert len(kept) == 1 and kept[0] is a


def test_instance_context_is_bounded_and_can_be_disabled():
    proposals = [masked_detection([x, 10, x + 100, 200], .9) for x in range(0, 40000, 200)]
    windows = list(instance_windows((40000, 3000), proposals))
    assert len(windows) <= 33
    assert all(0 <= left < right <= 40000 and 0 <= top < bottom <= 3000 for left, top, right, bottom in windows)
    assert list(instance_windows((40000, 3000), proposals, False)) == [(0, 0, 40000, 3000)]


def test_complete_detection_wins_over_more_confident_cutoff_and_wing_fragment():
    complete = {'box': [10, 10, 150, 150], 'confidence': .7, 'truncated': False}
    cutoff = {'box': [40, 10, 150, 150], 'confidence': .99, 'truncated': True}
    wing = {'box': [10, 10, 40, 100], 'confidence': .98, 'truncated': True}
    assert merge_detections([cutoff, wing, complete]) == [complete]


def test_unrecovered_tile_fragments_cannot_win_best_bird_score():
    image = checker_image()
    result = score_detections(image, [
        {'box': [0, 0, 160, 160], 'confidence': .99, 'truncated': True},
    ])
    assert result['score'] is None
    assert result['status'] == 'unjudgeable'
    assert result['birds'][0]['reason'] == 'truncated'


def test_tile_edge_target_gets_recentered_context_before_scoring():
    detector = BirdDetector()
    detector.model = object()
    detector.segmenter = mock.Mock()
    detector.segmenter.detect.side_effect = lambda image, proposals, enhanced: proposals
    image = Image.new('RGB', (4000, 3000))
    cutoff = {'box': [1500, 1000, 2300, 1400], 'confidence': .95, 'truncated': True}
    complete = {'box': [1400, 950, 2350, 1450], 'confidence': .7, 'truncated': False}
    original = (0, 0, 4000, 3000)
    context = (1260, 688, 2540, 1712)

    def detect_window(image, window):
        return [cutoff] if window == original else [complete]

    with mock.patch('models.bird_detector.detection_windows', return_value=iter([original])), \
            mock.patch.object(detector, '_detect_window', side_effect=detect_window) as detect:
        assert detector.detect(image) == [complete]
    assert detect.call_args_list == [mock.call(image, original), mock.call(image, context)]


def test_equal_sharpness_prefers_more_resolved_bird_over_a_small_fragment():
    image = checker_image(320)
    with mock.patch('models.bird_detector.local_sharpness', return_value=7.68):
        result = score_detections(image, [
            masked_detection([0, 0, 50, 100], .99),
            masked_detection([0, 0, 300, 300], .7),
        ])
    assert result['best_index'] == 1


def test_best_bird_is_chosen_independently_of_detection_confidence():
    image = Image.new('RGB', (320, 160))
    image.paste(checker_image().filter(ImageFilter.GaussianBlur(3)), (0, 0))
    image.paste(checker_image(), (160, 0))
    result = score_detections(image, [
        masked_detection([0, 0, 160, 160], .99),
        masked_detection([160, 0, 320, 160], .60),
        masked_detection([0, 0, 12, 12], .98),
    ])
    assert result['best_index'] == 1
    assert result['bbox'] == [164 / 320, 4 / 160, 316 / 320, 156 / 160]
    assert result['score'] == result['birds'][1]['sharpness']
    assert result['count'] == 3
    assert result['birds'][2]['sharpness'] is None


def test_no_bird_too_small_and_insufficient_detail_have_no_invented_score():
    image = Image.new('RGB', (100, 100))
    assert score_detections(image, [])['status'] == 'no_bird'
    small = score_detections(image, [{'box': [0, 0, 10, 10], 'confidence': .9}])
    assert small['status'] == 'unjudgeable'
    assert small['score'] is None
    flat = score_detections(image, [masked_detection([0, 0, 100, 100], .9)])
    assert flat['status'] == 'unjudgeable'
    assert flat['score'] is None
    assert flat['birds'][0]['reason'] == 'insufficient_detail'


def test_tiling_covers_every_edge_and_is_bounded():
    windows = list(detection_windows(7952, 5304))
    assert windows[0] == (0, 0, 7952, 5304)
    assert len(windows) <= 50
    assert any(w[2] == 7952 and w[3] == 5304 for w in windows[1:])
    assert list(detection_windows(7952, 5304, False)) == windows[:1]


@pytest.fixture
def library(tmp_path, monkeypatch):
    from api.routers import birds

    db_path = str(tmp_path / 'birds.db')
    init_database(db_path)
    paths = []
    for index in range(3):
        path = str(tmp_path / f'photo-{index}.jpg')
        checker_image().save(path)
        paths.append(path)
    with connect(db_path) as conn:
        conn.executemany('INSERT INTO photos (path,filename,category,aggregate,star_rating) VALUES (?,?,?,?,?)',
                         [(p, p.rsplit('/', 1)[-1], 'wildlife', 7.5, 4) for p in paths])

    @contextmanager
    def get_db():
        with connect(db_path) as conn:
            yield conn

    monkeypatch.setattr(birds, 'DEFAULT_DB_PATH', db_path)
    monkeypatch.setattr(birds, 'get_db', get_db)
    monkeypatch.setattr(birds, 'get_visibility_clause', lambda user_id: ('1=1', []))
    monkeypatch.setattr('api.routers.scan._recompute_conflict_detail', lambda: None)
    monkeypatch.setattr(birds, '_processes', {})
    # 此组假检测器验证任务编排；真实头眼与失焦对照由 test_bird_focus.py 独立覆盖。
    def fake_focus(image, result, min_area_percent, **kwargs):
        result = dict(result)
        result['birds'] = [dict(bird) for bird in result['birds']]
        for bird in result['birds']:
            box = bird['bbox']
            if bird['sharpness'] is not None and (box[2] - box[0]) * (box[3] - box[1]) * 100 >= min_area_percent:
                bird.update(focus_verified=True, focus_check_version=FOCUS_CHECK_VERSION)
        return result
    monkeypatch.setattr('processing.bird_detection.verify_focus', fake_focus)
    return db_path, paths


def seed_job(db_path, paths, *, job_id='test-job', user_id='test', recompute=False, cancel=False,
             min_area_percent=1, tiled=True):
    with connect(db_path) as conn:
        conn.execute(
            'INSERT INTO bird_detection_jobs '
            '(id,user_id,status,phase,total,paths_json,options_json,created_at,updated_at,pid,cancel_requested) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (job_id, user_id, 'queued', 'preparing', len(paths), json.dumps(paths),
             json.dumps({'tiled': tiled, 'recompute': recompute, 'min_area_percent': min_area_percent}),
             time.time(), time.time(), os.getpid(), int(cancel)),
        )
    return job_id


class FakeDetector:
    loaded = 0

    def __init__(self, tiled=True, cancelled=lambda: False):
        self.cancelled = cancelled

    def load(self):
        type(self).loaded += 1

    def detect(self, image):
        return [masked_detection([0, 0, image.width, image.height], .9)]


def test_size_change_reuses_cached_birds_without_loading_a_model(library):
    db_path, paths = library
    seed_job(db_path, paths[:1])
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    seed_job(db_path, paths[:1], job_id='new-size', min_area_percent=99)
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('size changes must not load a model')):
        run_job(db_path, 'new-size', FakeDetector, Image.open, str)
    job = get_job(db_path, 'new-size')
    assert (job['status'], job['succeeded'], job['failed'], job['skipped']) == ('completed', 1, 0, 0)
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_status,bird_version,bird_detection_json FROM photos WHERE path=?', (paths[0],)).fetchone()
    assert row['bird_status'] == 'unjudgeable'
    assert row['bird_version'] == result_version(True, 99)
    assert json.loads(row['bird_detection_json'])['birds'][0]['sharpness'] is not None


def test_third_version_details_are_reselected_without_running_detection(library):
    db_path, paths = library
    seed_job(db_path, paths[:1])
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? WHERE path=?',
                     ('scored', f'{MODEL_ID}:bird-interior-edge-width-v3:tiled=1', source_signature(paths[0]),
                      json.dumps(mixed_size_details()), paths[0]))
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('reuse third-version bird details')):
        run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    assert get_job(db_path, 'test-job')['succeeded'] == 1
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_sharpness,bird_version FROM photos WHERE path=?', (paths[0],)).fetchone()
    assert row['bird_sharpness'] == 7.68 and row['bird_version'] == result_version(True)


def test_fifth_version_edge_fragment_is_reselected_without_models_or_original_decode(library):
    db_path, paths = library
    details = mixed_size_details()
    details['width'], details['height'] = 7952, 5304
    details['birds'][0].update(bbox=[.168, .799, .461, .9981], width=2331, height=1058)
    for bird in details['birds']:
        bird.update(focus_verified=True, focus_check_version=FOCUS_CHECK_VERSION)
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? WHERE path=?',
                     ('scored', f'{MODEL_ID}:bird-interior-edge-width-v5:tiled=1:min_area=1',
                      source_signature(paths[0]), json.dumps(details), paths[0]))
    seed_job(db_path, paths[:1])
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('reuse cached detail')):
        run_job(db_path, 'test-job', FakeDetector, mock.Mock(side_effect=AssertionError('no decode needed')), str)
    assert get_job(db_path, 'test-job')['succeeded'] == 1
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_sharpness,bird_version,bird_detection_json FROM photos WHERE path=?', (paths[0],)).fetchone()
    assert row['bird_sharpness'] == 7.68 and row['bird_version'] == result_version(True)
    assert json.loads(row['bird_detection_json'])['birds'][0]['exclusion_reason'] == 'frame_edge'


@pytest.mark.parametrize('tiled,recompute', [(False, False), (True, True)])
def test_changed_detection_mode_or_forced_detection_does_not_reuse_details(library, tiled, recompute):
    db_path, paths = library
    seed_job(db_path, paths[:1])
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    seed_job(db_path, paths[:1], job_id='fresh-model', tiled=tiled, recompute=recompute)
    loaded = FakeDetector.loaded
    run_job(db_path, 'fresh-model', FakeDetector, Image.open, str)
    assert FakeDetector.loaded == loaded + 1
    assert get_job(db_path, 'fresh-model')['succeeded'] == 1


def test_older_focus_estimates_are_rechecked_without_loading_detection_models(library, monkeypatch):
    db_path, paths = library
    seed_job(db_path, paths[:1])
    details = mixed_size_details()
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? WHERE path=?',
                     ('scored', f'{MODEL_ID}:bird-interior-edge-width-v4:tiled=1:min_area=1',
                      source_signature(paths[0]), json.dumps(details), paths[0]))
    calls = []

    def reject_focus(image, result, minimum, **kwargs):
        calls.append(image.size)
        for bird in result['birds']:
            bird.update(focus_verified=False, focus_check_version=FOCUS_CHECK_VERSION)
        return result

    monkeypatch.setattr('processing.bird_detection.verify_focus', reject_focus)
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('reuse saved bird locations')):
        run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    assert calls == [(160, 160)]
    assert get_job(db_path, 'test-job')['succeeded'] == 1
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_status,bird_sharpness,bird_detection_json FROM photos WHERE path=?', (paths[0],)).fetchone()
    assert row['bird_status'] == 'unjudgeable' and row['bird_sharpness'] is None
    assert json.loads(row['bird_detection_json'])['birds'][1]['sharpness'] == 7.68


def test_sixth_version_outline_score_is_replaced_by_real_head_eye_measurement_and_cached(library, monkeypatch):
    from models.bird_focus import verify_focus
    from tests.test_bird_focus import KnownLandmarks, focus_scene

    db_path, paths = library
    image, details = focus_scene(head_blur=4)
    image.save(paths[0])
    details['birds'][0].update(focus_verified=True, focus_check_version='outline-step-v1')
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? '
                     'WHERE path=?', ('scored', f'{MODEL_ID}:bird-interior-edge-width-v6:tiled=1:min_area=1',
                                     source_signature(paths[0]), json.dumps(details), paths[0]))
    detector = KnownLandmarks()
    monkeypatch.setattr('processing.bird_detection.verify_focus', verify_focus)
    monkeypatch.setattr('processing.bird_detection.BirdLandmarkDetector', lambda **kwargs: detector)
    seed_job(db_path, paths[:1])
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('reuse bird segmentation')):
        run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_sharpness,bird_detection_json,bird_version FROM photos WHERE path=?',
                           (paths[0],)).fetchone()
    result = json.loads(row['bird_detection_json'])
    assert row['bird_sharpness'] < 4 and row['bird_version'] == result_version(True)
    assert result['birds'][0]['body_sharpness'] == 9.3 and result['score_basis'] == 'bird_head_eye'
    assert detector.calls == 1
    seed_job(db_path, paths[:1], job_id='repeat')
    run_job(db_path, 'repeat', FakeDetector, mock.Mock(side_effect=AssertionError('cached image')), str)
    assert get_job(db_path, 'repeat')['skipped'] == 1 and detector.calls == 1


def test_seventh_version_failed_head_is_retried_with_refinement_and_oriented_original(library, monkeypatch):
    from models.bird_focus import verify_focus
    from tests.test_bird_focus import RefinedLandmarks, focus_scene

    db_path, paths = library
    image, details = focus_scene(head_blur=4)
    # 模拟相机将横向像素保存为需按元数据旋转的原图；定位使用显示方向。
    raw = image.transpose(Image.Transpose.ROTATE_270)
    exif = Image.Exif()
    exif[274] = 8
    raw.save(paths[0], exif=exif)
    details['score_basis'] = 'bird_head_eye'
    details['birds'][0].update(sharpness=None, body_sharpness=9.3, reason='head_unlocalized',
                               focus_verified=False, focus_check_version='head-eye-v1')
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? '
                     'WHERE path=?', ('unjudgeable', f'{MODEL_ID}:bird-head-eye-edge-width-v7:tiled=1:min_area=1',
                                     source_signature(paths[0]), json.dumps(details), paths[0]))
    detector = RefinedLandmarks()
    monkeypatch.setattr('processing.bird_detection.verify_focus', verify_focus)
    monkeypatch.setattr('processing.bird_detection.BirdLandmarkDetector', lambda **kwargs: detector)
    seed_job(db_path, paths[:1])
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('reuse original bird locations')):
        run_job(db_path, 'test-job', FakeDetector, resolver=str)
    assert get_job(db_path, 'test-job')['status'] == 'completed'
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_status,bird_sharpness,bird_detection_json,bird_version FROM photos WHERE path=?',
                           (paths[0],)).fetchone()
    assert row['bird_status'] == 'scored' and row['bird_sharpness'] < 4
    assert row['bird_version'] == result_version(True) and detector.refine_calls == 1
    result = json.loads(row['bird_detection_json'])
    assert result['birds'][0]['body_sharpness'] == 9.3
    assert result['birds'][0]['focus_evidence']['landmarks']['support_views'] == ['original', 'isolated']


def test_eighth_version_consistent_but_wrong_eye_is_rechecked_without_bird_redetection(library, monkeypatch):
    from models.bird_focus import verify_focus
    from tests.test_bird_focus import KnownLandmarks, focus_scene

    db_path, paths = library
    image, details = focus_scene()
    image.save(paths[0])
    details['score_basis'] = 'bird_head_eye'
    details['birds'][0].update(body_sharpness=9.3, focus_verified=True, focus_check_version='head-eye-v2')
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? '
                     'WHERE path=?', ('scored', f'{MODEL_ID}:bird-head-eye-edge-width-v8:tiled=1:min_area=1',
                                     source_signature(paths[0]), json.dumps(details), paths[0]))
    detector = KnownLandmarks()
    for point in detector.points:
        point[1] -= 136 / 480
    detector.views = [list(map(list, detector.points)), list(map(list, detector.points))]
    monkeypatch.setattr('processing.bird_detection.verify_focus', verify_focus)
    monkeypatch.setattr('processing.bird_detection.BirdLandmarkDetector', lambda **kwargs: detector)
    seed_job(db_path, paths[:1])
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('reuse bird segmentation')):
        run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_version,bird_detection_json FROM photos WHERE path=?', (paths[0],)).fetchone()
    result = json.loads(row['bird_detection_json'])
    assert result['score'] > 8 and row['bird_version'] == result_version(True)
    eye = result['birds'][0]['focus_evidence']['eyes'][0]
    np.testing.assert_allclose(np.asarray(eye['centre']) * image.size, [304, 276], atol=3)
    assert eye['verification']['selected_centre'] is not None and detector.calls == 1
    seed_job(db_path, paths[:1], job_id='repeat-verified')
    run_job(db_path, 'repeat-verified', FakeDetector, mock.Mock(side_effect=AssertionError('reuse verified eye')), str)
    assert get_job(db_path, 'repeat-verified')['skipped'] == 1 and detector.calls == 1


def test_upgrade_with_all_birds_below_size_threshold_never_publishes_body_scores(library, monkeypatch):
    from models.bird_focus import verify_focus

    db_path, paths = library
    details = mixed_size_details()
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_detection_json=? '
                     'WHERE path=?', ('scored', f'{MODEL_ID}:bird-interior-edge-width-v6:tiled=1:min_area=1',
                                     source_signature(paths[0]), json.dumps(details), paths[0]))
    monkeypatch.setattr('processing.bird_detection.verify_focus', verify_focus)
    seed_job(db_path, paths[:1], min_area_percent=99)
    run_job(db_path, 'test-job', FakeDetector, mock.Mock(side_effect=AssertionError('no decode needed')), str)
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_sharpness,bird_detection_json FROM photos WHERE path=?', (paths[0],)).fetchone()
    result = json.loads(row['bird_detection_json'])
    assert row['bird_sharpness'] is None and result['excluded_small_count'] == 2
    assert all(bird['sharpness'] is None and bird['body_sharpness'] > 7 for bird in result['birds'])
    assert result['score_basis'] == 'bird_head_eye'


def test_head_model_load_failure_stops_job_without_overwriting_photo_results(library, monkeypatch):
    from models.bird_landmarks import BirdLandmarkLoadError

    db_path, paths = library
    seed_job(db_path, paths)
    def unavailable(*args, **kwargs):
        raise BirdLandmarkLoadError('unavailable')
    monkeypatch.setattr('processing.bird_detection.verify_focus', unavailable)
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    job = get_job(db_path, 'test-job')
    assert job['status'] == 'failed' and job['completed'] == 0 and job['failed'] == 0
    with connect(db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM photos WHERE bird_status IS NOT NULL').fetchone()[0] == 0


def test_worker_updates_only_selected_photos_and_preserves_existing_scores(library):
    db_path, paths = library
    seed_job(db_path, paths[:2])
    run_job(db_path, 'test-job', detector_factory=FakeDetector, loader=Image.open, resolver=str)
    job = get_job(db_path, 'test-job')
    assert (job['status'], job['completed'], job['succeeded'], job['failed']) == ('completed', 2, 2, 0)
    with connect(db_path) as conn:
        rows = list(conn.execute('SELECT * FROM photos ORDER BY path'))
    assert [r['bird_status'] for r in rows] == ['scored', 'scored', None]
    assert all(r['aggregate'] == 7.5 and r['star_rating'] == 4 and r['category'] == 'wildlife' for r in rows)
    assert rows[0]['bird_source_signature'] == source_signature(paths[0])
    assert rows[0]['bird_version'] == result_version(True)


def test_cached_results_skip_model_but_changed_original_is_reprocessed(library):
    db_path, paths = library
    seed_job(db_path, paths[:1])
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    seed_job(db_path, paths[:1], job_id='cached')
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('cache should skip model')):
        run_job(db_path, 'cached', FakeDetector, Image.open, str)
    assert get_job(db_path, 'cached')['skipped'] == 1
    Image.new('RGB', (180, 180), 'gray').save(paths[0])
    seed_job(db_path, paths[:1], job_id='changed')
    run_job(db_path, 'changed', FakeDetector, Image.open, str)
    assert get_job(db_path, 'changed')['succeeded'] == 1
    assert get_job(db_path, 'changed')['skipped'] == 0


def test_older_score_version_is_reprocessed_without_force(library):
    db_path, paths = library
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_source_signature=?,bird_sharpness=2.77 WHERE path=?',
                     ('scored', 'old-version', source_signature(paths[0]), paths[0]))
    seed_job(db_path, paths[:1])
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    assert get_job(db_path, 'test-job')['succeeded'] == 1
    assert get_job(db_path, 'test-job')['skipped'] == 0


def test_cancelled_job_keeps_completed_results_and_can_be_resumed(library):
    db_path, paths = library
    seed_job(db_path, paths)

    class StopAfterFirst(FakeDetector):
        def detect(self, image):
            with connect(db_path) as conn:
                conn.execute("UPDATE bird_detection_jobs SET cancel_requested=1 WHERE id='test-job'")
            return super().detect(image)

    run_job(db_path, 'test-job', StopAfterFirst, Image.open, str)
    assert get_job(db_path, 'test-job')['status'] == 'cancelled'
    assert get_job(db_path, 'test-job')['completed'] == 1
    seed_job(db_path, paths, job_id='resumed')
    run_job(db_path, 'resumed', FakeDetector, Image.open, str)
    job = get_job(db_path, 'resumed')
    assert job['skipped'] == 1 and job['succeeded'] == 2 and job['completed'] == 3


def test_same_cancelled_job_resumes_after_saved_progress_without_repeating_forced_detection(library):
    db_path, paths = library
    seed_job(db_path, paths, recompute=True)

    class StopAfterFirst(FakeDetector):
        def detect(self, image):
            with connect(db_path) as conn:
                conn.execute("UPDATE bird_detection_jobs SET cancel_requested=1 WHERE id='test-job'")
            return super().detect(image)

    run_job(db_path, 'test-job', StopAfterFirst, Image.open, str)
    assert get_job(db_path, 'test-job')['completed'] == 1
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_version=? WHERE path=?', ('saved-prefix', paths[0]))
    run_job(db_path, 'test-job', FakeDetector, Image.open, str, resume=True)
    job = get_job(db_path, 'test-job')
    assert (job['status'], job['completed'], job['succeeded'], job['failed'], job['skipped']) == ('completed', 3, 3, 0, 0)
    assert job['cancel_requested'] == 0
    with connect(db_path) as conn:
        assert conn.execute('SELECT bird_version FROM photos WHERE path=?', (paths[0],)).fetchone()[0] == 'saved-prefix'


def test_running_job_cannot_be_started_again_with_resume(library):
    db_path, paths = library
    seed_job(db_path, paths)
    with connect(db_path) as conn:
        conn.execute("UPDATE bird_detection_jobs SET status='running' WHERE id='test-job'")
    with mock.patch.object(FakeDetector, 'load', side_effect=AssertionError('already running')):
        run_job(db_path, 'test-job', FakeDetector, Image.open, str, resume=True)
    assert get_job(db_path, 'test-job')['status'] == 'running'


def test_failed_photo_does_not_stop_the_remaining_selection(library):
    db_path, paths = library
    os.unlink(paths[0])
    seed_job(db_path, paths)
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    job = get_job(db_path, 'test-job')
    assert (job['status'], job['failed'], job['succeeded']) == ('completed', 1, 2)
    with connect(db_path) as conn:
        assert conn.execute('SELECT bird_sharpness FROM photos WHERE path=?', (paths[0],)).fetchone()[0] is None


def test_model_failure_is_reported_once_without_marking_every_photo_bad(library):
    db_path, paths = library
    seed_job(db_path, paths)
    with mock.patch.object(FakeDetector, 'load', side_effect=RuntimeError('offline')):
        run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    job = get_job(db_path, 'test-job')
    assert job['status'] == 'failed' and job['completed'] == 0
    with connect(db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM photos WHERE bird_status IS NOT NULL').fetchone()[0] == 0


def test_start_job_passes_exact_selection_to_background_worker(library, edition_client):
    db_path, paths = library
    process = mock.Mock(pid=os.getpid())
    process.poll.return_value = None
    with (mock.patch('api.routers.birds.subprocess.Popen', return_value=process) as spawn,
          mock.patch('api.routers.birds.threading.Thread')):
        resp = edition_client.post('/api/birds/jobs', json={'paths': [paths[0], paths[1], paths[0]], 'min_area_percent': .25})
    assert resp.status_code == 200, resp.text
    job = get_job(db_path, resp.json()['id'])
    assert json.loads(job['paths_json']) == paths[:2]
    assert job['total'] == 2
    assert json.loads(job['options_json'])['min_area_percent'] == .25
    assert '-m' in spawn.call_args.args[0]
    assert 'processing.bird_detection' in spawn.call_args.args[0]


@pytest.mark.parametrize('threshold', [-1, 101, 'nan', 'inf'])
def test_api_rejects_invalid_size_before_starting_a_worker(library, edition_client, threshold):
    _, paths = library
    with mock.patch('api.routers.birds.subprocess.Popen') as spawn:
        response = edition_client.post('/api/birds/jobs', json={'paths': paths[:1], 'min_area_percent': threshold})
    assert response.status_code == 422
    spawn.assert_not_called()


def test_background_worker_is_reaped_and_unexpected_exit_is_reported(library):
    from api.routers.birds import _watch_process

    db_path, paths = library
    seed_job(db_path, paths)
    process = mock.Mock()
    _watch_process('test-job', process)
    process.wait.assert_called_once()
    assert get_job(db_path, 'test-job')['status'] == 'failed'


@pytest.mark.parametrize('paths', [[], [''], ['  '], ['a', '']])
def test_start_rejects_empty_paths(paths, edition_client):
    assert edition_client.post('/api/birds/jobs', json={'paths': paths}).status_code == 422


def test_start_requires_edition(regular_client):
    assert regular_client.post('/api/birds/jobs', json={'paths': ['/a.jpg']}).status_code == 403


def test_start_rejects_nonvisible_photos(library, edition_client, monkeypatch):
    db_path, paths = library
    monkeypatch.setattr('api.routers.birds.get_visibility_clause', lambda uid: ('path != ?', [paths[1]]))
    with mock.patch('api.routers.birds.subprocess.Popen') as spawn:
        resp = edition_client.post('/api/birds/jobs', json={'paths': paths[:2]})
    assert resp.status_code == 404
    spawn.assert_not_called()
    with connect(db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM bird_detection_jobs').fetchone()[0] == 0


def test_other_users_cannot_read_or_cancel_a_job(library, edition_client):
    db_path, paths = library
    seed_job(db_path, paths, user_id='other')
    assert edition_client.get('/api/birds/jobs/test-job').status_code == 404
    assert edition_client.post('/api/birds/jobs/test-job/cancel', json={'confirm': True}).status_code == 404


def test_status_survives_process_restart_and_stop_is_persisted(library, edition_client):
    db_path, paths = library
    seed_job(db_path, paths)
    resp = edition_client.get('/api/birds/jobs/test-job')
    assert resp.status_code == 200 and resp.json()['total'] == 3
    assert 'paths_json' not in resp.json()
    resp = edition_client.post('/api/birds/jobs/test-job/cancel', json={'confirm': True})
    assert resp.json()['cancel_requested'] == 1


def test_photo_preview_and_stale_results(library, edition_client, monkeypatch):
    db_path, paths = library
    seed_job(db_path, paths[:1])
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    monkeypatch.setattr('api.path_validation.resolve_photo_disk_path', lambda path: path)
    result = edition_client.get('/api/birds/photo', params={'path': paths[0]}).json()
    assert result['count'] == 1 and result['stale'] is False
    for params in ({'path': paths[0]}, {'path': paths[0], 'index': 0}):
        resp = edition_client.get('/api/birds/preview', params=params)
        assert resp.status_code == 200 and resp.headers['content-type'] == 'image/jpeg'
    assert edition_client.get('/api/birds/preview', params={'path': paths[0], 'index': 100}).status_code == 404
    Image.new('RGB', (120, 120)).save(paths[0])
    assert edition_client.get('/api/birds/photo', params={'path': paths[0]}).json()['stale'] is True
    assert edition_client.get('/api/birds/preview', params={'path': paths[0]}).status_code == 409


def test_three_way_assessment_matches_details_and_gallery_and_invalidates_changed_original(library, edition_client, monkeypatch):
    from api.db_helpers import build_photo_select_columns, sanitize_float_values
    from models.bird_landmarks import LANDMARK_MODEL_ID
    from processing.bird_detection import save_result

    db_path, paths = library
    Image.new('RGB', (1000, 1000), 'white').save(paths[0])
    monkeypatch.setattr('api.path_validation.resolve_photo_disk_path', lambda path: path)
    bird = {'bbox': [.2, .2, .8, .8], 'sharpness': 8.62, 'focus_verified': True,
            'focus_check_version': FOCUS_CHECK_VERSION, 'mask_outline': [[[.2, .2], [.8, .2], [.8, .8]]],
            'focus_evidence': {'landmarks': {'model': LANDMARK_MODEL_ID}, 'reason': None,
                               'eyes': [{'centre': [.4, .4], 'head_score': 8.62, 'eye_score': 8.62,
                                         'verification': {'selected_centre': [400, 400], 'selected_diameter': 30}}]}}
    result = {'status': 'scored', 'count': 1, 'birds': [bird], 'width': 1000, 'height': 1000, 'score': 8.62}
    save_result(db_path, paths[0], result, source_signature(paths[0]), result_version(True))

    def gallery_row():
        with connect(db_path) as conn:
            cols = build_photo_select_columns(conn)
            row = dict(conn.execute('SELECT ' + ','.join(cols) + ' FROM photos WHERE path=?', (paths[0],)).fetchone())
        sanitize_float_values([row])
        assert '_bird_detection' not in row and '_bird_source_signature' not in row
        return row

    assert edition_client.get('/api/birds/photo', params={'path': paths[0]}).json()['assessment'] == 'pass'
    assert gallery_row()['bird_assessment'] == 'pass'
    assert gallery_row()['bird_focus_score'] == 8.62
    assert edition_client.get('/api/birds/photo', params={'path': paths[0]}).json()['focus_score'] == 8.62
    Image.new('RGB', (1100, 1000), 'black').save(paths[0])
    detail = edition_client.get('/api/birds/photo', params={'path': paths[0]}).json()
    assert detail['assessment'] == 'review' and detail['assessment_reason'] == 'stale'
    assert gallery_row()['bird_assessment'] == 'review'
    assert gallery_row()['bird_focus_score'] is None
    with connect(db_path) as conn:
        assert conn.execute('SELECT bird_sharpness FROM photos WHERE path=?', (paths[0],)).fetchone()[0] == 8.62


def test_crop_preview_draws_actual_head_and_eye_measurement_regions(library, edition_client, monkeypatch):
    db_path, paths = library
    Image.new('RGB', (160, 160), (180, 180, 180)).save(paths[0])
    seed_job(db_path, paths[:1])
    run_job(db_path, 'test-job', FakeDetector, Image.open, str)
    with connect(db_path) as conn:
        row = conn.execute('SELECT bird_detection_json FROM photos WHERE path=?', (paths[0],)).fetchone()
        result = json.loads(row[0])
        result['birds'][0]['focus_evidence'] = {'eyes': [
            {'centre': [.5, .5], 'eye_radius': 16, 'head_radius': 32},
        ]}
        conn.execute('UPDATE photos SET bird_detection_json=? WHERE path=?', (json.dumps(result), paths[0]))
    monkeypatch.setattr('api.path_validation.resolve_photo_disk_path', lambda path: path)
    response = edition_client.get('/api/birds/preview', params={'path': paths[0], 'index': 0})
    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as crop:
        pixels = np.asarray(crop).astype(int)
    # 裁剪从原图 (4, 4) 开始，原图中心 (80, 80) 应变为裁剪中的 (76, 76)。
    head = pixels[43:47, 74:79]
    eye = pixels[59:63, 74:79]
    assert np.any(head[:, :, 2] > head[:, :, 0] + 50)
    assert np.any(eye[:, :, 0] > eye[:, :, 1] + 60)


def test_gallery_filters_scope_scores_and_owner(library):
    from api.routers.gallery import _build_gallery_where, _resolve_order_by

    db_path, paths = library
    seed_job(db_path, paths[:2])
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_sharpness=2,bird_status=?,bird_version=? WHERE path=?', ('scored', result_version(True), paths[0]))
        conn.execute('UPDATE photos SET bird_sharpness=8,bird_status=?,bird_version=? WHERE path=?', ('scored', result_version(True), paths[1]))
        conn.execute('UPDATE photos SET bird_sharpness=9.9,bird_status=?,bird_version=? WHERE path=?', ('scored', 'old-version', paths[2]))
        where, params = _build_gallery_where({'bird_job': 'test-job', 'min_bird_sharpness': '5'}, user_id='test')
        result = conn.execute('SELECT path FROM photos WHERE ' + ' AND '.join(where), params).fetchall()
        assert [r[0] for r in result] == [paths[1]]
        where, params = _build_gallery_where({'bird_job': 'test-job'}, user_id='other')
        assert not conn.execute('SELECT path FROM photos WHERE ' + ' AND '.join(where), params).fetchall()
        order = _resolve_order_by({'sort': 'bird_sharpness', 'dir': 'ASC'})
        ordered = conn.execute('SELECT path FROM photos ORDER BY ' + order).fetchall()
        assert ordered[-1][0] == paths[2]
        from api.db_helpers import build_photo_select_columns
        selected = conn.execute('SELECT ' + ','.join(build_photo_select_columns(conn)) + ' FROM photos WHERE path=?', (paths[2],)).fetchone()
        assert selected['bird_sharpness'] is None
        assert selected['bird_score_outdated'] == 1


def test_detail_marks_old_score_instead_of_presenting_it_as_current(library, edition_client):
    db_path, paths = library
    with connect(db_path) as conn:
        conn.execute('UPDATE photos SET bird_status=?,bird_version=?,bird_detection_json=? WHERE path=?',
                     ('scored', 'old-version', json.dumps({'score': 2.77}), paths[0]))
    resp = edition_client.get('/api/birds/photo', params={'path': paths[0]})
    assert resp.status_code == 200
    assert resp.json()['outdated'] is True
    assert resp.json()['score'] is None


def test_migration_preserves_existing_photo_data(library):
    db_path, _ = library
    init_database(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM photos WHERE aggregate=7.5').fetchone()[0] == 3
