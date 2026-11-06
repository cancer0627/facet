"""头眼与翅膀分离的合焦对照，以及关键点稳定性、缓存和尺寸门槛。"""

import copy
from unittest import mock

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from models.bird_focus import FOCUS_CHECK_VERSION, needs_focus_verification, verify_focus
from models.bird_landmarks import BirdLandmarkDetector, BirdLandmarkLoadError
from models.bird_eye import eye_candidates, verify_eye


class KnownLandmarks:
    def __init__(self):
        self.calls = 0
        self.points = [[.35, .45], [.36, .49], [.22, .45]]
        self.visibility = [.99, .01, .99]
        self.views = [copy.deepcopy(self.points), copy.deepcopy(self.points)]

    def detect(self, crop):
        self.calls += 1
        return {'points': copy.deepcopy(self.points), 'visibility': self.visibility[:],
                'views': copy.deepcopy(self.views), 'model': 'synthetic-keypoints'}

    def inspect_eye(self, crop):
        # 数学样例用深色圆表示眼睛、较亮三角形表示喙；假局部模型区分这两种标签。
        pixels = np.asarray(crop)
        y, x = crop.height // 2, crop.width // 2
        central = np.median(pixels[y - 2:y + 3, x - 2:x + 3], axis=(0, 1))
        is_eye = np.median(central) < 45
        eye_visibility = [.01, .99] if central[2] - central[0] > 20 else [.99, .01]
        point = [.5, .5] if is_eye else [.05, .05]
        points = [point, point, [.2, .5]]
        return {'points': points, 'views': [copy.deepcopy(points), copy.deepcopy(points)],
                'visibility': [*eye_visibility, .99], 'model': 'synthetic-local-keypoints'}


def focus_scene(head_blur=0, wing_blur=0):
    image = Image.new('RGB', (800, 600), (90, 90, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 720, 540), fill=(220, 220, 220))
    for x in range(480, 680, 15):
        draw.rectangle((x, 120, x + 5, 450), fill=(25, 25, 25))
    # 已知关键点对应原图中的眼睛与喙；眼部细节不与翅膀区域相交。
    draw.ellipse((291, 263, 317, 289), fill=(15, 15, 15))
    draw.ellipse((299, 267, 305, 273), fill=(250, 250, 250))
    draw.polygon([(221, 276), (255, 260), (255, 285)], fill=(60, 60, 60))
    if head_blur:
        with image.filter(ImageFilter.GaussianBlur(head_blur)) as blurred:
            image.paste(blurred.crop((180, 160, 430, 400)), (180, 160))
    if wing_blur:
        with image.filter(ImageFilter.GaussianBlur(wing_blur)) as blurred:
            image.paste(blurred.crop((470, 90, 700, 480)), (470, 90))
    bird = {'bbox': [.1, .1, .9, .9], 'width': 640, 'height': 480, 'sharpness': 9.3,
            'mask_outline': [[[.1, .1], [.9, .1], [.9, .9], [.1, .9]]], 'frame_margin': [3, 3]}
    return image, {'width': 800, 'height': 600, 'score_basis': 'bird_interior', 'birds': [bird]}


def test_blurred_head_and_eye_cannot_be_outvoted_by_sharp_wings():
    sharp, details = focus_scene()
    blurred, _ = focus_scene(head_blur=4)
    good = verify_focus(sharp, details, 1, detector=KnownLandmarks())
    bad = verify_focus(blurred, details, 1, detector=KnownLandmarks())
    assert good['score'] > 8
    assert bad['score'] < 4
    assert bad['birds'][0]['body_sharpness'] == 9.3
    assert bad['birds'][0]['focus_verified'] is True  # 可靠测出模糊，与无法判断不同。
    assert bad['score_basis'] == 'bird_head_eye'
    assert details['birds'][0]['sharpness'] == 9.3


def test_motion_in_wings_does_not_lower_a_sharp_eye_score():
    still, details = focus_scene()
    moving, _ = focus_scene(wing_blur=8)
    assert verify_focus(still, details, 1, detector=KnownLandmarks())['score'] == verify_focus(
        moving, details, 1, detector=KnownLandmarks(),
    )['score']


def test_a_sharp_second_eye_cannot_compensate_for_a_blurred_first_eye():
    image, details = focus_scene(head_blur=4)
    detector = KnownLandmarks()
    detector.points[1] = [.4, .65]
    detector.views = [copy.deepcopy(detector.points), copy.deepcopy(detector.points)]
    detector.visibility = [.99, .99, .99]
    # 第二只眼及头部保持清楚，第一只眼仍为模糊的原区域。
    draw = ImageDraw.Draw(image)
    # 蓝色通道标识第二只眼，供假模型区分左右；不按清晰度决定眼睛身份。
    draw.ellipse((323, 359, 349, 385), fill=(15, 15, 70))
    result = verify_focus(image, details, 1, detector=detector)
    regions = result['birds'][0]['focus_evidence']['eyes']
    assert len(regions) == 2 and regions[1]['eye_score'] > 8
    assert result['score'] < 4


def test_missing_body_detail_does_not_prevent_measuring_a_clear_eye():
    image, details = focus_scene()
    details['birds'][0].update(sharpness=None, reason='insufficient_detail')
    result = verify_focus(image, details, 1, detector=KnownLandmarks())
    assert result['score'] > 8 and result['birds'][0]['reason'] is None


def test_background_and_wings_cannot_raise_a_blurred_eye_score():
    image, details = focus_scene(head_blur=4)
    expected = verify_focus(image, details, 1, detector=KnownLandmarks())['score']
    draw = ImageDraw.Draw(image)
    for x in range(450, 800, 4):
        draw.line((x, 0, x, 600), fill='black', width=2)
    assert verify_focus(image, details, 1, detector=KnownLandmarks())['score'] == expected


@pytest.mark.parametrize('visibility', [[.1, .1, .99], [.99, .01, .1]])
def test_missing_reliable_eye_or_beak_is_reviewed_without_body_fallback(visibility):
    image, details = focus_scene()
    detector = KnownLandmarks()
    detector.visibility = visibility
    result = verify_focus(image, details, 1, detector=detector)
    assert result['status'] == 'unjudgeable' and result['score'] is None
    bird = result['birds'][0]
    assert bird['sharpness'] is None and bird['body_sharpness'] == 9.3
    assert bird['reason'] == 'head_unlocalized'
    assert bird['exclusion_reason'] == 'focus_unverified'
    assert result['focus_unverified_count'] == 1


def test_inconsistent_mirrored_keypoints_are_not_used_to_guess_a_head():
    image, details = focus_scene()
    detector = KnownLandmarks()
    detector.views[1][0][0] += .2
    result = verify_focus(image, details, 1, detector=detector)
    assert result['score'] is None and result['birds'][0]['reason'] == 'head_unlocalized'


class RefinedLandmarks(KnownLandmarks):
    def __init__(self):
        super().__init__()
        self.refine_calls = 0
        self.raw_views = [copy.deepcopy(self.points), copy.deepcopy(self.points)]
        self.raw_views[1][0] = [.6, .3]
        self.raw_views[1][2] = [.5, .3]
        self.raw_visibility = [[.99, .01, .99], [.36, .01, .81]]
        self.isolated_views = copy.deepcopy(self.raw_views)
        self.isolated_visibility = [[.98, .01, .98], [.1, .01, .6]]

    @staticmethod
    def observations(views, visibility):
        return {'points': np.mean(views, axis=0).tolist(), 'views': copy.deepcopy(views),
                'visibility': np.min(visibility, axis=0).tolist(),
                'view_visibility': copy.deepcopy(visibility), 'model': 'synthetic-keypoints'}

    def detect(self, crop):
        self.calls += 1
        return self.observations(self.raw_views, self.raw_visibility)

    def refine(self, crop, outline):
        self.refine_calls += 1
        return self.observations(self.isolated_views, self.isolated_visibility)


def test_background_refinement_recovers_a_head_when_the_mirrored_view_is_distracted():
    image, details = focus_scene()
    before = np.asarray(image).copy()
    detector = RefinedLandmarks()
    result = verify_focus(image, details, 1, detector=detector)
    evidence = result['birds'][0]['focus_evidence']
    assert result['score'] > 8 and detector.refine_calls == 1
    assert evidence['landmarks']['support_views'] == ['original', 'isolated']
    assert len(evidence['localization_attempts']) == 2
    np.testing.assert_array_equal(np.asarray(image), before)


def test_refinement_does_not_accept_an_eye_supported_only_by_an_altered_background():
    image, details = focus_scene()
    detector = RefinedLandmarks()
    detector.raw_visibility = [[.36, .01, .99], [.36, .01, .99]]
    detector.isolated_views = [copy.deepcopy(detector.points), copy.deepcopy(detector.points)]
    detector.isolated_visibility = [[.99, .01, .99], [.99, .01, .99]]
    assert verify_focus(image, details, 1, detector=detector)['score'] is None


def test_refinement_does_not_relax_position_consistency_to_force_a_detection():
    image, details = focus_scene()
    detector = RefinedLandmarks()
    detector.isolated_views[0][0] = [.65, .45]
    detector.isolated_views[0][2] = [.52, .45]
    result = verify_focus(image, details, 1, detector=detector)
    assert result['score'] is None and result['birds'][0]['reason'] == 'head_unlocalized'


def test_two_conflicting_supported_heads_in_an_overlap_remain_unjudgeable():
    image, details = focus_scene()
    detector = RefinedLandmarks()
    detector.raw_views[1] = copy.deepcopy(detector.raw_views[0])
    detector.raw_views[1][0] = [.65, .45]
    detector.raw_views[1][2] = [.52, .45]
    detector.raw_visibility = [[.99, .01, .99], [.99, .01, .99]]
    detector.isolated_views = copy.deepcopy(detector.raw_views)
    detector.isolated_visibility = copy.deepcopy(detector.raw_visibility)
    assert verify_focus(image, details, 1, detector=detector)['score'] is None


def test_already_consistent_head_does_not_run_background_refinement():
    image, details = focus_scene()
    detector = RefinedLandmarks()
    detector.raw_views = [copy.deepcopy(detector.points), copy.deepcopy(detector.points)]
    detector.raw_visibility = [[.99, .01, .99], [.99, .01, .99]]
    assert verify_focus(image, details, 1, detector=detector)['score'] > 8
    assert detector.refine_calls == 0


def test_a_predicted_eye_outside_the_bird_is_not_scored():
    image, details = focus_scene()
    details['birds'][0]['mask_outline'] = [[[.6, .1], [.9, .1], [.9, .9], [.6, .9]]]
    assert verify_focus(image, details, 1, detector=KnownLandmarks())['score'] is None


def test_flat_eye_region_remains_unknown_instead_of_zero_or_a_body_high_score():
    image, details = focus_scene()
    ImageDraw.Draw(image).rectangle((180, 160, 430, 400), fill=(220, 220, 220))
    result = verify_focus(image, details, 1, detector=KnownLandmarks())
    assert result['score'] is None and result['birds'][0]['reason'] == 'eye_unverified'


def test_consistent_high_confidence_wing_root_is_corrected_to_an_actual_eye():
    image, details = focus_scene()
    detector = KnownLandmarks()
    for point in detector.points:
        point[1] -= 136 / 480
    detector.views = [copy.deepcopy(detector.points), copy.deepcopy(detector.points)]
    result = verify_focus(image, details, 1, detector=detector)
    assert result['score'] > 8
    measured = result['birds'][0]['focus_evidence']['eyes'][0]
    np.testing.assert_allclose(np.asarray(measured['centre']) * image.size, [304, 276], atol=3)
    verification = measured['verification']
    assert any(item['support'] == 'local_model' and item['offset'] > item['diameter']
               for item in verification['candidates'])


def test_a_dark_ear_spot_is_rejected_when_local_keypoints_point_elsewhere():
    image, details = focus_scene()
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 160, 430, 400), fill=(220, 220, 220))
    draw.ellipse((291, 263, 317, 289), fill=(60, 60, 60))
    result = verify_focus(image, details, 1, detector=KnownLandmarks())
    assert result['score'] is None and result['birds'][0]['reason'] == 'eye_unverified'
    candidates = result['birds'][0]['focus_evidence']['eye_verification']['candidates']
    assert candidates and all(not item['accepted'] for item in candidates)


def test_eye_candidates_allow_a_highlight_inside_a_dark_iris():
    image, details = focus_scene()
    ImageDraw.Draw(image).ellipse((300, 272, 308, 280), fill='white')
    candidates = eye_candidates(image, details['birds'][0], np.asarray([304, 276]), 83.2)
    assert any(np.linalg.norm(np.asarray(item['centre']) - [304, 276]) < 3 for item in candidates)


def test_two_supported_eye_structures_for_the_same_keypoint_are_not_ranked_by_sharpness():
    image, details = focus_scene()
    ImageDraw.Draw(image).ellipse((370, 305, 396, 331), fill=(15, 15, 15))
    centre, evidence = verify_eye(image, details['birds'][0], np.asarray([304, 276]), 83.2,
                                 KnownLandmarks(), 0)
    assert centre is None and evidence['ambiguous'] is True


def test_background_noise_does_not_validate_an_eye_at_a_consistent_model_location():
    image, details = focus_scene()
    pixels = np.clip(128 + np.random.default_rng(42).normal(0, 10, (600, 800, 3)), 0, 255).astype('uint8')
    result = verify_focus(Image.fromarray(pixels), details, 1, detector=KnownLandmarks())
    assert result['score'] is None and result['birds'][0]['reason'] == 'eye_unverified'


def test_missing_segmentation_does_not_run_landmark_model_or_score_rectangle_background():
    image, details = focus_scene()
    details['birds'][0]['mask_outline'] = []
    detector = KnownLandmarks()
    assert verify_focus(image, details, 1, detector=detector)['score'] is None
    assert detector.calls == 0


def test_head_eye_scores_are_cached_and_lower_size_threshold_checks_new_candidates():
    image, details = focus_scene()
    details['birds'].append(copy.deepcopy(details['birds'][0]))
    details['birds'][1].update(bbox=[.1, .1, .5, .7], width=320, height=360)
    detector = KnownLandmarks()
    checked = verify_focus(image, details, 30, detector=detector)
    assert detector.calls == 1
    assert checked['birds'][0]['focus_check_version'] == FOCUS_CHECK_VERSION
    assert checked['birds'][1]['sharpness'] is None
    assert checked['birds'][1]['body_sharpness'] == 9.3
    assert not needs_focus_verification(checked, 30)
    assert needs_focus_verification(checked, 0)
    verify_focus(image, checked, 0, detector=detector)
    assert detector.calls == 2
    assert 'focus_check_version' not in details['birds'][0]


def test_old_outline_approval_cannot_bypass_head_eye_measurement():
    image, details = focus_scene(head_blur=4)
    details['birds'][0].update(focus_check_version='outline-step-v1', focus_verified=True)
    assert needs_focus_verification(details, 1)
    result = verify_focus(image, details, 1, detector=KnownLandmarks())
    assert result['score'] < 4 and result['birds'][0]['body_sharpness'] == 9.3


def test_frame_fragment_is_not_verified_as_a_complete_bird():
    image, details = focus_scene()
    details['birds'][0]['frame_edge'] = True
    detector = KnownLandmarks()
    assert not needs_focus_verification(details, 0)
    result = verify_focus(image, details, 0, detector=detector)
    assert result['score'] is None and detector.calls == 0
    assert result['birds'][0]['body_sharpness'] == 9.3


def test_landmark_preprocessing_and_mirrored_eye_mapping_without_network():
    detector = BirdLandmarkDetector()
    coords = np.asarray([[[.3, .4], [.7, .4], [.2, .5]]], dtype=np.float32)
    mirrored = coords[:, [1, 0, 2]].copy()
    mirrored[:, :, 0] = 1 - mirrored[:, :, 0]
    detector.session = mock.Mock()
    detector.session.run.side_effect = [(coords, np.asarray([[.9, .1, .8]])),
                                        (mirrored, np.asarray([[.2, .95, .85]]))]
    image = Image.new('RGB', (80, 60), (230, 100, 30))
    ImageDraw.Draw(image).rectangle((0, 0, 20, 60), fill='black')
    before = np.asarray(image).copy()
    result = detector.detect(image)
    np.testing.assert_allclose(result['points'], coords[0], atol=1e-6)
    np.testing.assert_allclose(result['visibility'], [.9, .1, .8])
    first, second = [call.args[1]['input'] for call in detector.session.run.call_args_list]
    assert first.shape == (1, 3, 416, 416) and first.dtype == np.float32
    np.testing.assert_array_equal(first[:, :, :, ::-1], second)
    np.testing.assert_array_equal(np.asarray(image), before)


def test_background_refinement_changes_only_localizer_input_outside_segmentation():
    detector = BirdLandmarkDetector()
    coords = np.asarray([[[.3, .4], [.7, .4], [.2, .5]]], dtype=np.float32)
    detector.session = mock.Mock()
    detector.session.run.return_value = (coords, np.asarray([[.9, .1, .8]]))
    image = Image.new('RGB', (80, 60), (230, 100, 30))
    before = np.asarray(image).copy()
    detector.refine(image, [[(.5, 0), (1, 0), (1, 1), (.5, 1)]])
    first, second = [call.args[1]['input'] for call in detector.session.run.call_args_list]
    mean = np.asarray([.485, .456, .406], dtype=np.float32)
    std = np.asarray([.229, .224, .225], dtype=np.float32)
    np.testing.assert_allclose(first[0, :, 100, 100], (127 / 255 - mean) / std, atol=1e-6)
    np.testing.assert_allclose(first[0, :, 100, 300], (np.asarray([230, 100, 30]) / 255 - mean) / std,
                               atol=1e-6)
    np.testing.assert_array_equal(first[:, :, :, ::-1], second)
    np.testing.assert_array_equal(np.asarray(image), before)


def test_landmark_load_failure_is_distinct_from_an_unjudgeable_photo(tmp_path, monkeypatch):
    detector = BirdLandmarkDetector(tmp_path)
    def unavailable(*args, **kwargs):
        raise OSError('download unavailable')
    monkeypatch.setattr('models.bird_landmarks.download_weights', unavailable)
    with pytest.raises(BirdLandmarkLoadError):
        detector.load()


def test_landmark_cancel_does_not_download_or_run_a_model(tmp_path, monkeypatch):
    detector = BirdLandmarkDetector(tmp_path, cancelled=lambda: True)
    monkeypatch.setattr('models.bird_landmarks.download_weights', lambda *a, **kw: pytest.fail('cancelled'))
    with pytest.raises(InterruptedError):
        detector.detect(Image.new('RGB', (80, 60)))
