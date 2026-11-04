"""原图轮廓证据的对照：真实失焦、清楚轮廓、平坦区域与尺寸门槛。"""

import copy

import cv2
from PIL import Image, ImageDraw, ImageFilter

from models.bird_focus import FOCUS_CHECK_VERSION, needs_focus_verification, outline_evidence, verify_focus
from models.bird_detector import select_eligible_birds


def ellipse_scene(blur=0):
    image = Image.new('RGB', (512, 512), (50, 50, 50))
    ImageDraw.Draw(image).ellipse((100, 130, 410, 380), fill=(220, 220, 220))
    image = image.filter(ImageFilter.GaussianBlur(blur))
    polygon = cv2.ellipse2Poly((255, 255), (155, 125), 0, 0, 360, 3)
    bird = {'bbox': [100 / 512, 130 / 512, 410 / 512, 380 / 512],
            'width': 310, 'height': 250, 'sharpness': 9.3,
            'mask_outline': [[(float(x) / 512, float(y) / 512) for x, y in polygon]]}
    return image, {'width': 512, 'height': 512, 'birds': [bird]}


def test_broad_true_outline_rejects_an_unsupported_high_interior_score():
    sharp, details = ellipse_scene()
    blurred, _ = ellipse_scene(8)
    sharp_evidence = outline_evidence(sharp, details['birds'][0])
    blurred_evidence = outline_evidence(blurred, details['birds'][0])
    assert sharp_evidence['sigma'] < 1.5
    assert blurred_evidence['sigma'] >= 6
    assert verify_focus(sharp, details, 1)['birds'][0]['focus_verified'] is True
    checked = verify_focus(blurred, details, 1)
    assert checked['birds'][0]['focus_verified'] is False
    assert checked['birds'][0]['sharpness'] == 9.3  # 留作诊断，不改造成另一个任意分数。
    selected = select_eligible_birds(checked, 1)
    assert selected['score'] is None and selected['status'] == 'unjudgeable'
    assert selected['birds'][0]['exclusion_reason'] == 'focus_unverified'
    assert selected['focus_unverified_count'] == 1


def test_flat_image_does_not_get_a_sharp_edge_from_the_segmentation_outline():
    _, details = ellipse_scene()
    flat = Image.new('RGB', (512, 512), (100, 100, 100))
    evidence = outline_evidence(flat, details['birds'][0])
    assert evidence['sigma'] is None and evidence['coverage'] == 0
    assert verify_focus(flat, details, 1)['birds'][0]['focus_verified'] is False


def test_unverified_estimate_never_beats_supported_lower_score_when_size_rule_changes():
    _, details = ellipse_scene()
    details['birds'][0]['focus_verified'] = False
    supported = copy.deepcopy(details['birds'][0])
    supported.update(sharpness=5., focus_verified=True)
    details['birds'].append(supported)
    for minimum in (0, 1, 3):
        selected = select_eligible_birds(details, minimum)
        assert selected['best_index'] == 1 and selected['score'] == 5.


def test_missing_outline_or_original_frame_outside_pixels_cannot_verify_focus():
    image = Image.new('RGB', (256, 256), 'white')
    bird = {'bbox': [0, 0, 1, 1], 'width': 256, 'height': 256, 'sharpness': 9.3}
    assert outline_evidence(image, bird)['sigma'] is None
    bird['mask_outline'] = [[(0, 0), (1, 0), (1, 1), (0, 1)]]
    assert outline_evidence(image, bird)['sigma'] is None


def test_verification_is_cached_per_bird_and_lower_size_threshold_checks_new_candidates(monkeypatch):
    image, details = ellipse_scene()
    small = copy.deepcopy(details['birds'][0])
    small['bbox'] = [.1, .1, .12, .12]
    details['birds'].append(small)
    checked = verify_focus(image, details, 1)
    assert checked['birds'][0]['focus_check_version'] == FOCUS_CHECK_VERSION
    assert 'focus_check_version' not in checked['birds'][1]
    assert not needs_focus_verification(checked, 1)
    assert needs_focus_verification(checked, 0)
    calls = []

    def evidence(image, bird):
        calls.append(bird['bbox'])
        return {'sigma': .55, 'coverage': .8, 'samples': 100}

    monkeypatch.setattr('models.bird_focus.outline_evidence', evidence)
    verify_focus(image, checked, 0)
    assert calls == [small['bbox']]
    assert 'focus_check_version' not in details['birds'][0]
