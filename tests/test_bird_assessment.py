"""选片判断必须依靠当前头眼证据，不能因整鸟高分或另一只清楚就误判。"""

import copy

import pytest

from models.bird_assessment import assess_result
from models.bird_focus import FOCUS_CHECK_VERSION
from models.bird_landmarks import LANDMARK_MODEL_ID


def clear_bird():
    return {'bbox': [.2, .2, .8, .8], 'mask_outline': [[[.2, .2], [.8, .2], [.8, .8]]],
            'frame_edge': False, 'sharpness': 8.62, 'focus_verified': True,
            'focus_check_version': FOCUS_CHECK_VERSION,
            'focus_evidence': {'landmarks': {'model': LANDMARK_MODEL_ID}, 'reason': None, 'eyes': [
                {'centre': [.4, .4], 'head_score': 8.62, 'eye_score': 8.62,
                 'verification': {'selected_centre': [400, 400], 'selected_diameter': 30}}]}}


def result(*birds, **overrides):
    return {'width': 1000, 'height': 1000, 'status': 'scored', 'min_area_percent': 1,
            'score': 8.62, 'birds': list(birds), **overrides}


@pytest.mark.parametrize('key', ['head_score', 'eye_score'])
def test_clear_wings_or_another_clear_region_cannot_compensate_for_verified_blur(key):
    bird = clear_bird()
    bird['sharpness'] = 9.3
    bird['focus_evidence']['eyes'][0][key] = 2.0
    r = assess_result(result(bird))
    assert r['assessment'] == 'fail'
    assert r['assessment_reason'] == 'head_eye_blurred'


def test_one_complete_clear_bird_qualifies_even_if_another_bird_is_clipped():
    clipped = clear_bird()
    clipped['bbox'][0] = 0
    r = assess_result(result(clipped, clear_bird()))
    assert r['assessment'] == 'pass' and r['assessment_index'] == 1
    assert r['birds'][0]['assessment'] == 'fail'


def test_unknown_candidate_prevents_whole_photo_failure_when_another_bird_is_clipped():
    clipped, unknown = clear_bird(), clear_bird()
    clipped['bbox'][0] = 0
    unknown.pop('focus_evidence')
    r = assess_result(result(clipped, unknown))
    assert r['assessment'] == 'review' and r['assessment_index'] == 1


def test_missing_eye_proof_cannot_be_replaced_with_a_high_stored_sharpness():
    bird = clear_bird()
    bird['focus_evidence']['eyes'][0].pop('verification')
    assert assess_result(result(bird))['assessment'] == 'review'


def test_selected_eye_coordinates_must_match_the_recorded_verified_original_location():
    bird = clear_bird()
    bird['focus_evidence']['eyes'][0]['centre'] = [.7, .5]
    assert assess_result(result(bird))['assessment'] == 'review'


@pytest.mark.parametrize('updates', [{'outdated': True}, {'stale': True}, {'status': 'error'}])
def test_stale_old_or_failed_results_never_pass_or_fail_from_old_measurements(updates):
    assert assess_result(result(clear_bird(), **updates))['assessment'] == 'review'


def test_unvalidated_old_focus_version_stays_for_review():
    bird = clear_bird()
    bird['focus_check_version'] = 'head-eye-v2'
    assert assess_result(result(bird))['assessment'] == 'review'
    assert assess_result(result(clear_bird()), current=False)['assessment'] == 'review'


def test_near_edge_is_uncertain_instead_of_asserting_the_body_is_cut_off():
    bird = clear_bird()
    bird['frame_edge'] = True
    r = assess_result(result(bird))
    assert r['assessment'] == 'review' and r['assessment_reason'] == 'body_uncertain'


def test_tiny_sharp_bird_cannot_rescue_a_larger_blurred_bird():
    large, small = clear_bird(), clear_bird()
    large['focus_evidence']['eyes'][0]['eye_score'] = 2
    small['bbox'] = [.1, .1, .11, .11]
    r = assess_result(result(large, small))
    assert r['assessment'] == 'fail'
    assert r['birds'][1]['assessment_reason'] == 'below_min_area'


def test_confirmed_blurred_head_can_fail_when_eye_detail_is_insufficient():
    bird = clear_bird()
    bird['focus_verified'] = False
    bird['focus_evidence']['reason'] = 'eye_detail_insufficient'
    bird['focus_evidence']['eyes'][0].update(head_score=2, eye_score=None)
    assert assess_result(result(bird))['assessment'] == 'fail'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -1, 11, True, 6.4])
def test_missing_invalid_or_borderline_eye_measurements_are_not_a_pass(value):
    bird = clear_bird()
    bird['focus_evidence']['eyes'][0]['eye_score'] = value
    assert assess_result(result(bird))['assessment'] == 'review'


def test_derived_assessment_does_not_modify_cached_evidence_or_reference_scores():
    original = result(clear_bird())
    before = copy.deepcopy(original)
    assessed = assess_result(original)
    assert assessed['assessment'] == 'pass'
    assert original == before and assessed['score'] == 8.62


def test_no_bird_is_review_without_inventing_a_successful_subject():
    assert assess_result(result(status='no_bird'))['assessment'] == 'review'


@pytest.mark.parametrize('malformed', [None, [], {'eyes': None},
                                     {'landmarks': {'model': LANDMARK_MODEL_ID}, 'eyes': [None]},
                                     {'landmarks': {'model': LANDMARK_MODEL_ID}, 'eyes': [
                                         {'centre': None, 'verification': {'selected_centre': None}}]}])
def test_incomplete_cached_evidence_stays_review_without_breaking_the_gallery(malformed):
    bird = clear_bird()
    bird['focus_evidence'] = malformed
    assert assess_result(result(bird))['assessment'] == 'review'


def test_ten_point_score_uses_all_verified_regions_and_keeps_midrange_measurements():
    bird = clear_bird()
    bird['focus_evidence']['eyes'][0]['eye_score'] = 6.4
    r = assess_result(result(bird))
    assert r['focus_score'] == 6.4
    assert r['birds'][0]['focus_score'] == 6.4
    assert r['assessment'] == 'review'  # 内部三态不把中间参考分强行改成固定 5 分。


def test_unknown_clipped_and_stale_birds_have_no_score_instead_of_an_invented_score():
    bird = clear_bird()
    bird['bbox'][0] = 0
    assert assess_result(result(bird))['focus_score'] is None
    assert assess_result(result(clear_bird(), stale=True))['focus_score'] is None
    bird = clear_bird()
    bird.pop('focus_evidence')
    assert assess_result(result(bird))['focus_score'] is None


def test_ten_point_score_does_not_use_detection_confidence_or_a_higher_body_score():
    bird = clear_bird()
    bird.update(sharpness=9.3, confidence=.999)
    bird['focus_evidence']['eyes'][0]['eye_score'] = 2
    assert assess_result(result(bird))['focus_score'] == 2.0
