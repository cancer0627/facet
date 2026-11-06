"""验证独立头框的补边坐标、相邻头保留及眼部范围约束。"""

import numpy as np
import pytest
from PIL import Image

from models.bird_head_trial import IndependentBirdHeadTrial, decode_heads, map_head_landmarks
from scripts.compare_bird_heads import constrain_eye_to_head, expanded_crop


def test_portrait_square_padding_maps_head_without_stretching_horizontal_axis():
    result = decode_heads([[2, 1, 0, -3]], [[.2, .4, .2, .2]], (100, 200))
    np.testing.assert_allclose(result['heads'][0]['bbox'], [20, 60, 60, 100])


def test_duplicate_queries_collapse_but_two_overlapping_birds_keep_separate_heads():
    result = decode_heads([[2, 1, -3, -3], [1, 2, -3, -3], [1, 2, -3, -3]],
                          [[.2, .2, .1, .1], [.205, .2, .1, .1], [.35, .25, .1, .1]], (200, 100))
    assert len(result['heads']) == 2
    assert [r['reason'] for r in result['proposals']].count('duplicate') == 1


def test_whole_bird_and_padding_detections_do_not_become_head_proposals():
    result = decode_heads([[3, 2, 4, 0], [3, 2, -3, 0]],
                          [[.5, .25, 1, .5], [.5, .9, .1, .1]], (200, 100))
    assert result['heads'] == []
    assert {r['reason'] for r in result['proposals']} == {'whole_subject_box', 'empty_box'}


def test_head_prompt_cannot_override_stronger_bird_or_wing_evidence():
    result = decode_heads([[2, 1, 3, 0], [2, 1, 0, 3]],
                          [[.2, .2, .1, .1], [.5, .2, .1, .1]], (200, 100))
    assert result['heads'] == []
    assert {r['reason'] for r in result['proposals']} == {'non_head_label_wins'}


def test_empty_or_nonfinite_observations_do_not_invent_a_head():
    assert decode_heads(np.empty((0, 4)), np.empty((0, 4)), (200, 100))['heads'] == []
    assert decode_heads([[np.nan, 2, 0, 0]], [[.2, .2, .1, .1]], (200, 100))['heads'] == []
    assert decode_heads([[-10, -10, 2, 0]], [[.2, .2, .1, .1]], (200, 100))['heads'] == []


def test_head_landmarks_preserve_both_views_when_mapped_to_the_same_bird_crop():
    raw = {'points': [[.5, .5], [.6, .5], [.1, .8]],
           'views': [[[.45, .5], [.6, .5], [.1, .8]], [[.55, .5], [.6, .5], [.1, .8]]],
           'visibility': [.9, .1, .9]}
    result = map_head_landmarks(raw, (100, 200, 300, 400), (1000, 500))
    np.testing.assert_allclose(result['points'][0], [.2, .6])
    np.testing.assert_allclose(result['views'][0][0], [.19, .6])
    np.testing.assert_allclose(result['views'][1][0], [.21, .6])
    assert raw['points'][0] == [.5, .5]


def test_eye_search_cannot_jump_from_one_head_to_a_neighbor_and_keep_a_score():
    evidence = {'eyes': [{'centre': [.8, .5]}], 'score': 8.62, 'reason': None}
    result = constrain_eye_to_head(evidence, (10, 10, 100, 100), (200, 200), (1000, 1000))
    assert 'score' not in result
    assert result['eyes'] == []
    assert result['reason'] == 'eye_outside_independent_head'
    assert result['rejected_eye_regions'][0]['centre'] == [.8, .5]


def test_eye_inside_independent_head_keeps_its_original_image_position():
    evidence = {'eyes': [{'centre': [.25, .25]}], 'score': 2.0, 'reason': None}
    result = constrain_eye_to_head(evidence, (10, 10, 100, 100), (200, 200), (1000, 1000))
    assert result['score'] == 2.0
    assert result['eyes'][0]['centre'] == [.25, .25]


def test_head_crop_adds_context_and_clips_at_original_bird_boundary():
    assert expanded_crop((0, 10, 100, 110), (120, 120)) == (0, 0, 120, 120)


def test_cancelled_trial_does_not_load_the_independent_model():
    with pytest.raises(InterruptedError):
        IndependentBirdHeadTrial(cancelled=lambda: True).detect(Image.new('RGB', (100, 100)))


def test_unknown_model_shape_is_reported_instead_of_decoded():
    with pytest.raises(ValueError, match='output shape'):
        decode_heads(np.zeros((1, 3)), np.zeros((1, 4)), (100, 100))
