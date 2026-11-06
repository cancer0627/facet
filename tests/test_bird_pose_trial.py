"""候选姿态模型的坐标、左右眼、重复目标和无目标行为。"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from models.bird_pose_trial import BirdPoseTrialDetector
from scripts.compare_bird_landmarks import ensure_settings, selected_bird


def prediction(box=(320, 320, 640, 320), confidence=.9, mirrored=False):
    points = np.asarray([[128, 320, .95], [256, 288, .9], [384, 288, .8],
                         [320, 230, .9], [280, 350, .8], [360, 350, .8]], dtype=np.float32)
    if mirrored:
        points = points[[0, 2, 1, 3, 5, 4]].copy()
        points[:, 0] = 640 - points[:, 0]
    return np.asarray([*box, confidence, *points.reshape(-1)], dtype=np.float32)


def detector_with_outputs(outputs):
    detector = BirdPoseTrialDetector()
    detector.session = mock.Mock()
    detector.session.get_inputs.return_value = [SimpleNamespace(name='images')]
    detector.session.run.side_effect = [[output] for output in outputs]
    return detector


def test_letterbox_and_mirrored_keypoints_map_back_to_the_same_original_eye():
    outputs = [prediction()[None, :, None], prediction(mirrored=True)[None, :, None]]
    detector = detector_with_outputs(outputs)
    image = Image.new('RGB', (200, 100), (200, 100, 50))
    before = np.asarray(image).copy()
    result = detector.detect(image)
    np.testing.assert_allclose(result['points'], [[.4, .4], [.6, .4], [.2, .5]], atol=1e-6)
    np.testing.assert_allclose(result['views'][0], result['views'][1], atol=1e-6)
    np.testing.assert_allclose(result['visibility'], [.9, .8, .95])
    inputs = detector.session.run.call_args_list[0].args[1]['images']
    assert inputs.shape == (1, 3, 640, 640) and inputs.dtype == np.float32
    np.testing.assert_allclose(inputs[0, :, 0, 0], 114 / 255, atol=1e-6)
    np.testing.assert_allclose(inputs[0, :, 300, 100], np.asarray([200, 100, 50]) / 255, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(image), before)


def test_empty_detection_has_zero_visibility_instead_of_an_invented_eye():
    output = np.zeros((1, 23, 4), dtype=np.float32)
    detector = detector_with_outputs([output, output])
    result = detector.detect(Image.new('RGB', (200, 100)))
    assert result['visibility'] == [0., 0., 0.]
    assert all(not item['detected'] for item in result['detections'])


def test_small_neighboring_bird_does_not_replace_the_large_target_with_higher_confidence():
    small = prediction(box=(500, 310, 100, 100), confidence=.99)
    large = prediction(confidence=.8)
    output = np.stack([small, large], axis=1)[None]
    detector = detector_with_outputs([output, output])
    result = detector.detect(Image.new('RGB', (200, 100)))
    assert result['detections'][0]['box_confidence'] == pytest.approx(.8)
    assert result['detections'][0]['candidate_count'] == 2


def test_duplicate_predictions_are_suppressed_before_target_selection():
    output = np.stack([prediction(confidence=.9), prediction(confidence=.7)], axis=1)[None]
    detector = detector_with_outputs([output, output])
    result = detector.detect(Image.new('RGB', (200, 100)))
    assert result['detections'][0]['candidate_count'] == 1


def test_refinement_masks_only_localizer_input_and_keeps_original_pixels():
    output = prediction()[None, :, None]
    detector = detector_with_outputs([output, output])
    image = Image.new('RGB', (200, 100), (200, 100, 50))
    detector.refine(image, [[(.5, 0), (1, 0), (1, 1), (.5, 1)]])
    first = detector.session.run.call_args_list[0].args[1]['images']
    np.testing.assert_allclose(first[0, :, 300, 100], 127 / 255, atol=1e-6)
    np.testing.assert_allclose(first[0, :, 300, 500], np.asarray([200, 100, 50]) / 255, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(image)[0, 0], [200, 100, 50])


def test_malformed_model_output_is_reported_instead_of_decoding_wrong_keypoints():
    detector = detector_with_outputs([np.zeros((1, 56, 4), dtype=np.float32)])
    with pytest.raises(ValueError, match='output shape'):
        detector.detect(Image.new('RGB', (200, 100)))


@pytest.mark.parametrize('size', [0, 319, 641, 1312])
def test_unbounded_or_unaligned_input_sizes_are_rejected(size):
    with pytest.raises(ValueError):
        BirdPoseTrialDetector(input_size=size)


def test_cancelled_trial_does_not_load_weights_or_run_inference():
    detector = BirdPoseTrialDetector(cancelled=lambda: True)
    with pytest.raises(InterruptedError):
        detector.detect(Image.new('RGB', (200, 100)))


def test_comparison_keeps_the_same_target_and_excludes_frame_fragments():
    details = {'min_area_percent': 1, 'birds': [
        {'bbox': [0, 0, 1, 1], 'frame_edge': True, 'mask_outline': [[(0, 0), (1, 0), (1, 1)]]},
        {'bbox': [.2, .2, .8, .8], 'mask_outline': [[(.2, .2), (.8, .2), (.8, .8)]]},
    ]}
    assert selected_bird(details) == 1
    details['best_index'] = 1
    assert selected_bird(details) == 1


def test_comparison_resume_rejects_different_model_or_preprocessing(tmp_path):
    ensure_settings(tmp_path, {'input_sizes': [640], 'model_hash': 'first'})
    ensure_settings(tmp_path, {'input_sizes': [640], 'model_hash': 'first'})
    with pytest.raises(ValueError, match='settings changed'):
        ensure_settings(tmp_path, {'input_sizes': [1280], 'model_hash': 'first'})
    with pytest.raises(ValueError, match='settings changed'):
        ensure_settings(tmp_path, {'input_sizes': [640], 'model_hash': 'second'})


def test_unidentified_old_comparison_cannot_silently_mix_with_new_results(tmp_path):
    (tmp_path / 'results.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='no provenance'):
        ensure_settings(tmp_path, {'input_sizes': [640]})
