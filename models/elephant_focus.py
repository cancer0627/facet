"""只用可靠定位的可见眼周与头部评分；身体和背景细节不能补偿模糊眼部。"""

import math

import numpy as np

from models.bird_detector import score_detections
from models.bird_focus import _region_score
from models.elephant_landmarks import ElephantLandmarkDetector, LANDMARK_MODEL_ID

FOCUS_CHECK_VERSION = 'elephant-head-eye-v1'
MIN_VISIBILITY = .5


def _eyes(landmarks, rect, outlines, size):
    import cv2

    points = np.asarray(landmarks.get('points'), dtype=float)
    views = np.asarray(landmarks.get('views'), dtype=float)
    confidence = np.asarray(landmarks.get('visibility'), dtype=float)
    if (points.shape != (17, 2) or views.shape != (2, 17, 2) or confidence.shape != (17,)
            or not all(np.isfinite(item).all() for item in (points, views, confidence))):
        return []
    extent = np.asarray([rect[2] - rect[0], rect[3] - rect[1]])
    points = points * extent + rect[:2]
    disagreement = np.linalg.norm((views[0] - views[1]) * extent, axis=1)
    polygons = [np.asarray(polygon, dtype=np.float32) * np.asarray(size, dtype=np.float32)
                for polygon in outlines if len(polygon) >= 3]
    found = []
    for index in (0, 1):
        # 象鼻尖位置随姿态大幅变化，用眼到颈部的距离定义头部上下文。
        distance = float(np.linalg.norm(points[index] - points[3]))
        if (min(confidence[index], confidence[3]) < MIN_VISIBILITY
                or not 100 <= distance <= max(extent) * .8
                or max(disagreement[index], disagreement[3]) > .15 * distance
                or not any(cv2.pointPolygonTest(poly, tuple(points[index]), False) >= 0 for poly in polygons)):
            continue
        found.append((index, points[index], distance))
    return found


def head_eye_evidence(image, animal, detector):
    box = animal['bbox']
    rect = (max(0, math.floor(box[0] * image.width)), max(0, math.floor(box[1] * image.height)),
            min(image.width, math.ceil(box[2] * image.width)), min(image.height, math.ceil(box[3] * image.height)))
    evidence = {'reason': 'head_unlocalized', 'eyes': [], 'unverified_eyes': []}
    if not animal.get('mask_outline') or min(rect[2] - rect[0], rect[3] - rect[1]) < 48:
        return evidence
    with image.crop(rect) as crop:
        landmarks = detector.detect(crop)
    evidence['landmarks'] = landmarks
    eyes = _eyes(landmarks, rect, animal['mask_outline'], image.size)
    if not eyes:
        return evidence
    scores = []
    for index, centre, distance in eyes:
        # 在较近的原图上下文中独立复核。只按位置核验，不按清晰度挑眼睛。
        radius = .85 * distance
        local_rect = (max(0, math.floor(centre[0] - radius)), max(0, math.floor(centre[1] - radius)),
                      min(image.width, math.ceil(centre[0] + radius)), min(image.height, math.ceil(centre[1] + radius)))
        with image.crop(local_rect) as crop:
            local = detector.detect(crop)
        points = np.asarray(local['views'], dtype=float)
        confidence = np.asarray(local['view_visibility'], dtype=float)
        if (points.shape != (2, 17, 2) or confidence.shape != (2, 17)
                or not np.isfinite(points).all() or not np.isfinite(confidence).all()):
            evidence['unverified_eyes'].append({'index': index, 'reason': 'invalid_localization'})
            continue
        observed = points[:, index] * [local_rect[2] - local_rect[0], local_rect[3] - local_rect[1]] + local_rect[:2]
        if (min(confidence[:, index]) < MIN_VISIBILITY
                or max(np.linalg.norm(observed - centre, axis=1)) > .12 * distance
                or np.linalg.norm(observed[0] - observed[1]) > .08 * distance):
            evidence['unverified_eyes'].append({'index': index, 'reason': 'eye_unverified', 'localization': local})
            continue
        eye_radius, head_radius = .12 * distance, .65 * distance
        scale = min(1., 768 / (2 * head_radius + 2))
        eye_score = _region_score(image, animal, centre, eye_radius, scale)
        head_score = _region_score(image, animal, centre, head_radius, scale)
        evidence['eyes'].append({
            'index': index, 'centre': (centre / image.size).tolist(),
            'eye_radius': eye_radius, 'head_radius': head_radius, 'eye_score': eye_score,
            'head_score': head_score, 'observation_scale': scale, 'localization': local,
        })
        if eye_score is None or head_score is None:
            evidence['reason'] = 'eye_detail_insufficient'
            return evidence
        scores.extend([eye_score, head_score])
    if scores:
        evidence.update(score=round(min(scores), 2), reason=None)
    else:
        evidence['reason'] = 'eye_unverified'
    return evidence


def score_elephants(image, detections, min_area_percent=1, *, detector=None):
    if not math.isfinite(min_area_percent) or not 0 <= min_area_percent <= 100:
        raise ValueError('Invalid minimum elephant area')
    # 共用原图分割轮廓提取；鸟类的整身完整性与评分结论不会用于大象。
    segmented = score_detections(image, detections, min_area_percent)
    animals = []
    detector = detector if detector is not None else ElephantLandmarkDetector()
    for raw in segmented['birds']:
        animal = {key: raw[key] for key in ('bbox', 'confidence', 'width', 'height', 'mask_outline', 'area_percent')}
        if 'frame_margin' in raw:
            animal['frame_margin'] = raw['frame_margin']
        animal.update(sharpness=None, focus_score=None, focus_verified=False,
                      focus_check_version=FOCUS_CHECK_VERSION, reason=None)
        if animal['area_percent'] + 1e-9 < min_area_percent:
            animal.update(reason='below_min_area', exclusion_reason='below_min_area')
        else:
            evidence = head_eye_evidence(image, animal, detector)
            animal.update(focus_evidence=evidence, sharpness=evidence.get('score'),
                          focus_score=evidence.get('score'), focus_verified=evidence.get('score') is not None,
                          reason=evidence['reason'])
        animals.append(animal)
    valid = [(i, animal) for i, animal in enumerate(animals) if animal['focus_verified']]
    best_index = max(valid, key=lambda item: item[1]['sharpness'])[0] if valid else None
    best = animals[best_index] if best_index is not None else None
    return assess_result({
        'width': image.width, 'height': image.height, 'elephants': animals, 'count': len(animals),
        'score_basis': 'elephant_head_eye', 'status': 'scored' if best else ('unjudgeable' if animals else 'no_elephant'),
        'score': best['sharpness'] if best else None, 'bbox': best['bbox'] if best else None,
        'best_index': best_index, 'min_area_percent': min_area_percent,
        'eligible_count': len(valid),
        'excluded_small_count': sum(a['reason'] == 'below_min_area' for a in animals),
        'focus_unverified_count': sum(a['reason'] not in (None, 'below_min_area') for a in animals),
    })


def assess_result(result, *, current=True):
    result = dict(result)
    reason = ('stale' if result.get('stale') else 'outdated' if result.get('outdated') else
              'unavailable' if not current or result.get('status') in (None, 'error') else None)
    animals = [dict(a) for a in result.get('elephants', [])]
    measured = []
    for index, animal in enumerate(animals):
        evidence = animal.get('focus_evidence') or {}
        eyes = evidence.get('eyes') or []
        values = [eye.get(key) for eye in eyes for key in ('eye_score', 'head_score')]
        valid = (not reason and animal.get('focus_verified')
                 and animal.get('focus_check_version') == FOCUS_CHECK_VERSION
                 and (evidence.get('landmarks') or {}).get('model') == LANDMARK_MODEL_ID
                 and not evidence.get('reason') and values
                 and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 10 for v in values))
        score = round(min(values), 2) if valid else None
        animal.update(focus_score=score, assessment='review',
                      assessment_reason=reason or animal.get('reason') or 'focus_borderline')
        if score is not None:
            measured.append((index, score))
            animal['assessment'] = 'pass' if score >= 7.5 else ('fail' if score <= 3 else 'review')
            animal['assessment_reason'] = ('complete_and_clear' if score >= 7.5 else
                                           'head_eye_blurred' if score <= 3 else 'focus_borderline')
    best_index, score = max(measured, key=lambda item: item[1]) if measured else (None, None)
    best = animals[best_index] if best_index is not None else None
    result.update(elephants=animals, score=score, focus_score=score, focus_score_index=best_index,
                  assessment_index=best_index, assessment=best['assessment'] if best else 'review',
                  assessment_reason=best['assessment_reason'] if best else
                  reason or (animals[0].get('reason') if animals else None) or 'no_eligible_elephant',
                  assessment_version=FOCUS_CHECK_VERSION,
                  assessment_counts={status: sum(a['assessment'] == status for a in animals)
                                     for status in ('pass', 'fail', 'review')})
    return result
