"""海鸥选片三态：完整入镜且头眼清楚才合格，不确定的测量保留复核。"""

import math

from models.bird_focus import FOCUS_CHECK_VERSION
from models.bird_landmarks import LANDMARK_MODEL_ID

ASSESSMENT_VERSION = 'bird-selection-v3-score10'
# 首轮保守工程区间，尚未经独立人工标注集校准；中间区域不强行二分。
BLURRED_MAX = 3.0
CLEAR_MIN = 7.5


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _confirmed_regions(bird, size):
    evidence = bird.get('focus_evidence') or {}
    if not isinstance(evidence, dict) or not isinstance(evidence.get('landmarks'), dict):
        return []
    if (bird.get('focus_check_version') != FOCUS_CHECK_VERSION
            or (evidence.get('landmarks') or {}).get('model') != LANDMARK_MODEL_ID):
        return []
    regions = []
    if not isinstance(evidence.get('eyes'), list):
        return []
    for eye in evidence['eyes']:
        if not isinstance(eye, dict):
            continue
        centre = eye.get('centre', [])
        verification = eye.get('verification') or {}
        if not isinstance(verification, dict):
            continue
        selected = verification.get('selected_centre', [])
        diameter = verification.get('selected_diameter')
        if (not isinstance(centre, (list, tuple)) or not isinstance(selected, (list, tuple))
                or len(centre) != 2 or len(selected) != 2 or not all(_number(x) for x in [*centre, *selected])
                or not all(0 <= x <= 1 for x in centre) or not _number(diameter) or diameter <= 0):
            continue
        if math.hypot(centre[0] * size[0] - selected[0], centre[1] * size[1] - selected[1]) > 1:
            continue
        regions.append(eye)
    return regions


def assess_bird(bird, size):
    box = bird.get('bbox') or []
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(_number(x) and 0 <= x <= 1 for x in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        return 'review', 'body_uncertain'
    if min(box[:2]) <= 0 or max(box[2:]) >= 1:
        return 'fail', 'body_clipped'
    if (bird.get('frame_edge') or bird.get('reason') in ('frame_edge', 'truncated')
            or bird.get('exclusion_reason') == 'frame_edge' or not bird.get('mask_outline')):
        return 'review', 'body_uncertain'
    regions = _confirmed_regions(bird, size)
    measured = [eye.get(key) for eye in regions for key in ('head_score', 'eye_score')]
    valid = [value for value in measured if _number(value) and 0 <= value <= 10]
    # 可靠定位后，任一头部或眼周明确模糊就足以否定；另一部位清楚不能补偿。
    if any(value <= BLURRED_MAX for value in valid):
        return 'fail', 'head_eye_blurred'
    evidence = bird.get('focus_evidence') or {}
    if not isinstance(evidence, dict):
        return 'review', 'head_eye_uncertain'
    if (not regions or len(regions) != len(evidence.get('eyes', [])) or not bird.get('focus_verified')
            or evidence.get('reason') or len(valid) != 2 * len(regions)):
        return 'review', 'head_eye_uncertain'
    if all(value >= CLEAR_MIN for value in valid):
        return 'pass', 'complete_and_clear'
    return 'review', 'focus_borderline'


def measured_focus_score(bird, size):
    """有效头眼测量使用 0–10 分制；不是分类置信度，不把未知区域填成零分。"""
    regions = _confirmed_regions(bird, size)
    evidence = bird.get('focus_evidence') or {}
    if (not regions or not bird.get('focus_verified') or not isinstance(evidence, dict)
            or evidence.get('reason') or len(regions) != len(evidence.get('eyes', []))):
        return None
    values = [eye.get(key) for eye in regions for key in ('head_score', 'eye_score')]
    if not all(_number(value) and 0 <= value <= 10 for value in values):
        return None
    return round(min(values), 2)


def assess_result(result, *, current=True):
    """从同一份有效证据派生结论；不改写内部参考分、不用整鸟分数代替眼部判断。"""
    result = dict(result)
    raw_birds = result.get('birds')
    birds = [dict(bird) if isinstance(bird, dict) else {} for bird in raw_birds] if isinstance(raw_birds, list) else []
    unavailable = ('stale' if result.get('stale') else 'outdated' if result.get('outdated') else
                   'unavailable' if not current or result.get('status') in ('error', 'unprocessed', None) else None)
    width, height = result.get('width'), result.get('height')
    if not all(_number(x) and x > 0 for x in (width, height)):
        unavailable = unavailable or 'unavailable'
    min_area = result.get('min_area_percent', 1)
    if not _number(min_area) or not 0 <= min_area <= 100:
        unavailable = unavailable or 'unavailable'
    candidates = []
    for index, bird in enumerate(birds):
        bird['focus_score'] = None
        if unavailable:
            status, reason = 'review', unavailable
        else:
            box = bird.get('bbox') or []
            area = ((box[2] - box[0]) * (box[3] - box[1]) * 100
                    if isinstance(box, (list, tuple)) and len(box) == 4 and all(_number(x) for x in box) else None)
            if area is not None and 0 <= area < min_area - 1e-9:
                bird.update(assessment='review', assessment_reason='below_min_area')
                continue
            status, reason = assess_bird(bird, (width, height))
            if reason not in ('body_clipped', 'body_uncertain'):
                bird['focus_score'] = measured_focus_score(bird, (width, height))
        bird.update(assessment=status, assessment_reason=reason)
        candidates.append((index, status, reason))
    # 照片里有一只合格的候选即合格；仍有未确定候选时，不能因为另一只失败而否定全图。
    selected = next((x for x in candidates if x[1] == 'pass'), None)
    if selected is None:
        selected = next((x for x in candidates if x[1] == 'review'), None)
    if selected is None and candidates:
        selected = candidates[0]
    index, status, reason = selected if selected else (None, 'review', unavailable or 'no_eligible_bird')
    measured = [(i, bird['focus_score']) for i, bird in enumerate(birds) if bird['focus_score'] is not None]
    focus_index, focus_score = max(measured, key=lambda x: x[1]) if measured else (None, None)
    result.update(birds=birds, assessment=status, assessment_reason=reason, assessment_index=index,
                  focus_score=focus_score, focus_score_index=focus_index,
                  assessment_version=ASSESSMENT_VERSION,
                  assessment_counts={s: sum(x[1] == s for x in candidates) for s in ('pass', 'fail', 'review')})
    return result
