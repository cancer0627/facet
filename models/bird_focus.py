"""以稳定定位的头眼为评分区域；鸟体细节只保留作诊断。"""

import math

import numpy as np

from models.bird_landmarks import BirdLandmarkDetector
from models.bird_eye import MIN_VISIBILITY, verify_eye

FOCUS_CHECK_VERSION = 'head-eye-v3'


def _candidate(bird, min_area_percent):
    box = bird['bbox']
    area = (box[2] - box[0]) * (box[3] - box[1]) * 100
    return (not bird.get('frame_edge') and bird.get('reason') not in ('frame_edge', 'truncated')
            and area + 1e-9 >= min_area_percent)


def needs_focus_verification(result, min_area_percent):
    return any(_candidate(bird, min_area_percent) and bird.get('focus_check_version') != FOCUS_CHECK_VERSION
               for bird in result.get('birds', []))


def _region_score(image, bird, centre, radius, scale):
    """在原图尺度测量圆形区域；分割误差、圆边界和矩形背景均不参与。"""
    import cv2
    from PIL import Image
    from models.bird_detector import local_sharpness

    left, top = math.floor(centre[0] - radius), math.floor(centre[1] - radius)
    right, bottom = math.ceil(centre[0] + radius), math.ceil(centre[1] + radius)
    if left < 0 or top < 0 or right > image.width or bottom > image.height or radius < 24:
        return None
    size = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
    with image.resize(size, Image.Resampling.LANCZOS, box=(left, top, right, bottom)) as crop:
        sx, sy = crop.width / (right - left), crop.height / (bottom - top)
        mask = np.zeros((crop.height, crop.width), dtype=np.uint8)
        polygons = [np.asarray([[(x * image.width - left) * sx, (y * image.height - top) * sy]
                               for x, y in outline],
                               dtype=np.int32) for outline in bird['mask_outline'] if len(outline) >= 3]
        if not polygons:
            return None
        cv2.fillPoly(mask, polygons, 1)
        # 原分割的一格不等于原图一像素；排除约两格的轮廓定位误差。
        margins = bird.get('frame_margin', [3 * max(image.size) / 768] * 2)
        uncertainty = max(1, math.ceil(2 * max(margins) * max(sx, sy) / 3))
        mask = cv2.erode(mask, np.ones((2 * uncertainty + 1, 2 * uncertainty + 1), dtype=np.uint8))
        yy, xx = np.indices(mask.shape)
        mask &= (((xx / sx - (centre[0] - left)) ** 2 + (yy / sy - (centre[1] - top)) ** 2)
                 <= radius ** 2)
        return local_sharpness(crop, mask, edge_percentile=50)


def _located_eyes(image, bird, landmarks, rect):
    """统一核验每一对观察；补充定位也不能绕过可见性、位置与一致性条件。"""
    import cv2

    left, top, right, bottom = rect
    points = np.asarray(landmarks['points'], dtype=float)
    views = np.asarray(landmarks['views'], dtype=float)
    visibility = np.asarray(landmarks['visibility'], dtype=float)
    if (points.shape != (3, 2) or views.shape != (2, 3, 2) or visibility.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(views).all()
            or not np.isfinite(visibility).all() or (points < 0).any() or (points > 1).any()
            or (views < 0).any() or (views > 1).any() or visibility[2] < MIN_VISIBILITY):
        return []
    size, offset = np.asarray([right - left, bottom - top]), np.asarray([left, top])
    points = points * size + offset
    disagreement = np.linalg.norm((views[0] - views[1]) * size, axis=1)
    polygons = [np.asarray([[(x * image.width), (y * image.height)] for x, y in outline], dtype=np.float32)
                for outline in bird['mask_outline'] if len(outline) >= 3]
    eyes = []
    for index in (0, 1):
        distance = float(np.linalg.norm(points[index] - points[2]))
        # 没有可靠眼喙距离时不能拿鸟框大小猜头的位置；不同观察不一致也不评分。
        if (visibility[index] < MIN_VISIBILITY or not 54 <= distance <= .5 * min(size)
                or max(disagreement[index], disagreement[2]) > .5 * distance
                or not any(cv2.pointPolygonTest(poly, tuple(points[index]), False) >= 0 for poly in polygons)):
            continue
        eyes.append((index, points[index], distance))
    return eyes


def _refined_consensus(image, bird, original, isolated, rect):
    """原像素与抑制背景后的观察相互印证；不单凭处理后的图像确认。"""
    candidates = []
    for raw_index in (0, 1):
        for isolated_index in (0, 1):
            raw = original['views'][raw_index]
            foreground = isolated['views'][isolated_index]
            visibility = np.minimum(original['view_visibility'][raw_index],
                                    isolated['view_visibility'][isolated_index])
            landmarks = {
                'points': ((np.asarray(raw) + foreground) / 2).tolist(),
                'visibility': visibility.tolist(), 'views': [raw, foreground],
                'view_visibility': [original['view_visibility'][raw_index],
                                    isolated['view_visibility'][isolated_index]],
                'model': original['model'],
                'support_views': ['original' if raw_index == 0 else 'original_mirrored',
                                  'isolated' if isolated_index == 0 else 'isolated_mirrored'],
            }
            eyes = _located_eyes(image, bird, landmarks, rect)
            if eyes:
                confidence = min(min(visibility[index], visibility[2]) for index, _, _ in eyes)
                candidates.append((landmarks, eyes, confidence))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (len(item[1]), item[2]))
    # 重叠鸟若出现相互矛盾的可靠头部，保留待复核，不任意挑一个位置。
    for _, eyes, _ in candidates:
        for index, centre, distance in eyes:
            for best_index, best_centre, best_distance in best[1]:
                if index == best_index and np.linalg.norm(centre - best_centre) > .5 * min(distance, best_distance):
                    return None
    return best[0]


def head_eye_evidence(image, bird, detector):
    """双视图一致、位于鸟体内的眼睛和喙，才能定义头眼测量范围。"""
    box = bird['bbox']
    left, top = math.floor(box[0] * image.width), math.floor(box[1] * image.height)
    right, bottom = math.ceil(box[2] * image.width), math.ceil(box[3] * image.height)
    rect = (left, top, right, bottom)
    if (not bird.get('mask_outline') or min(right - left, bottom - top) < 48
            or not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height):
        return {'reason': 'head_unlocalized'}
    with image.crop(rect) as crop:
        landmarks = detector.detect(crop)
        eyes = _located_eyes(image, bird, landmarks, rect)
        attempts = []
        if not eyes and 'view_visibility' in landmarks and hasattr(detector, 'refine'):
            outline = [[((x * image.width - left) / crop.width, (y * image.height - top) / crop.height)
                        for x, y in polygon] for polygon in bird['mask_outline']]
            isolated = detector.refine(crop, outline)
            attempts = [landmarks, isolated]
            consensus = _refined_consensus(image, bird, landmarks, isolated, rect)
            if consensus is not None:
                landmarks = consensus
                eyes = _located_eyes(image, bird, landmarks, rect)
    evidence = {'landmarks': landmarks, 'eyes': [], 'reason': 'head_unlocalized'}
    if attempts:
        evidence['localization_attempts'] = attempts
    if not eyes:
        return evidence
    scores = []
    for index, centre, distance in eyes:
        verified_centre, eye_evidence = verify_eye(image, bird, centre, distance, detector, index)
        if verified_centre is None:
            evidence['eye_verification'] = eye_evidence
            evidence['reason'] = 'eye_unverified'
            return evidence
        centre = verified_centre
        # 眼周分数为上限；羽毛、喙或另一只眼的细节不能抵消眼周模糊。
        eye_radius, head_radius = .45 * distance, 1.2 * distance
        scale = min(1., 768 / (2 * head_radius + 2))
        eye_score = _region_score(image, bird, centre, eye_radius, scale)
        head_score = _region_score(image, bird, centre, head_radius, scale)
        evidence['eyes'].append({
            'index': index, 'centre': [float(centre[0] / image.width), float(centre[1] / image.height)],
            'eye_radius': eye_radius, 'head_radius': head_radius,
            'eye_score': eye_score, 'head_score': head_score, 'observation_scale': scale,
            'verification': eye_evidence,
        })
        if eye_score is None or head_score is None:
            evidence['reason'] = 'eye_detail_insufficient'
            return evidence
        scores.extend([eye_score, head_score])
    evidence.update(score=min(scores), reason=None)
    return evidence


def verify_focus(image, result, min_area_percent, *, detector=None):
    result = dict(result)
    birds = [dict(bird) for bird in result['birds']]
    detector = detector if detector is not None else BirdLandmarkDetector()
    for bird in birds:
        if bird.get('focus_check_version') != FOCUS_CHECK_VERSION:
            bird.setdefault('body_sharpness', bird['sharpness'])
            bird['sharpness'] = None
            bird.pop('focus_verified', None)
        if not _candidate(bird, min_area_percent) or bird.get('focus_check_version') == FOCUS_CHECK_VERSION:
            continue
        evidence = head_eye_evidence(image, bird, detector)
        bird.update(sharpness=evidence.get('score'), focus_verified=evidence.get('score') is not None,
                    focus_check_version=FOCUS_CHECK_VERSION, focus_evidence=evidence,
                    reason=evidence['reason'])
    result.update(birds=birds, score_basis='bird_head_eye')
    from models.bird_detector import select_eligible_birds

    return select_eligible_birds(result, min_area_percent)
