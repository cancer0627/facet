"""用原图眼部结构与局部关键点定位核验眼睛；不按清晰度挑测量位置。"""

import math

import numpy as np

MIN_VISIBILITY = .65
MAX_CANDIDATES = 8


def eye_candidates(image, bird, centre, distance):
    """在有界邻域内找暗色紧凑结构，所有像素来自未改动的原图。"""
    import cv2
    from PIL import Image

    radius = 3.5 * distance
    left, top = max(0, math.floor(centre[0] - radius)), max(0, math.floor(centre[1] - radius))
    right, bottom = min(image.width, math.ceil(centre[0] + radius)), min(image.height, math.ceil(centre[1] + radius))
    scale = min(1., 1024 / max(right - left, bottom - top))
    size = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
    with image.resize(size, Image.Resampling.BILINEAR, box=(left, top, right, bottom)) as crop:
        gray = cv2.cvtColor(np.asarray(crop.convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), .7)
    sx, sy = size[0] / (right - left), size[1] / (bottom - top)
    observed_distance = distance * min(sx, sy)
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold, params.maxThreshold, params.thresholdStep = 10, 220, 10
    params.minRepeatability = 2
    params.filterByColor, params.blobColor = True, 0
    params.filterByArea = True
    params.minArea = max(9, math.pi * (observed_distance * .025) ** 2)
    params.maxArea = math.pi * (observed_distance * .35) ** 2
    if params.maxArea <= params.minArea:
        return []
    params.filterByCircularity, params.minCircularity = True, .5
    params.filterByConvexity, params.minConvexity = True, .75
    params.filterByInertia, params.minInertiaRatio = True, .35
    # 候选提取时填补眼内的小高光，避免高光轮廓吞掉暗色虹膜候选。
    # 亮度核验仍读原始灰度图；这一步不参与清晰度测量。
    kernel_size = max(3, round(observed_distance * .12) | 1)
    detection_gray = cv2.morphologyEx(
        gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    polygons = [np.asarray([[(x * image.width), (y * image.height)] for x, y in poly], dtype=np.float32)
                for poly in bird['mask_outline'] if len(poly) >= 3]
    candidates = []
    for blob in cv2.SimpleBlobDetector_create(params).detect(detection_gray):
        x, y = blob.pt
        point = np.asarray([left + x / sx, top + y / sy])
        diameter = blob.size / min(sx, sy)
        offset = float(np.linalg.norm(point - centre))
        if offset > radius or not any(cv2.pointPolygonTest(poly, tuple(point), False) >= 0 for poly in polygons):
            continue
        extent = math.ceil(blob.size)
        ix, iy = round(x), round(y)
        if ix < extent or iy < extent or ix + extent >= size[0] or iy + extent >= size[1]:
            continue
        patch = gray[iy - extent:iy + extent + 1, ix - extent:ix + extent + 1]
        yy, xx = np.indices(patch.shape)
        radii = np.hypot(xx - extent, yy - extent)
        inner = patch[radii <= .3 * blob.size]
        ring = patch[(radii >= .65 * blob.size) & (radii <= .95 * blob.size)]
        contrast = float(np.median(ring) - np.median(inner))
        if contrast < 12:
            continue
        candidates.append({'centre': point.tolist(), 'diameter': diameter,
                           'contrast': contrast, 'offset': offset})
    # 排序只用位置，不使用边缘宽度或清晰度，避免把较清楚的耳斑当眼睛。
    return sorted(candidates, key=lambda item: item['offset'])[:MAX_CANDIDATES]


def verify_eye(image, bird, centre, distance, detector, eye_index):
    """近处结构佐证原定位；远处修正还必须得到局部模型的明确支持。"""
    candidates = eye_candidates(image, bird, centre, distance)
    accepted = []
    for candidate in candidates:
        point = np.asarray(candidate['centre'])
        diameter = candidate['diameter']
        radius = max(3 * diameter, .9 * distance)
        rect = (max(0, math.floor(point[0] - radius)), max(0, math.floor(point[1] - radius)),
                min(image.width, math.ceil(point[0] + radius)), min(image.height, math.ceil(point[1] + radius)))
        with image.crop(rect) as context:
            landmarks = detector.inspect_eye(context)
            size = np.asarray(context.size)
        points = np.asarray(landmarks['points'], dtype=float)
        views = np.asarray(landmarks['views'], dtype=float)
        visibility = np.asarray(landmarks['visibility'], dtype=float)
        if (points.shape != (3, 2) or views.shape != (2, 3, 2) or visibility.shape != (3,)
                or not np.isfinite(points).all() or not np.isfinite(views).all()
                or not np.isfinite(visibility).all() or (views < 0).any() or (views > 1).any()):
            candidate.update(accepted=False, reason='invalid_localization')
            continue
        observed = views * size + np.asarray(rect[:2])
        offsets = np.linalg.norm(observed[:, eye_index] - point, axis=1)
        eye_drift = float(np.linalg.norm(observed[0, eye_index] - observed[1, eye_index]))
        beak_drift = float(np.linalg.norm(observed[0, 2] - observed[1, 2]))
        eye_beak = float(np.linalg.norm((points[eye_index] - points[2]) * size))
        eye_confident = visibility[eye_index] >= MIN_VISIBILITY
        agrees = bool(max(offsets) <= diameter and eye_drift <= .5 * diameter)
        local_support = bool(eye_confident and visibility[2] >= MIN_VISIBILITY and agrees
                             and beak_drift <= .5 * distance and .5 * diameter <= eye_beak <= 2 * distance)
        # 局部裁剪可能缺少完整鸟体；低置信度不推翻已与原定位吻合的眼部结构。
        # 但局部模型明确指出眼睛在别处时，近处暗斑也不能通过。
        nearby_support = candidate['offset'] <= diameter and not (eye_confident and not agrees)
        candidate.update(accepted=bool(local_support or nearby_support),
                         support='local_model' if local_support else ('coarse_and_structure' if nearby_support else None),
                         local_visibility=visibility.tolist(), local_offsets=offsets.tolist(), eye_drift=eye_drift)
        if candidate['accepted']:
            accepted.append(candidate)
    evidence = {'candidates': candidates, 'search_radius': 3.5 * distance}
    if not accepted:
        return None, evidence
    best = accepted[0]
    if any(np.linalg.norm(np.asarray(item['centre']) - best['centre']) > max(item['diameter'], best['diameter'])
           for item in accepted[1:]):
        evidence['ambiguous'] = True
        return None, evidence
    evidence['selected_centre'] = best['centre']
    evidence['selected_diameter'] = best['diameter']
    return np.asarray(best['centre']), evidence
