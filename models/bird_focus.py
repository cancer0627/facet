"""用鸟体轮廓核对局部细节证据；证据不足或矛盾时保留原始估计，交给人工复核。"""

import math
from functools import lru_cache

import numpy as np

FOCUS_CHECK_VERSION = 'outline-step-v1'


@lru_cache(maxsize=1)
def _outline_templates():
    from scipy.special import erf

    offsets = np.arange(-24, 25, dtype=np.float32)
    templates, widths = [], []
    for sigma in (.55, .8, 1.1, 1.5, 2., 2.8, 4., 6., 8., 12.):
        for centre in range(-8, 9, 2):
            profile = .5 * (1 + erf((offsets - centre) / (math.sqrt(2) * sigma)))
            profile = (profile - profile[:4].mean()) / (profile[-4:].mean() - profile[:4].mean())
            templates.append(profile)
            widths.append(sigma)
    return offsets, np.asarray(templates), np.asarray(widths)


def outline_evidence(image, bird):
    """读取原图真实轮廓两侧的过渡，不对涂黑背景后的边界评分。"""
    import cv2
    from PIL import Image

    box = bird['bbox']
    x0, y0 = math.floor(box[0] * image.width), math.floor(box[1] * image.height)
    x1, y1 = math.ceil(box[2] * image.width), math.ceil(box[3] * image.height)
    if x1 <= x0 or y1 <= y0 or not bird.get('mask_outline'):
        return {'sigma': None, 'coverage': 0., 'samples': 0}
    scale = min(1., 768 / max(x1 - x0, y1 - y0))
    padding = math.ceil(28 / scale)
    left, top = max(0, x0 - padding), max(0, y0 - padding)
    right, bottom = min(image.width, x1 + padding), min(image.height, y1 + padding)
    with image.crop((left, top, right, bottom)) as context:
        observed = context.convert('RGB').resize((max(1, round((right - left) * scale)),
                                                max(1, round((bottom - top) * scale))), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(observed), cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 7, 1).astype(np.float32)
    observed.close()
    mask = np.zeros(gray.shape, dtype=np.uint8)
    polygons = []
    for outline in bird['mask_outline']:
        if len(outline) >= 3:
            polygons.append(np.asarray([[(x * image.width - left) * scale,
                                         (y * image.height - top) * scale] for x, y in outline], dtype=np.int32))
    if not polygons:
        return {'sigma': None, 'coverage': 0., 'samples': 0}
    cv2.fillPoly(mask, polygons, 1)
    smooth = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    boundary = (mask > 0) & (cv2.erode(mask, np.ones((3, 3), dtype=np.uint8)) == 0)
    # 原图外没有可用像素，不能用填充色制造清楚的边界。
    boundary[:25] = boundary[-25:] = False
    boundary[:, :25] = boundary[:, -25:] = False
    ys, xs = np.nonzero(boundary)
    if len(xs) < 32:
        return {'sigma': None, 'coverage': 0., 'samples': 0}
    if len(xs) > 1000:
        chosen = np.linspace(0, len(xs) - 1, 1000, dtype=int)
        xs, ys = xs[chosen], ys[chosen]
    magnitude = np.maximum(np.hypot(gx[ys, xs], gy[ys, xs]), 1e-6)
    nx, ny = gx[ys, xs] / magnitude, gy[ys, xs] / magnitude
    offsets, templates, widths = _outline_templates()
    profiles = cv2.remap(gray, (xs[:, None] + nx[:, None] * offsets).astype(np.float32),
                         (ys[:, None] + ny[:, None] * offsets).astype(np.float32), cv2.INTER_LINEAR)
    outside, inside = profiles[:, :4].mean(axis=1), profiles[:, -4:].mean(axis=1)
    contrast = inside - outside
    normalized = (profiles - outside[:, None]) / np.where(abs(contrast[:, None]) > 1, contrast[:, None], 1)
    errors = ((normalized[:, None, :] - templates[None, :, :]) ** 2).mean(axis=2)
    best = errors.argmin(axis=1)
    valid = (abs(contrast) >= 20) & (errors[np.arange(len(best)), best] < .02)
    count, coverage = int(valid.sum()), float(valid.mean())
    # 允许头部清楚、翅膀运动，但不能只凭少数偶然对齐的背景边缘确认。
    sigma = float(np.percentile(widths[best][valid], 10)) if count >= 32 and coverage >= .15 else None
    return {'sigma': sigma, 'coverage': round(coverage, 4), 'samples': count}


def needs_focus_verification(result, min_area_percent):
    for bird in result.get('birds', []):
        box = bird['bbox']
        area = (box[2] - box[0]) * (box[3] - box[1]) * 100
        if (bird.get('sharpness') is not None and area + 1e-9 >= min_area_percent
                and bird.get('focus_check_version') != FOCUS_CHECK_VERSION):
            return True
    return False


def verify_focus(image, result, min_area_percent):
    result = dict(result)
    birds = [dict(bird) for bird in result['birds']]
    for bird in birds:
        box = bird['bbox']
        area = (box[2] - box[0]) * (box[3] - box[1]) * 100
        if (bird['sharpness'] is None or area + 1e-9 < min_area_percent
                or bird.get('focus_check_version') == FOCUS_CHECK_VERSION):
            continue
        evidence = outline_evidence(image, bird)
        local_sigma = 2 * math.sqrt(max(0, 10 / max(.01, bird['sharpness']) - 1))
        # 两种测量允许约一像素的定位差异及两倍尺度差异；不另行修改原始参考分。
        supported = evidence['sigma'] is not None and evidence['sigma'] <= max(local_sigma + 1, local_sigma * 2)
        bird.update({'focus_verified': bool(supported), 'focus_check_version': FOCUS_CHECK_VERSION,
                     'focus_evidence': evidence})
    result['birds'] = birds
    return result
