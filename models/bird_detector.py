"""本地鸟类定位和原图局部清晰度估计；不把检测置信度当作清晰度。"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_ID = 'torchvision-ssdlite320-maskrcnn-resnet50-fpn-v2-coco-v1'
SCORE_VERSION = 'bird-head-eye-edge-width-v9'
RESULT_VERSION_PREFIX = f'{MODEL_ID}:{SCORE_VERSION}:tiled='
DEFAULT_MIN_AREA_PERCENT = 1.0
INSTANCE_MAX_SIZE = 768
FRAME_EDGE_MARGIN = 3
MODEL_DIR = Path(__file__).resolve().parents[1] / 'pretrained_models' / 'birds'


def detection_windows(width, height, tiled=True):
    """整图加重叠分块；大图最多约 49 块，避免无界的推理开销。"""
    yield (0, 0, width, height)
    size = max(1024, math.ceil(max(width, height) / 5))
    if not tiled or max(width, height) <= size:
        return
    step = max(1, int(size * 0.8))

    def starts(length):
        positions = list(range(0, max(1, length - size + 1), step))
        return sorted(set(positions + [max(0, length - size)]))

    for y in starts(height):
        for x in starts(width):
            yield (x, y, min(width, x + size), min(height, y + size))


@lru_cache(maxsize=2)
def _edge_templates(radius):
    """高斯模糊后的阶跃边缘模板；固定尺度，跨照片使用同一把尺。"""
    from scipy.special import erf

    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    templates, widths = [], []
    for sigma in (0.55, 0.8, 1.1, 1.5, 2.0, 2.8, 4.0, 6.0):
        for centre in (-0.5, 0.0, 0.5):
            profile = 0.5 * (1 + erf((offsets - centre) / (math.sqrt(2) * sigma)))
            profile = (profile - profile[:2].mean()) / (profile[-2:].mean() - profile[:2].mean())
            templates.append(profile)
            widths.append(sigma)
    return offsets, np.asarray(templates), np.asarray(widths)


def local_sharpness(crop, mask=None, *, edge_percentile=25):
    """估计边缘过渡宽度；提供鸟体掩膜时，只接受完全位于鸟体内的边缘剖面。"""
    import cv2

    if min(crop.size) < 24 or max(crop.size) < 48:
        return None
    # 固定观察尺度，不放大小目标。平坦区域不进入边缘样本的平均。
    observed = crop.convert('RGB')
    observed.thumbnail((768, 768))
    rgb = np.asarray(observed)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # 保边去噪只抑制微小亮度波动；方向图上的平滑不参与宽度拟合。
    gray = cv2.bilateralFilter(gray, 5, 7, 1)
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.6)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, scale=1 / 8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, scale=1 / 8)
    edges = cv2.Canny(gray, 20, 50) > 0
    if mask is not None:
        # 不通过涂黑背景制造清楚的轮廓。只筛选样本位置，灰度数据保持原样。
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or min(mask.shape) < 24:
            return None
        foreground = cv2.resize(mask, observed.size, interpolation=cv2.INTER_NEAREST)
        # 分割图可能比原图粗很多；额外排除两格分割像素的误差，不能把粗掩膜放大当成精细分割。
        radius = max(8, math.ceil(2 * max(observed.width / mask.shape[1], observed.height / mask.shape[0])))
        interior = cv2.erode(foreground, np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)) > 0
        edges &= interior
    edges[:8] = edges[-8:] = False
    edges[:, :8] = edges[:, -8:] = False
    ys, xs = np.nonzero(edges)
    if len(xs) < 8:
        return None if mask is not None else 0.0
    if len(xs) > 3000:
        sample = np.linspace(0, len(xs) - 1, 3000, dtype=int)
        xs, ys = xs[sample], ys[sample]
    magnitude = np.maximum(np.hypot(gx[ys, xs], gy[ys, xs]), 1e-6)
    nx, ny = gx[ys, xs] / magnitude, gy[ys, xs] / magnitude
    estimates = []
    for radius in (3, 7):
        offsets, templates, widths = _edge_templates(radius)
        profiles = cv2.remap(
            gray.astype(np.float32),
            (xs[:, None] + nx[:, None] * offsets).astype(np.float32),
            (ys[:, None] + ny[:, None] * offsets).astype(np.float32), cv2.INTER_LINEAR,
        )
        left, right = profiles[:, :2].mean(axis=1), profiles[:, -2:].mean(axis=1)
        contrast = right - left
        normalized = (profiles - left[:, None]) / np.maximum(contrast[:, None], 1)
        errors = ((normalized[:, None, :] - templates[None, :, :]) ** 2).mean(axis=2)
        best = np.argmin(errors, axis=1)
        valid = (contrast > 12) & (errors[np.arange(len(best)), best] < 0.025)
        estimates.append((widths[best], valid))
    narrow, narrow_valid = estimates[0]
    wide, wide_valid = estimates[1]
    valid = narrow_valid & ((narrow < 1.5) | wide_valid)
    sigmas = np.where(narrow < 1.5, narrow, wide)[valid]
    # 随机噪点和交叉纹理通常无法形成可靠的单调过渡，不凭少数偶合样本给高分。
    if len(sigmas) < max(8, math.ceil(len(xs) * 0.05)):
        return None if mask is not None else 0.0
    # 鸟体诊断保留原统计；头眼使用中位数，避免少数锐利边缘撑起高分。
    sigma = float(np.percentile(sigmas, edge_percentile))
    return round(10.0 / (1.0 + (sigma / 2.0) ** 2), 2)


def merge_detections(candidates):
    """完整框优先于分块截断框，并抑制完整框内重复出现的翅膀碎片。"""
    kept = []
    for candidate in sorted(candidates, key=lambda b: (not b.get('truncated', False), b['confidence']), reverse=True):
        box = candidate['box']
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        if area <= 0:
            continue
        duplicate = False
        for other in kept:
            ob = other['box']
            overlap = max(0, min(box[2], ob[2]) - max(box[0], ob[0])) * max(0, min(box[3], ob[3]) - max(box[1], ob[1]))
            other_area = (ob[2] - ob[0]) * (ob[3] - ob[1])
            if (overlap / (area + other_area - overlap) >= 0.45
                    or (candidate.get('truncated') and overlap / area >= 0.8)):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) == 200:
            break
    return kept


def select_eligible_birds(result, min_area_percent=DEFAULT_MIN_AREA_PERCENT):
    """保留每只鸟的清晰度，只让达到用户大小门槛的鸟决定照片得分。大小使用定位框面积。"""
    if not math.isfinite(min_area_percent) or not 0 <= min_area_percent <= 100:
        raise ValueError('Minimum bird area must be between 0 and 100 percent')
    result = dict(result)
    birds = [dict(bird) for bird in result['birds']]
    for bird in birds:
        box = bird['bbox']
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1]) * 100
        bird['area_percent'] = round(area, 6)
        enough_area = area + 1e-9 >= min_area_percent
        frame_edge = bird.get('frame_edge', False) or bird.get('reason') == 'frame_edge'
        bird['eligible'] = (bird['sharpness'] is not None and enough_area
                            and not frame_edge and bird.get('focus_verified') is not False)
        bird['exclusion_reason'] = (
            'frame_edge' if frame_edge else
            'below_min_area' if not enough_area and (
                bird['sharpness'] is not None or bird.get('body_sharpness') is not None) else
            'focus_unverified' if enough_area and bird.get('focus_verified') is False else None
        )
    valid = [(i, bird) for i, bird in enumerate(birds) if bird['eligible']]
    best_index = max(valid, key=lambda item: (
        item[1]['sharpness'], min(item[1]['width'], item[1]['height']),
    ))[0] if valid else None
    best = birds[best_index] if best_index is not None else None
    result.update({
        'birds': birds, 'best_index': best_index, 'count': len(birds),
        'status': 'scored' if best else ('unjudgeable' if birds else 'no_bird'),
        'score': best['sharpness'] if best else None, 'bbox': best['bbox'] if best else None,
        'min_area_percent': min_area_percent, 'eligible_count': len(valid),
        'excluded_small_count': sum(bird['exclusion_reason'] == 'below_min_area' for bird in birds),
        'focus_unverified_count': sum(bird['exclusion_reason'] == 'focus_unverified' for bird in birds),
    })
    return result


def guard_frame_edges(result, min_area_percent=DEFAULT_MIN_AREA_PERCENT):
    """按检测尺度排除疑似截断主体；保留局部细节估计，但不让残缺翅膀决定照片得分。"""
    result = dict(result)
    width, height = result['width'], result['height']
    birds = [dict(bird) for bird in result['birds']]
    # 旧结果未保存分割图比例，按整图最长边缩至 768 的尺度保守复核。
    legacy_margin = FRAME_EDGE_MARGIN * max(1, max(width, height) / INSTANCE_MAX_SIZE)
    for bird in birds:
        box = bird['bbox']
        margin_x, margin_y = bird.get('frame_margin', [legacy_margin, legacy_margin])
        bird['frame_edge'] = bool(
            box[0] * width <= margin_x or box[1] * height <= margin_y
            or (1 - box[2]) * width <= margin_x or (1 - box[3]) * height <= margin_y
        )
    result['birds'] = birds
    return select_eligible_birds(result, min_area_percent)


def score_detections(image, detections, min_area_percent=DEFAULT_MIN_AREA_PERCENT):
    """只对已分离鸟体且有可靠内部细节的目标评分；不回退到矩形背景评分。"""
    import cv2

    width, height = image.size
    birds = []
    for detection in detections:
        x0, y0, x1, y1 = detection['box']
        x0, x1 = max(0, math.floor(x0)), min(width, math.ceil(x1))
        y0, y1 = max(0, math.floor(y0)), min(height, math.ceil(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        truncated = bool(detection.get('truncated'))
        mask = detection.get('_mask')
        reason = 'truncated' if truncated else ('no_mask' if mask is None else None)
        if reason is None and min(mask.shape) < 24:
            reason = 'low_mask_resolution'
        with image.crop((x0, y0, x1, y1)) as crop:
            score = None if reason else local_sharpness(crop, mask)
            if reason is None and score is None:
                reason = 'too_small' if min(crop.size) < 24 or max(crop.size) < 48 else 'insufficient_detail'
        outlines = []
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
                if cv2.contourArea(contour) < 4:
                    continue
                points = cv2.approxPolyDP(contour, 1.0, True)[:, 0, :]
                outlines.append([[(x0 + float(x) * (x1 - x0) / mask.shape[1]) / width,
                                  (y0 + float(y) * (y1 - y0) / mask.shape[0]) / height] for x, y in points])
        bird = {
            'bbox': [x0 / width, y0 / height, x1 / width, y1 / height],
            'confidence': round(float(detection['confidence']), 4),
            'sharpness': score, 'width': x1 - x0, 'height': y1 - y0,
            'reason': reason, 'mask_outline': outlines,
        }
        if 'frame_margin' in detection:
            bird['frame_margin'] = list(detection['frame_margin'])
        birds.append(bird)
    return guard_frame_edges({
        'width': width, 'height': height, 'birds': birds,
        'score_basis': 'bird_interior',
    }, min_area_percent)


class BirdDetector:
    species = 'bird'

    def __init__(self, tiled=True, cancelled=lambda: False):
        self.tiled = tiled
        self.cancelled = cancelled
        self.model = None
        self.segmenter = None

    def load(self):
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large,
        )

        # 使用现有运行库，权重存入项目模型目录；只从官方地址下载并验证哈希。
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        state = torch.hub.load_state_dict_from_url(
            weights.url, model_dir=str(MODEL_DIR), check_hash=True,
            progress=False, map_location='cpu', weights_only=True,
        )
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.model.load_state_dict(state)
        self.model.eval()
        self.bird_label = weights.meta['categories'].index(self.species)
        # 该轻量模型在处理器运行，保持跨设备结果一致，避免占用其他评分模型的显存。
        torch.set_num_threads(min(4, torch.get_num_threads()))
        from models.bird_instances import BirdInstanceSegmenter
        self.segmenter = BirdInstanceSegmenter(MODEL_DIR, self.cancelled, species=self.species)
        self.segmenter.load()

    def _detect_window(self, image, window):
        import torch
        from torchvision.transforms.functional import to_tensor

        if self.cancelled():
            raise InterruptedError('Bird detection cancelled')
        x0, y0, x1, y1 = window
        tile = image.crop(window).convert('RGB')
        tile.thumbnail((1024, 1024))
        scale_x, scale_y = (x1 - x0) / tile.width, (y1 - y0) / tile.height
        with torch.inference_mode():
            output = self.model([to_tensor(tile)])[0]
        keep = (output['labels'] == self.bird_label) & (output['scores'] >= 0.5)
        candidates = []
        for box, confidence in zip(output['boxes'][keep].tolist(), output['scores'][keep].tolist()):
            truncated = ((x0 > 0 and box[0] <= 3) or (y0 > 0 and box[1] <= 3)
                         or (x1 < image.width and box[2] >= tile.width - 3)
                         or (y1 < image.height and box[3] >= tile.height - 3))
            candidates.append({
                'box': [x0 + box[0] * scale_x, y0 + box[1] * scale_y,
                        x0 + box[2] * scale_x, y0 + box[3] * scale_y],
                'confidence': confidence, 'truncated': truncated,
            })
        return candidates

    def detect(self, image):
        if self.model is None:
            self.load()
        candidates = []
        windows = set(detection_windows(*image.size, self.tiled))
        for window in sorted(windows):
            candidates.extend(self._detect_window(image, window))
        # 对分块边界上的目标补看更宽的上下文，恢复完整头部与翅膀；限制额外开销。
        partials = sorted((b for b in candidates if b['truncated']), key=lambda b: b['confidence'], reverse=True)
        for bird in partials[:8]:
            x0, y0, x1, y1 = bird['box']
            # 沿目标原本的长宽各扩展上下文；正方形会引入过多相邻海鸥，妨碍恢复。
            width = max(1024, math.ceil(1.6 * (x1 - x0)))
            height = max(1024, math.ceil(1.6 * (y1 - y0)))
            left = max(0, min(image.width - width, int((x0 + x1 - width) / 2)))
            top = max(0, min(image.height - height, int((y0 + y1 - height) / 2)))
            window = (left, top, min(image.width, left + width), min(image.height, top + height))
            if window not in windows:
                windows.add(window)
                candidates.extend(self._detect_window(image, window))
        proposals = merge_detections(candidates)
        if self.segmenter is None:
            from models.bird_instances import BirdInstanceSegmenter
            self.segmenter = BirdInstanceSegmenter(MODEL_DIR, self.cancelled, species=self.species)
        instances = self.segmenter.detect(image, proposals, enhanced=self.tiled)
        # 分割没有确认鸟体时，候选仍供人工查看，但不产生清晰度分数。
        return instances or proposals
