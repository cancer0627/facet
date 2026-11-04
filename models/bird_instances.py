"""逐只分离鸟体；矩形候选框只用于安排补看区域，不直接参与清晰度评分。"""

import math

import numpy as np


def instance_windows(size, proposals, enhanced=True):
    """整图检查，加上有界、去重的局部上下文；避免重复放大同一片鸟群。"""
    width, height = size
    whole = (0, 0, width, height)
    yield whole
    if not enhanced:
        return
    details = []
    for bird in sorted(proposals, key=lambda b: b['confidence'], reverse=True):
        x0, y0, x1, y1 = bird['box']
        if any(left <= x0 and top <= y0 and right >= x1 and bottom >= y1
               and min(x1 - x0, y1 - y0) * 768 / max(right - left, bottom - top) >= 96
               for left, top, right, bottom in details):
            continue
        w, h = min(width, max(768, math.ceil((x1 - x0) * 1.4))), min(height, max(768, math.ceil((y1 - y0) * 1.4)))
        left = max(0, min(width - w, int((x0 + x1 - w) / 2)))
        top = max(0, min(height - h, int((y0 + y1 - h) / 2)))
        window = (left, top, left + w, top + h)
        if window != whole and window not in details:
            details.append(window)
            yield window
        if len(details) >= 32:
            return


def _mask_overlap(a, b):
    """在共同区域对齐鸟体掩膜；交叉的矩形框不等于同一只鸟。"""
    import cv2

    ab, bb = a['box'], b['box']
    left, top = max(ab[0], bb[0]), max(ab[1], bb[1])
    right, bottom = min(ab[2], bb[2]), min(ab[3], bb[3])
    if right <= left or bottom <= top:
        return 0.0
    scale = min(1, 256 / max(right - left, bottom - top))
    xs = np.linspace(left, right, max(1, math.ceil((right - left) * scale)), endpoint=False)
    ys = np.linspace(top, bottom, max(1, math.ceil((bottom - top) * scale)), endpoint=False)
    masks = []
    for bird in (a, b):
        box, mask = bird['box'], bird['_mask']
        mx, my = np.meshgrid((xs - box[0]) * mask.shape[1] / (box[2] - box[0]),
                             (ys - box[1]) * mask.shape[0] / (box[3] - box[1]))
        masks.append(cv2.remap(mask.astype(np.uint8), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_NEAREST) > 0)
    overlap = np.count_nonzero(masks[0] & masks[1]) / scale ** 2
    areas = [np.mean(bird['_mask']) * (bird['box'][2] - bird['box'][0]) * (bird['box'][3] - bird['box'][1])
             for bird in (a, b)]
    return overlap / max(1, min(areas))


def merge_instances(candidates):
    """以鸟体相交去重；同一鸟体优先保留完整且分辨率较高的掩膜。"""
    def area(bird):
        box = bird['box']
        return float(np.mean(bird['_mask'])) * (box[2] - box[0]) * (box[3] - box[1])

    kept = []
    for bird in sorted(candidates, key=lambda b: (not b['truncated'], area(b)), reverse=True):
        duplicate = next((i for i, other in enumerate(kept) if _mask_overlap(bird, other) >= .7), None)
        if duplicate is not None:
            other = kept[duplicate]
            # 相近完整度时采用更精细的局部分割；小翅膀碎片不能覆盖完整鸟体。
            if (not bird['truncated'] and area(bird) >= .7 * area(other)
                    and bird['_mask'].sum() > other['_mask'].sum()):
                kept[duplicate] = bird
            continue
        kept.append(bird)
        if len(kept) >= 200:
            break
    return kept


class BirdInstanceSegmenter:
    def __init__(self, model_dir, cancelled=lambda: False):
        self.model_dir = model_dir
        self.cancelled = cancelled
        self.model = None

    def load(self):
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(self.model_dir),
                                                 check_hash=True, map_location='cpu', weights_only=True, progress=False)
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, min_size=512, max_size=768,
            rpn_pre_nms_top_n_test=500, rpn_post_nms_top_n_test=200, box_detections_per_img=50,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.bird_label = weights.meta['categories'].index('bird')

    def _detect_window(self, image, window):
        import torch
        from torchvision.transforms.functional import to_tensor

        if self.cancelled():
            raise InterruptedError('Bird detection cancelled')
        tile = image.crop(window).convert('RGB')
        tile.thumbnail((768, 768))
        with torch.inference_mode():
            output = self.model([to_tensor(tile)])[0]
        sx, sy = (window[2] - window[0]) / tile.width, (window[3] - window[1]) / tile.height
        birds = []
        keep = (output['labels'] == self.bird_label) & (output['scores'] >= .6)
        for index in keep.nonzero().flatten().tolist():
            box = output['boxes'][index].tolist()
            x0, y0, x1, y1 = (max(0, math.floor(box[0])), max(0, math.floor(box[1])),
                              min(tile.width, math.ceil(box[2])), min(tile.height, math.ceil(box[3])))
            if x1 <= x0 or y1 <= y0:
                continue
            mask = (output['masks'][index, 0, y0:y1, x0:x1] >= .5).numpy()
            if not mask.any():
                continue
            birds.append({
                'box': [window[0] + x0 * sx, window[1] + y0 * sy, window[0] + x1 * sx, window[1] + y1 * sy],
                'confidence': float(output['scores'][index]), '_mask': mask,
                'truncated': ((window[0] > 0 and x0 <= 3) or (window[1] > 0 and y0 <= 3)
                              or (window[2] < image.width and x1 >= tile.width - 3)
                              or (window[3] < image.height and y1 >= tile.height - 3)),
            })
        tile.close()
        return birds

    def detect(self, image, proposals, enhanced=True):
        if self.model is None:
            self.load()
        candidates = []
        for window in instance_windows(image.size, proposals, enhanced):
            candidates.extend(self._detect_window(image, window))
        return merge_instances(candidates)
