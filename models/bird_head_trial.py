"""独立头部范围定位试验；不由眼点反推头框，不接入正式评分。"""

from pathlib import Path

import numpy as np

from models.weights import PRETRAINED_MODELS_DIR

HEAD_REPOSITORY = 'google/owlv2-base-patch16-ensemble'
HEAD_REVISION = 'cfd3195ba4ea9592eec887ded089f4c08eff231d'
HEAD_WEIGHTS_SHA256 = 'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
HEAD_MODEL_PATH = PRETRAINED_MODELS_DIR / 'birds' / 'owlv2-base-patch16-ensemble'
HEAD_QUERIES = ("a bird's head", "a seagull's head", 'a bird', "a bird's wing")


def decode_heads(logits, boxes, image_size, threshold=.1, limit=4):
    """模型先补成正方形；坐标必须按原图长边映射，不分别拉伸横纵轴。"""
    logits, boxes = np.asarray(logits), np.asarray(boxes)
    if (logits.ndim != 2 or logits.shape[1] != len(HEAD_QUERIES)
            or boxes.shape != (len(logits), 4)):
        raise ValueError('Unexpected independent head detector output shape')
    width, height = image_size
    if min(width, height) <= 0:
        raise ValueError('Invalid image size')
    scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    starts = (boxes[:, :2] - boxes[:, 2:] / 2) * max(width, height)
    ends = (boxes[:, :2] + boxes[:, 2:] / 2) * max(width, height)
    finite = np.isfinite(logits).all(axis=1) & np.isfinite(boxes).all(axis=1)
    records = []
    for index in np.flatnonzero(finite):
        query = int(np.argmax(scores[index, :2]))
        score = float(scores[index, query])
        if score < threshold:
            continue
        rect = [float(np.clip(starts[index, 0], 0, width)), float(np.clip(starts[index, 1], 0, height)),
                float(np.clip(ends[index, 0], 0, width)), float(np.clip(ends[index, 1], 0, height))]
        area = max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])
        # 仅是试验门槛：整鸟框不能冒充头框；拒绝项也保留，便于复核。
        reason = 'empty_box' if area == 0 else ('whole_subject_box' if area > width * height * .35 else None)
        if reason is None and score <= max(scores[index, 2:]):
            reason = 'non_head_label_wins'
        records.append({'bbox': rect, 'score': score, 'query': HEAD_QUERIES[query], 'reason': reason,
                        'all_scores': scores[index].tolist()})
    records.sort(key=lambda item: item['score'], reverse=True)
    accepted = []
    for record in records:
        if record['reason']:
            continue
        a = np.asarray(record['bbox'])
        for previous in accepted:
            b = np.asarray(previous['bbox'])
            intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
            union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
            if intersection / union > .3:
                record['reason'] = 'duplicate'
                break
        if record['reason'] is None:
            if len(accepted) < limit:
                accepted.append(record)
            else:
                record['reason'] = 'candidate_limit'
    return {'heads': accepted, 'proposals': records[:40]}


def map_head_landmarks(landmarks, head_rect, bird_size):
    """局部头框关键点映射回同一鸟体裁剪；保持原图、翻转两个观察独立。"""
    left, top, right, bottom = head_rect
    size = np.asarray([right - left, bottom - top])
    if min(size) <= 0:
        raise ValueError('Invalid head crop')
    result = dict(landmarks)
    for key in ('points', 'views'):
        values = np.asarray(landmarks[key], dtype=float)
        expected = (3, 2) if key == 'points' else (2, 3, 2)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError('Invalid head landmark coordinates')
        result[key] = ((values * size + [left, top]) / bird_size).tolist()
    return result


class IndependentBirdHeadTrial:
    """使用固定公开权重按文字定位头部；仅在隔离试验环境加载。"""

    def __init__(self, model_path=HEAD_MODEL_PATH, cancelled=lambda: False):
        self.path = Path(model_path)
        self.cancelled = cancelled
        self.model = self.processor = None

    def load(self):
        if self.cancelled():
            raise InterruptedError('Head trial cancelled')
        if self.model is not None:
            return
        import transformers
        if transformers.__version__ != '4.44.2':
            raise RuntimeError('Run the head trial in the isolated transformers==4.44.2 environment')
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        torch.set_num_threads(4)
        self.processor = Owlv2Processor.from_pretrained(self.path, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.path, local_files_only=True, use_safetensors=True,
        ).eval()

    def detect(self, crop):
        self.load()
        import torch

        inputs = self.processor(text=[list(HEAD_QUERIES)], images=crop, return_tensors='pt')
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if self.cancelled():
            raise InterruptedError('Head trial cancelled')
        return decode_heads(outputs.logits[0].numpy(), outputs.pred_boxes[0].numpy(), crop.size)
