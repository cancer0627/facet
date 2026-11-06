"""用独立头部范围定位，再在原图头部裁剪定位眼睛；只输出试验记录。"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import ImageDraw  # noqa: E402

from models.bird_focus import FOCUS_CHECK_VERSION, head_eye_evidence  # noqa: E402
from models.bird_head_trial import (  # noqa: E402
    HEAD_MODEL_PATH, HEAD_QUERIES, HEAD_REPOSITORY, HEAD_REVISION, HEAD_WEIGHTS_SHA256,
    IndependentBirdHeadTrial, map_head_landmarks,
)
from models.bird_landmarks import BirdLandmarkDetector, LANDMARK_MODEL_ID  # noqa: E402
from models.bird_pose_trial import BirdPoseTrialDetector, POSE_MODEL_PATH, POSE_REVISION  # noqa: E402
from processing.bird_detection import source_signature  # noqa: E402
from scripts.compare_bird_landmarks import ensure_settings  # noqa: E402
from utils.image_loading import load_image_from_path  # noqa: E402


class HeadObservation:
    """固定独立头框中的观察，不回退到整鸟定位，也不按清晰度选择头部。"""

    def __init__(self, detector, landmarks):
        self.detector, self.landmarks = detector, landmarks

    def detect(self, crop):
        return copy.deepcopy(self.landmarks)

    def inspect_eye(self, crop):
        return self.detector.inspect_eye(crop)


def expanded_crop(box, size):
    left, top, right, bottom = box
    dx, dy = .2 * (right - left), .2 * (bottom - top)
    return (max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
            min(size[0], math.ceil(right + dx)), min(size[1], math.ceil(bottom + dy)))


def constrain_eye_to_head(evidence, head_box, bird_origin, image_size):
    """眼部结构搜索不能跳出独立头框，改测相邻鸟或其他身体区域。"""
    for eye in evidence.get('eyes', []):
        x = eye['centre'][0] * image_size[0] - bird_origin[0]
        y = eye['centre'][1] * image_size[1] - bird_origin[1]
        if not head_box[0] <= x <= head_box[2] or not head_box[1] <= y <= head_box[3]:
            evidence['rejected_eye_regions'] = evidence.pop('eyes')
            evidence.pop('score', None)
            evidence.update(reason='eye_outside_independent_head', eyes=[])
            break
    return evidence


def write_preview(image, rect, heads, destination):
    with image.crop(rect) as crop:
        crop.thumbnail((1100, 850))
        sx, sy = crop.width / (rect[2] - rect[0]), crop.height / (rect[3] - rect[1])
        draw = ImageDraw.Draw(crop)
        for number, head in enumerate(heads, 1):
            box = head['bbox']
            draw.rectangle((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy), outline='#00aaff', width=3)
            draw.text((box[0] * sx, box[1] * sy), str(number), fill='#ff2222', stroke_width=1)
            for variant in head['variants']:
                color = '#00ff66' if variant['eye_model'] == 'current' else '#ff00cc'
                raw = variant['landmarks']
                for p, v in zip(raw['points'][:2], raw['visibility'][:2]):
                    if v >= .65:
                        x, y = p[0] * crop.width, p[1] * crop.height
                        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
                for eye in variant['focus_evidence'].get('eyes', []):
                    x = (eye['centre'][0] * image.width - rect[0]) * sx
                    y = (eye['centre'][1] * image.height - rect[1]) * sy
                    draw.line((x - 7, y, x + 7, y), fill=color, width=3)
                    draw.line((x, y - 7, x, y + 7), fill=color, width=3)
        crop.save(destination, quality=95)


def run(manifest, output, limit=24):
    output.mkdir(parents=True, exist_ok=True)
    weights_hash = hashlib.sha256((HEAD_MODEL_PATH / 'model.safetensors').read_bytes()).hexdigest()
    if weights_hash != HEAD_WEIGHTS_SHA256:
        raise ValueError('Independent head model checksum mismatch')
    ensure_settings(output, {'protocol': 'independent-head-label-competition-v2',
                            'head_repository': HEAD_REPOSITORY, 'head_revision': HEAD_REVISION,
                            'weights_sha256': weights_hash, 'queries': list(HEAD_QUERIES),
                            'eye_model': LANDMARK_MODEL_ID, 'pose_revision': POSE_REVISION,
                            'pose_sha256': hashlib.sha256(POSE_MODEL_PATH.read_bytes()).hexdigest(),
                            'processor_config_sha256': hashlib.sha256(
                                (HEAD_MODEL_PATH / 'preprocessor_config.json').read_bytes()).hexdigest(),
                            'focus_version': FOCUS_CHECK_VERSION,
                            'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            'head_threshold': .1, 'max_bird_area_fraction': .35,
                            'head_crop_padding': .2, 'limit': limit})
    samples = json.loads(manifest.read_text())
    # 优先运行三张问题图和两张漏检；其余照片仍使用原来的固定目标。
    priority = ['DSC02049.JPG', 'DSC00935.JPG', 'DSC02086.JPG', 'DSC01654.JPG', 'DSC02136.JPG']
    samples.sort(key=lambda item: priority.index(item['filename']) if item['filename'] in priority else len(priority))
    samples = samples[:limit]
    if not samples:
        raise ValueError('No photos in the source manifest')
    (output / 'sample.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
    records_path = output / 'results.jsonl'
    completed = {}
    if records_path.exists():
        completed = {r['path']: r for r in map(json.loads, records_path.read_text().splitlines())}
    head_detector = IndependentBirdHeadTrial()
    eye_detectors = {'current': BirdLandmarkDetector(), 'pose640': BirdPoseTrialDetector()}
    if any(s['path'] not in completed for s in samples):
        head_detector.load()
        for detector in eye_detectors.values():
            detector.load()
    for number, sample in enumerate(samples, 1):
        if source_signature(sample['path']) != sample['source_signature']:
            raise ValueError('Photo changed since sampling: ' + sample['filename'])
        if sample['path'] in completed:
            continue
        image, _ = load_image_from_path(sample['path'], use_thumbnail=False)
        if image is None:
            raise ValueError('Could not decode original: ' + sample['filename'])
        try:
            bird = sample['details']['birds'][sample['bird_index']]
            b = bird['bbox']
            rect = (math.floor(b[0] * image.width), math.floor(b[1] * image.height),
                    math.ceil(b[2] * image.width), math.ceil(b[3] * image.height))
            started = time.perf_counter()
            with image.crop(rect) as crop:
                independent = head_detector.detect(crop)
                heads = []
                for index, proposal in enumerate(independent['heads'], 1):
                    local_rect = expanded_crop(proposal['bbox'], crop.size)
                    with crop.crop(local_rect) as head_crop:
                        head_crop.save(output / f'{number:03d}-head-{index}-original.jpg', quality=97)
                        variants = []
                        for name, eye_detector in eye_detectors.items():
                            raw = eye_detector.detect(head_crop)
                            mapped = map_head_landmarks(raw, local_rect, crop.size)
                            evidence = head_eye_evidence(image, bird, HeadObservation(eye_detector, mapped))
                            constrain_eye_to_head(evidence, proposal['bbox'], rect[:2], image.size)
                            variants.append({'eye_model': name, 'landmarks': mapped, 'focus_evidence': evidence})
                    heads.append({**proposal, 'crop_rect': local_rect, 'variants': variants})
                preview = output / f'{number:03d}-heads.jpg'
                write_preview(image, rect, heads, preview)
                record = {'filename': sample['filename'], 'path': sample['path'],
                          'bird_index': sample['bird_index'], 'bird_rect': rect,
                          'total_seconds': time.perf_counter() - started,
                          'head_detection': independent, 'heads': heads, 'preview': preview.name}
                with records_path.open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                completed[sample['path']] = record
                print(f'{number}/{len(samples)} {sample["filename"]}: heads={len(heads)}, '
                      f'scores={[[v["focus_evidence"].get("score") for v in h["variants"]] for h in heads]}',
                      flush=True)
        finally:
            image.close()
        if source_signature(sample['path']) != sample['source_signature']:
            raise ValueError('Photo changed during trial: ' + sample['filename'])
    records = [completed[s['path']] for s in samples]
    summary = {'photos': len(records), 'photos_with_head_proposal': sum(bool(r['heads']) for r in records),
               'photos_with_score': {
                   name: sum(any(v['eye_model'] == name and v['focus_evidence'].get('score') is not None
                                 for h in r['heads'] for v in h['variants']) for r in records)
                   for name in eye_detectors},
               'mean_seconds': sum(r['total_seconds'] for r in records) / len(records)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# 海鸥独立头部定位试验', '',
             '蓝框为独立模型定位的头部范围；绿色为当前眼部模型，紫色为候选眼部模型。圆圈是原始眼点，十字为最终眼位。', '',
             '每个头部候选分别保留结果，不按清晰度选择目标，不向照片库写分。出现头框或分数均不等于识别正确，需目视复核。', '',
             '头框面积、置信度和补边参数是未经准确率校准的试验门槛；未重新验证整图鸟体检出与重叠分离。', '']
    for r in records:
        lines.extend(['## ' + r['filename'], ''])
        if not r['heads']:
            lines.extend(['未保留头部候选。', ''])
        for i, head in enumerate(r['heads'], 1):
            for variant in head['variants']:
                e = variant['focus_evidence']
                score = e.get('score', '未评分')
                reason = e.get('reason') or '通过现有校验，尚需检查眼位'
                label = '当前眼部模型' if variant['eye_model'] == 'current' else '候选眼部模型'
                lines.extend([f'头部候选 {i}，{label}：参考分 {score}；{reason}。', ''])
        lines.extend([f'![头眼定位对照]({output / r["preview"]})', ''])
    (output / 'report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('.facet_cache/birds/model-trial-official/sample.json'))
    parser.add_argument('--output', type=Path, default=Path('.facet_cache/birds/head-trial'))
    parser.add_argument('--limit', type=int, default=24)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    print(json.dumps(run(args.manifest.resolve(), args.output.resolve(), args.limit), ensure_ascii=False))
