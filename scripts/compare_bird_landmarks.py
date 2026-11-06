"""在同一批原图与鸟体裁剪上比较头眼模型；不写照片库，不切换正式模型。"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import ImageDraw  # noqa: E402

from models.bird_focus import FOCUS_CHECK_VERSION, verify_focus  # noqa: E402
from models.bird_landmarks import BirdLandmarkDetector, LANDMARK_MODEL_ID  # noqa: E402
from models.bird_pose_trial import BirdPoseTrialDetector, POSE_MODEL_PATH, POSE_REVISION  # noqa: E402
from processing.bird_detection import source_signature  # noqa: E402
from utils.image_loading import load_image_from_path  # noqa: E402

ANCHORS = ('DSC02049.JPG', 'DSC00935.JPG', 'DSC02086.JPG')


def ensure_settings(output, settings):
    """断点继续时，禁止将不同模型或预处理的输出混成同一组结果。"""
    path = output / 'settings.json'
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError('Comparison settings changed; use a different output directory')
    elif (output / 'results.jsonl').exists():
        raise ValueError('Existing comparison has no provenance; use a different output directory')
    else:
        path.write_text(json.dumps(settings, indent=2) + '\n')


def selected_bird(details):
    """两种模型始终测试同一只鸟，不允许通过改选相邻鸟获得更高分。"""
    if details.get('best_index') is not None:
        return details['best_index']
    eligible = []
    for index, bird in enumerate(details.get('birds', [])):
        box = bird['bbox']
        area = (box[2] - box[0]) * (box[3] - box[1]) * 100
        if (not bird.get('frame_edge') and bird.get('reason') not in ('frame_edge', 'truncated')
                and bird.get('mask_outline') and area >= details.get('min_area_percent', 1)):
            eligible.append((area, index))
    return max(eligible)[1] if eligible else None


def sample_photos(db_path, count, seed=42):
    with sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True) as conn:
        rows = conn.execute('SELECT path,filename,bird_sharpness,bird_detection_json,bird_source_signature '
                            'FROM photos WHERE bird_detection_json IS NOT NULL ORDER BY path').fetchall()
    groups = {'high': [], 'low': [], 'unjudgeable': [], 'other': []}
    anchors = {}
    for path, filename, score, raw, signature in rows:
        details = json.loads(raw)
        index = selected_bird(details)
        if index is None:
            continue
        record = {'path': path, 'filename': filename, 'stored_score': score, 'details': details,
                  'bird_index': index, 'source_signature': signature}
        if filename in ANCHORS:
            anchors[filename] = record
        else:
            group = 'unjudgeable' if score is None else ('high' if score >= 7 else ('low' if score <= 3 else 'other'))
            groups[group].append(record)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = [anchors[name] for name in ANCHORS if name in anchors][:count]
    while len(selected) < count and any(groups.values()):
        for values in groups.values():
            if values and len(selected) < count:
                selected.append(values.pop())
    return selected


class CachedInitialDetector:
    """复用已计时的第一次定位，避免在同一张裁剪上重复运行相同推理。"""
    def __init__(self, detector, initial):
        self.detector, self.initial = detector, initial

    def detect(self, crop):
        return copy.deepcopy(self.initial)

    def refine(self, crop, outline):
        return self.detector.refine(crop, outline)

    def inspect_eye(self, crop):
        return self.detector.inspect_eye(crop)


def preview(image, rect, raw, measured, destination):
    """橙色表示模型原始眼点，紫色表示评分流程最终采用的眼周区域。"""
    with image.crop(rect) as source:
        source.thumbnail((1000, 800))
        draw = ImageDraw.Draw(source)
        sx, sy = source.width / (rect[2] - rect[0]), source.height / (rect[3] - rect[1])
        for point, confidence in zip(raw['points'][:2], raw['visibility'][:2]):
            if confidence < .65:
                continue
            x, y = point[0] * source.width, point[1] * source.height
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline='#f97316', width=3)
        for region in measured.get('eyes', []):
            x = (region['centre'][0] * image.width - rect[0]) * sx
            y = (region['centre'][1] * image.height - rect[1]) * sy
            rx, ry = region['eye_radius'] * sx, region['eye_radius'] * sy
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline='#db2777', width=2)
        source.save(destination)


def run(db_path, output, count=24, sizes=(640, 960, 1280), model_path=POSE_MODEL_PATH):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ensure_settings(output, {
        'protocol': 'bird-pose-official-letterbox-v1', 'focus_version': FOCUS_CHECK_VERSION,
        'current_model': LANDMARK_MODEL_ID, 'pose_revision': POSE_REVISION,
        'exported_sha256': hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        'input_sizes': list(sizes), 'requested_count': count,
    })
    manifest_path = output / 'sample.json'
    if manifest_path.exists():
        samples = json.loads(manifest_path.read_text())
    else:
        samples = sample_photos(db_path, count)
        manifest_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
    detectors = {'current': BirdLandmarkDetector()}
    detectors.update({f'pose_{size}': BirdPoseTrialDetector(model_path, size) for size in sizes})
    for detector in detectors.values():
        detector.load()
    results_path = output / 'results.jsonl'
    completed = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            result = json.loads(line)
            completed[(result['path'], result['model'])] = result
    for number, sample in enumerate(samples, 1):
        if sample['source_signature'] != source_signature(sample['path']):
            raise ValueError('Photo changed since sampling: ' + sample['filename'])
        keys = [(sample['path'], name) for name in detectors]
        if all(key in completed for key in keys):
            continue
        image, _ = load_image_from_path(sample['path'], use_thumbnail=False)
        if image is None:
            raise ValueError('Original image could not be decoded: ' + sample['filename'])
        try:
            bird = copy.deepcopy(sample['details']['birds'][sample['bird_index']])
            box = bird['bbox']
            rect = (math.floor(box[0] * image.width), math.floor(box[1] * image.height),
                    math.ceil(box[2] * image.width), math.ceil(box[3] * image.height))
            with image.crop(rect) as crop:
                for name, detector in detectors.items():
                    key = (sample['path'], name)
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    raw = detector.detect(crop)
                    raw_seconds = time.perf_counter() - started
                    details = {'width': image.width, 'height': image.height, 'birds': [copy.deepcopy(bird)]}
                    details['birds'][0].pop('focus_check_version', None)
                    started = time.perf_counter()
                    result = verify_focus(image, details, sample['details']['min_area_percent'],
                                          detector=CachedInitialDetector(detector, raw))
                    seconds = raw_seconds + time.perf_counter() - started
                    measured = result['birds'][0]['focus_evidence']
                    result_path = output / f'{number:03d}-{name}.jpg'
                    preview(image, rect, raw, measured, result_path)
                    entry = {'path': sample['path'], 'filename': sample['filename'], 'model': name,
                             'model_id': raw['model'], 'bird_index': sample['bird_index'],
                             'score': result['score'], 'reason': result['birds'][0]['reason'],
                             'raw_seconds': raw_seconds, 'total_seconds': seconds,
                             'raw': raw, 'focus_evidence': measured, 'preview': result_path.name}
                    with results_path.open('a') as handle:
                        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    completed[key] = entry
            print(f'{number}/{len(samples)} {sample["filename"]}', flush=True)
        finally:
            image.close()
        if sample['source_signature'] != source_signature(sample['path']):
            raise ValueError('Photo changed during comparison: ' + sample['filename'])
    summary = {}
    for name in detectors:
        entries = [completed[(sample['path'], name)] for sample in samples]
        summary[name] = {'count': len(entries), 'scored': sum(item['score'] is not None for item in entries),
                         'reasons': dict(Counter(item['reason'] for item in entries if item['reason'])),
                         'mean_seconds': float(np.mean([item['total_seconds'] for item in entries]))}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 鸟类头眼模型对照', '',
             '同一批原图、同一只鸟及同一套评分校验。橙圈是模型原始眼点，紫圈是最终眼周测量区。', '',
             '**“得到分数”的数量不是准确率；未逐张人工标注的样本不能据此判断哪种模型更准确。**', '',
             '| 模型 | 样本数 | 得到分数 | 平均每张秒数 |', '|---|---:|---:|---:|']
    labels = {'current': '当前方案', **{f'pose_{size}': f'候选模型 {size}' for size in sizes}}
    reasons = {None: '通过当前校验', 'head_unlocalized': '头眼位置未可靠定位',
               'eye_unverified': '眼部结构核验未通过', 'eye_detail_insufficient': '眼周或头部细节不足'}
    for name, values in summary.items():
        lines.append(f'| {labels[name]} | {values["count"]} | {values["scored"]} | {values["mean_seconds"]:.3f} |')
    for sample in samples:
        lines.extend(['', '## ' + sample['filename'], ''])
        for name in detectors:
            item = completed[(sample['path'], name)]
            score = str(item['score']) if item['score'] is not None else '未评分'
            reason = reasons.get(item['reason'], item['reason'])
            lines.extend([f'**{labels[name]}**：{score}；{reason}。', '',
                          f'![{labels[name]}]({(output / item["preview"]).resolve()})', ''])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('photo_scores_pro.db'))
    parser.add_argument('--output', type=Path, default=Path('.facet_cache/birds/model-trial'))
    parser.add_argument('--sample', type=int, default=24)
    parser.add_argument('--sizes', type=int, nargs='+', default=[640, 960, 1280])
    parser.add_argument('--model-path', type=Path, default=POSE_MODEL_PATH)
    args = parser.parse_args()
    if args.sample < 1:
        parser.error('--sample must be positive')
    print(json.dumps(run(args.db, args.output.resolve(), args.sample, args.sizes, args.model_path), indent=2))


if __name__ == '__main__':
    main()
