"""导出固定版本的候选模型；只在专用试验环境执行，不安装或升级任何依赖。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.bird_pose_trial import (  # noqa: E402
    POSE_MODEL_PATH, POSE_REPOSITORY, POSE_REVISION, POSE_WEIGHTS_SHA256,
)
from models.weights import download_weights  # noqa: E402

EXPORTER_VERSION = '8.4.202'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=POSE_MODEL_PATH)
    args = parser.parse_args()
    if importlib.metadata.version('ultralytics') != EXPORTER_VERSION:
        raise RuntimeError(f'The trial exporter requires ultralytics=={EXPORTER_VERSION}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ.setdefault('YOLO_CONFIG_DIR', str(args.output.parent / '.trial-export-config'))
    weights = args.output.parent / 'eye_pose_v0.pt'
    url = f'https://huggingface.co/{POSE_REPOSITORY}/resolve/{POSE_REVISION}/eye_pose_v0.pt'
    if not weights.exists():
        download_weights(url, weights, sha256=POSE_WEIGHTS_SHA256)
    if hashlib.sha256(weights.read_bytes()).hexdigest() != POSE_WEIGHTS_SHA256:
        raise ValueError('Trial weights do not match the pinned checksum')
    from ultralytics import YOLO

    exported = Path(YOLO(str(weights)).export(format='onnx', imgsz=640, opset=17, simplify=False,
                                             dynamic=True, nms=False, device='cpu'))
    if exported.resolve() != args.output.resolve():
        exported.replace(args.output)
    manifest = {
        'repository': POSE_REPOSITORY, 'revision': POSE_REVISION, 'weights_sha256': POSE_WEIGHTS_SHA256,
        'exported_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'versions': {name: importlib.metadata.version(name) for name in ('ultralytics', 'torch', 'onnx')},
        'export': {'opset': 17, 'dynamic': True, 'nms': False, 'simplify': False},
    }
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
