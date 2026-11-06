"""鸟类专用关键点定位；只读取模型数据，不执行外部模型代码。"""

from pathlib import Path

import numpy as np

from models.bird_detector import MODEL_DIR
from models.weights import download_weights

LANDMARK_MODEL_ID = 'cub200-resnet50-939fbcc'
LANDMARK_FILENAME = 'cub200_keypoint_resnet50.onnx'
LANDMARK_SHA256 = '2b62d30e36357b807a909fb22ae55323f768feb2957fbf4ecbd840bd7525f514'
LANDMARK_URL = (
    'https://huggingface.co/jamesphotography/SuperPicky-models/resolve/'
    '939fbccdb0a9af538b218a77e19db3e01c05f399/' + LANDMARK_FILENAME
)


class BirdLandmarkLoadError(RuntimeError):
    """头眼模型无法加载；应停止整个任务，而非逐张覆写成照片错误。"""


class BirdLandmarkDetector:
    """使用左右眼和喙的坐标及可见性概率；概率本身不代表清晰度。"""

    def __init__(self, model_dir=MODEL_DIR, cancelled=lambda: False):
        self.path = Path(model_dir) / LANDMARK_FILENAME
        self.cancelled = cancelled
        self.session = None

    def load(self):
        if self.session is not None:
            return
        if self.cancelled():
            raise InterruptedError('Bird detection cancelled')
        try:
            if not self.path.exists():
                download_weights(LANDMARK_URL, self.path, sha256=LANDMARK_SHA256)
            if self.cancelled():
                raise InterruptedError('Bird detection cancelled')
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.path), sess_options=options, providers=['CPUExecutionProvider'],
            )
        except InterruptedError:
            raise
        except Exception as exc:
            raise BirdLandmarkLoadError('Bird landmark model could not be loaded') from exc

    def detect(self, crop):
        """原图与水平翻转各定位一次；将翻转结果的左右眼映射回原图。"""
        return self._detect(crop)

    def refine(self, crop, outline):
        """仅为定位抑制鸟体外的干扰；清晰度测量始终读取未改动的原图。"""
        return self._detect(crop, outline)

    def inspect_eye(self, crop):
        """在候选眼睛的局部上下文中重新定位，以核对暗斑是否对应眼睛。"""
        return self._detect(crop)

    def _detect(self, crop, outline=None):
        from PIL import Image, ImageDraw

        self.load()
        with crop.convert('RGB') as rgb, rgb.resize((416, 416), Image.Resampling.BILINEAR) as resized:
            pixels = np.asarray(resized).copy()
        if outline is not None:
            with Image.new('L', (416, 416)) as mask:
                draw = ImageDraw.Draw(mask)
                for polygon in outline:
                    if len(polygon) >= 3:
                        draw.polygon([(x * 416, y * 416) for x, y in polygon], fill=255)
                pixels[np.asarray(mask) == 0] = 127
        pixels = pixels.astype(np.float32) / 255
        pixels = (pixels - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray(
            [.229, .224, .225], dtype=np.float32,
        )
        predictions = []
        for observed in (pixels, pixels[:, ::-1]):
            if self.cancelled():
                raise InterruptedError('Bird detection cancelled')
            coords, visibility = self.session.run(
                ['coords', 'vis'], {'input': np.ascontiguousarray(observed.transpose(2, 0, 1)[None])},
            )
            predictions.append((coords[0], visibility[0]))
        coordinates, visibility = predictions[0]
        mirrored, mirrored_visibility = predictions[1]
        mirrored = mirrored[[1, 0, 2]].copy()
        mirrored[:, 0] = 1 - mirrored[:, 0]
        mirrored_visibility = mirrored_visibility[[1, 0, 2]]
        # 保存两次观察，调用方用真实像素距离核验定位稳定性。
        return {
            'points': ((coordinates + mirrored) / 2).tolist(),
            'visibility': np.minimum(visibility, mirrored_visibility).tolist(),
            'views': [coordinates.tolist(), mirrored.tolist()],
            'view_visibility': [visibility.tolist(), mirrored_visibility.tolist()],
            'model': LANDMARK_MODEL_ID,
        }
