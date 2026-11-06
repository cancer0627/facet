"""鸟类专用姿态模型的试验适配器；不改变正式评分使用的模型。"""

from pathlib import Path

import numpy as np

from models.bird_detector import MODEL_DIR
from models.bird_landmarks import BirdLandmarkLoadError

POSE_REPOSITORY = 'synthet/eye-pose-v0'
POSE_REVISION = '3025db2b961e2e0c611bcb7d13f179a5d5d085f2'
POSE_WEIGHTS_SHA256 = 'ec9e4b80a93eb6e2b21633e2ff2df30c49faa1eb7919dd24345c7dc295b2d1be'
POSE_MODEL_PATH = MODEL_DIR / 'eye_pose_v0_dynamic.onnx'


class BirdPoseTrialDetector:
    """将鸟类六关键点输出转换为现有头眼评分协议，供同图对照。"""

    def __init__(self, model_path=POSE_MODEL_PATH, input_size=640, cancelled=lambda: False):
        if input_size < 320 or input_size > 1280 or input_size % 32:
            raise ValueError('Input size must be a multiple of 32 between 320 and 1280')
        self.path = Path(model_path)
        self.input_size = input_size
        self.cancelled = cancelled
        self.session = None
        self.model_id = f'synthet-eye-pose-v0-{POSE_REVISION[:7]}-size{input_size}'

    def load(self):
        if self.session is not None:
            return
        if self.cancelled():
            raise InterruptedError('Bird detection cancelled')
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(self.path), sess_options=options,
                                               providers=['CPUExecutionProvider'])
        except Exception as exc:
            raise BirdLandmarkLoadError('Trial pose model could not be loaded; export it first') from exc

    def _predict(self, crop, outline=None, mirrored=False):
        import cv2
        from PIL import Image, ImageDraw

        if self.cancelled():
            raise InterruptedError('Bird detection cancelled')
        self.load()
        size = self.input_size
        scale = size / max(crop.size)
        width, height = round(crop.width * scale), round(crop.height * scale)
        left, top = round((size - width) / 2 - .1), round((size - height) / 2 - .1)
        # 与发布模型的官方推理预处理一致，不能用另一种缩图滤波改变小眼睛的细节。
        with crop.convert('RGB') as rgb:
            pixels = cv2.resize(np.asarray(rgb), (width, height), interpolation=cv2.INTER_LINEAR)
        coordinate_size = np.asarray([width, height])
        if outline is not None:
            with Image.new('L', (width, height)) as mask:
                draw = ImageDraw.Draw(mask)
                for polygon in outline:
                    if len(polygon) >= 3:
                        draw.polygon([(x * width, y * height) for x, y in polygon], fill=255)
                pixels[np.asarray(mask) == 0] = 127
        if mirrored:
            pixels = pixels[:, ::-1]
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[top:top + height, left:left + width] = pixels
        inputs = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = self.session.run(None, {self.session.get_inputs()[0].name: inputs})[0]
        if output.ndim != 3 or output.shape[:2] != (1, 23):
            raise ValueError('Unexpected bird pose output shape')
        predictions = output[0].T
        predictions = predictions[np.isfinite(predictions).all(axis=1) & (predictions[:, 4] >= .25)]
        empty = (np.full((3, 2), .5), np.zeros(3), {'detected': False})
        if not len(predictions):
            return empty
        centres, dimensions = predictions[:, :2], predictions[:, 2:4]
        xywh = np.column_stack((centres - dimensions / 2, dimensions))
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), predictions[:, 4].tolist(), .25, .45)
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if not len(indices):
            return empty
        boxes = xywh[indices].copy()
        boxes[:, :2] = (boxes[:, :2] - [left, top]) / coordinate_size
        boxes[:, 2:] /= coordinate_size
        ends = np.clip(boxes[:, :2] + boxes[:, 2:], 0, 1)
        starts = np.clip(boxes[:, :2], 0, 1)
        areas = np.prod(np.maximum(0, ends - starts), axis=1)
        # 输入是既有鸟体裁剪，优先覆盖该裁剪的主体，避免改选为相邻的小鸟。
        chosen = int(np.argmax(areas))
        prediction = predictions[indices[chosen]]
        keypoints = prediction[5:].reshape(6, 3)
        points = (keypoints[:, :2] - [left, top]) / coordinate_size
        visibility = keypoints[:, 2]
        # 发布模型顺序为喙、左眼、右眼、头顶、左右肩；转换为左眼、右眼、喙。
        order = [1, 2, 0]
        points, visibility = points[order], visibility[order]
        if mirrored:
            points = points[[1, 0, 2]].copy()
            points[:, 0] = 1 - points[:, 0]
            visibility = visibility[[1, 0, 2]]
        return points, visibility, {'detected': True, 'box_confidence': float(prediction[4]),
                                     'bbox': [*starts[chosen].tolist(), *ends[chosen].tolist()],
                                     'candidate_count': len(indices)}

    def _detect(self, crop, outline=None):
        first, first_visibility, first_detection = self._predict(crop, outline)
        second, second_visibility, second_detection = self._predict(crop, outline, mirrored=True)
        return {'points': ((first + second) / 2).tolist(),
                'visibility': np.minimum(first_visibility, second_visibility).tolist(),
                'views': [first.tolist(), second.tolist()],
                'view_visibility': [first_visibility.tolist(), second_visibility.tolist()],
                'detections': [first_detection, second_detection], 'model': self.model_id}

    def detect(self, crop):
        return self._detect(crop)

    def refine(self, crop, outline):
        return self._detect(crop, outline)

    def inspect_eye(self, crop):
        return self._detect(crop)
