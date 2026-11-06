"""大象眼部定位：哺乳动物关键点模型，原图与镜像分别核验。"""

from pathlib import Path
import tempfile
import zipfile

import numpy as np

from models.weights import download_weights, _sha256_file

MODEL_DIR = Path(__file__).resolve().parents[1] / 'pretrained_models' / 'elephants'
LANDMARK_MODEL_ID = 'rtmpose-m-ap10k-7a041aa1'
LANDMARK_SHA256 = '1cfd1c86e0d9e5d5f95178bcd95ee9a4e8386a624cd3c57519f27ff58cac7f28'
LANDMARK_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
    'rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip'
)


class ElephantLandmarkLoadError(RuntimeError):
    """定位模型加载失败，应停止任务并保留已有照片结果。"""


class ElephantLandmarkDetector:
    def __init__(self, model_dir=MODEL_DIR, cancelled=lambda: False):
        self.path = Path(model_dir) / 'rtmpose-m-ap10k.onnx'
        self.cancelled = cancelled
        self.session = None

    def load(self):
        if self.session is not None:
            return
        if self.cancelled():
            raise InterruptedError('Elephant detection cancelled')
        try:
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=self.path.parent) as folder:
                    archive = Path(folder) / 'pose.zip'
                    download_weights(LANDMARK_URL, archive)
                    with zipfile.ZipFile(archive) as bundle:
                        names = [name for name in bundle.namelist() if name.endswith('/end2end.onnx')]
                        if len(names) != 1:
                            raise ValueError('Unexpected model archive')
                        model = Path(folder) / 'pose.onnx'
                        model.write_bytes(bundle.read(names[0]))
                    if _sha256_file(model) != LANDMARK_SHA256:
                        raise ValueError('Elephant landmark model checksum mismatch')
                    model.replace(self.path)
            if _sha256_file(self.path) != LANDMARK_SHA256:
                raise ValueError('Elephant landmark model checksum mismatch')
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(self.path), sess_options=options,
                                               providers=['CPUExecutionProvider'])
        except InterruptedError:
            raise
        except Exception as exc:
            raise ElephantLandmarkLoadError('Elephant landmark model could not be loaded') from exc

    def detect(self, crop):
        import cv2

        self.load()
        # 官方模型使用等比例正方形上下文和 1.25 倍留白；不能直接拉伸目标。
        side = max(crop.size) * 1.25
        centre = np.asarray(crop.size, dtype=np.float32) / 2
        factor = 256 / side
        affine = np.asarray([[factor, 0, 128 - centre[0] * factor],
                             [0, factor, 128 - centre[1] * factor]], dtype=np.float32)
        with crop.convert('RGB') as rgb:
            pixels = cv2.warpAffine(np.asarray(rgb), affine, (256, 256))
        predictions = []
        for observed in (pixels, pixels[:, ::-1]):
            if self.cancelled():
                raise InterruptedError('Elephant detection cancelled')
            tensor = (observed.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
            x, y = self.session.run(None, {'input': np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)})
            # 模型输出分轴响应；峰值是定位置信度，不是清晰度分数。
            points = np.stack([x[0].argmax(axis=-1), y[0].argmax(axis=-1)], axis=-1).astype(float) / 2
            confidence = np.minimum(x[0].max(axis=-1), y[0].max(axis=-1))
            predictions.append((points, confidence))
        points, confidence = predictions[0]
        mirror, mirror_confidence = predictions[1]
        swap = [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]
        mirror, mirror_confidence = mirror[swap].copy(), mirror_confidence[swap]
        mirror[:, 0] = 255 - mirror[:, 0]
        views = [(observed - 128) / factor + centre for observed in (points, mirror)]
        size = np.asarray(crop.size)
        return {
            'points': ((views[0] + views[1]) / 2 / size).tolist(),
            'visibility': np.minimum(confidence, mirror_confidence).tolist(),
            'views': [(view / size).tolist() for view in views],
            'view_visibility': [confidence.tolist(), mirror_confidence.tolist()],
            'model': LANDMARK_MODEL_ID,
        }
