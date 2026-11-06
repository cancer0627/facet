"""复用官方多类别目标模型，明确仅提取大象；不改变照片分类。"""

from models.bird_detector import BirdDetector, MODEL_ID as BODY_MODEL_ID

DEFAULT_MIN_AREA_PERCENT = 1.0
MODEL_ID = f'{BODY_MODEL_ID}-elephant'
SCORE_VERSION = 'elephant-head-eye-v1'
RESULT_VERSION_PREFIX = f'{MODEL_ID}:{SCORE_VERSION}:tiled='


class ElephantDetector(BirdDetector):
    species = 'elephant'
