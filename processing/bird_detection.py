"""鸟类检测后台任务：逐张提交、断点跳过、可停止，复用照片库互斥锁。"""

import argparse
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from models.bird_detector import (
    BirdDetector, DEFAULT_MIN_AREA_PERCENT, MODEL_ID, SCORE_VERSION, guard_frame_edges, score_detections,
)
from models.bird_focus import needs_focus_verification, verify_focus

logger = logging.getLogger(__name__)
TERMINAL = {'completed', 'cancelled', 'failed'}


@contextmanager
def connect(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def source_signature(path):
    stat = Path(path).stat()
    return f'{stat.st_size}:{stat.st_mtime_ns}'


def result_version(tiled, min_area_percent=DEFAULT_MIN_AREA_PERCENT):
    return f'{MODEL_ID}:{SCORE_VERSION}:tiled={int(tiled)}:min_area={min_area_percent:g}'


def reusable_details(row, signature, tiled):
    """复用同一原图、同一检测模式的第三版起定位与局部估计；新轮廓规则另行读取原图核验。"""
    if row['bird_source_signature'] != signature or row['bird_status'] not in ('scored', 'no_bird', 'unjudgeable'):
        return None
    version = row['bird_version'] or ''
    if not (version == f'{MODEL_ID}:bird-interior-edge-width-v3:tiled={int(tiled)}'
            or version.startswith(f'{MODEL_ID}:bird-interior-edge-width-v4:tiled={int(tiled)}:')
            or version.startswith(f'{MODEL_ID}:{SCORE_VERSION}:tiled={int(tiled)}:')):
        return None
    try:
        details = json.loads(row['bird_detection_json'] or '{}')
        if (isinstance(details, dict) and details.get('score_basis') == 'bird_interior' and details.get('width', 0) > 0
                and details.get('height', 0) > 0 and isinstance(details.get('birds'), list)):
            return details
    except (ValueError, TypeError):
        pass
    return None


def update_job(db_path, job_id, **fields):
    fields['updated_at'] = time.time()
    with connect(db_path) as conn:
        conn.execute(
            'UPDATE bird_detection_jobs SET ' + ', '.join(f'{key} = ?' for key in fields) + ' WHERE id = ?',
            [*fields.values(), job_id],
        )


def get_job(db_path, job_id):
    with connect(db_path) as conn:
        row = conn.execute('SELECT * FROM bird_detection_jobs WHERE id = ?', (job_id,)).fetchone()
        return dict(row) if row else None


def save_result(db_path, path, result, signature, version, error=None):
    with connect(db_path) as conn:
        cursor = conn.execute(
            'UPDATE photos SET bird_status=?, bird_count=?, bird_sharpness=?, bird_bbox=?, '
            'bird_detection_json=?, bird_source_signature=?, bird_version=?, bird_detected_at=?, bird_error=? '
            'WHERE path=?',
            (result['status'], result.get('count'), result.get('score'),
             json.dumps(result['bbox']) if result.get('bbox') else None,
             json.dumps(result), signature, version, time.time(), error, path),
        )
        return cursor.rowcount == 1


def run_job(db_path, job_id, detector_factory=BirdDetector, loader=None, resolver=None):
    """模型只在出现待处理原图时加载，统计仅在照片结果提交后递增。"""
    if loader is None:
        from utils.image_loading import load_image_from_path

        def loader(path):
            return load_image_from_path(path, use_thumbnail=False)[0]
    if resolver is None:
        from api.path_validation import resolve_photo_disk_path
        resolver = resolve_photo_disk_path
    job = get_job(db_path, job_id)
    if not job or job['status'] in TERMINAL:
        return
    paths = json.loads(job['paths_json'])
    options = json.loads(job['options_json'])
    min_area_percent = options.get('min_area_percent', DEFAULT_MIN_AREA_PERCENT)
    version = result_version(options['tiled'], min_area_percent)
    completed = succeeded = failed = skipped = 0
    detector = None
    update_job(db_path, job_id, status='running', phase='preparing', pid=os.getpid())

    def cancelled():
        current = get_job(db_path, job_id)
        return not current or bool(current['cancel_requested'])

    try:
        for path in paths:
            if cancelled():
                update_job(db_path, job_id, status='cancelled', phase='cancelled')
                return
            try:
                disk_path = resolver(path)
                signature = source_signature(disk_path)
                with connect(db_path) as conn:
                    row = conn.execute(
                        'SELECT bird_status, bird_version, bird_source_signature, bird_detection_json '
                        'FROM photos WHERE path=?', (path,),
                    ).fetchone()
                if row is None:
                    raise FileNotFoundError('Photo no longer exists')
                details = reusable_details(row, signature, options['tiled'])
                if (not options['recompute'] and row['bird_version'] == version
                        and row['bird_source_signature'] == signature
                        and row['bird_status'] in ('scored', 'no_bird', 'unjudgeable')
                        and details is not None and not needs_focus_verification(details, min_area_percent)):
                    skipped += 1
                elif not options['recompute'] and details is not None:
                    if needs_focus_verification(details, min_area_percent):
                        update_job(db_path, job_id, phase='verifying_focus')
                        image = loader(disk_path)
                        if image is None:
                            raise ValueError('Original image could not be decoded')
                        try:
                            details = verify_focus(image, details, min_area_percent)
                        finally:
                            image.close()
                    update_job(db_path, job_id, phase='selecting_birds')
                    result = guard_frame_edges(details, min_area_percent)
                    if source_signature(disk_path) != signature:
                        raise ValueError('Original image changed during selection')
                    if not save_result(db_path, path, result, signature, version):
                        raise FileNotFoundError('Photo removed during selection')
                    succeeded += 1
                else:
                    if detector is None:
                        update_job(db_path, job_id, phase='loading_model')
                        detector = detector_factory(tiled=options['tiled'], cancelled=cancelled)
                        # 加载失败属于整个任务失败，避免逐张把同一问题写成照片失败。
                        try:
                            detector.load()
                        except Exception:
                            logger.exception('Bird model could not be loaded')
                            update_job(db_path, job_id, status='failed', phase='failed',
                                       error='模型加载失败，请检查网络和模型运行环境后重试。')
                            return
                    update_job(db_path, job_id, phase='detecting')
                    image = loader(disk_path)
                    if image is None:
                        raise ValueError('Original image could not be decoded')
                    try:
                        result = score_detections(image, detector.detect(image), min_area_percent)
                        update_job(db_path, job_id, phase='verifying_focus')
                        result = guard_frame_edges(verify_focus(image, result, min_area_percent), min_area_percent)
                    finally:
                        image.close()
                    if source_signature(disk_path) != signature:
                        raise ValueError('Original image changed during detection')
                    if not save_result(db_path, path, result, signature, version):
                        raise FileNotFoundError('Photo removed during detection')
                    succeeded += 1
            except InterruptedError:
                update_job(db_path, job_id, status='cancelled', phase='cancelled')
                return
            except Exception:
                logger.exception('Bird detection failed for one photo')
                save_result(db_path, path, {'status': 'error'}, None, version,
                            error='无法读取原图或完成检测，请检查原图并重试。')
                failed += 1
            completed += 1
            update_job(db_path, job_id, completed=completed, succeeded=succeeded,
                       failed=failed, skipped=skipped)
        update_job(db_path, job_id, status='completed', phase='completed')
    except BaseException:
        logger.exception('Bird detection job stopped unexpectedly')
        update_job(db_path, job_id, status='failed', phase='failed', error='检测任务意外停止，可重新选择照片后继续。')
        raise


def main():
    parser = argparse.ArgumentParser(description='Selected-photo bird detection')
    parser.add_argument('--db', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    # 必须在导入服务器配置和互斥锁前指定同一个照片库及配置。
    os.environ['DB_PATH'] = args.db
    os.environ['FACET_CONFIG'] = args.config
    from facet import LibraryLock
    try:
        with LibraryLock(args.db, kind='bird detection'):
            run_job(args.db, args.job_id)
    except Exception:
        logger.exception('Bird detection worker failed')
        update_job(args.db, args.job_id, status='failed', phase='failed',
                   error='无法启动检测任务，可能有其他照片库任务正在运行。请稍后重试。')
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
