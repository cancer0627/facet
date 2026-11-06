"""手选大象检测任务：独立结果、逐张持久化、源文件核验和照片库互斥锁。"""

import argparse
import json
import logging
import os
import time

from models.elephant_detector import ElephantDetector, MODEL_ID, SCORE_VERSION, DEFAULT_MIN_AREA_PERCENT
from models.elephant_focus import score_elephants
from models.elephant_landmarks import ElephantLandmarkDetector, ElephantLandmarkLoadError
from processing.bird_detection import connect, source_signature, TERMINAL

logger = logging.getLogger(__name__)


def result_version(tiled, min_area_percent=DEFAULT_MIN_AREA_PERCENT):
    return f'{MODEL_ID}:{SCORE_VERSION}:tiled={int(tiled)}:min_area={min_area_percent:g}'


def update_job(db_path, job_id, **fields):
    fields['updated_at'] = time.time()
    with connect(db_path) as conn:
        conn.execute('UPDATE elephant_detection_jobs SET ' + ', '.join(f'{key}=?' for key in fields) + ' WHERE id=?',
                     [*fields.values(), job_id])


def get_job(db_path, job_id):
    with connect(db_path) as conn:
        row = conn.execute('SELECT * FROM elephant_detection_jobs WHERE id=?', (job_id,)).fetchone()
        return dict(row) if row else None


def save_result(db_path, path, result, signature, version, error=None):
    with connect(db_path) as conn:
        cursor = conn.execute(
            'UPDATE photos SET elephant_status=?, elephant_count=?, elephant_sharpness=?, elephant_bbox=?, '
            'elephant_detection_json=?, elephant_source_signature=?, elephant_version=?, elephant_detected_at=?, elephant_error=? '
            'WHERE path=?',
            (result['status'], result.get('count'), result.get('score'),
             json.dumps(result['bbox']) if result.get('bbox') else None,
             json.dumps(result), signature, version, time.time(), error, path),
        )
        return cursor.rowcount == 1


def run_job(db_path, job_id, detector_factory=ElephantDetector, loader=None, resolver=None):
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
    paths, options = json.loads(job['paths_json']), json.loads(job['options_json'])
    version = result_version(options['tiled'], options['min_area_percent'])
    update_job(db_path, job_id, status='running', phase='preparing', pid=os.getpid())
    completed = succeeded = skipped = failed = 0

    def cancelled():
        current = get_job(db_path, job_id)
        return not current or bool(current['cancel_requested'])

    detector = None
    focus = ElephantLandmarkDetector(cancelled=cancelled)
    try:
        for path in paths:
            if cancelled():
                update_job(db_path, job_id, status='cancelled', phase='cancelled')
                return
            try:
                disk_path = resolver(path)
                signature = source_signature(disk_path)
                with connect(db_path) as conn:
                    row = conn.execute('SELECT elephant_status,elephant_version,elephant_source_signature '
                                       'FROM photos WHERE path=?', (path,)).fetchone()
                if row is None:
                    raise FileNotFoundError('Photo no longer exists')
                if (not options['recompute'] and row['elephant_version'] == version
                        and row['elephant_source_signature'] == signature
                        and row['elephant_status'] in ('scored', 'no_elephant', 'unjudgeable')):
                    skipped += 1
                else:
                    if detector is None:
                        update_job(db_path, job_id, phase='loading_model')
                        detector = detector_factory(tiled=options['tiled'], cancelled=cancelled)
                        try:
                            detector.load()
                        except InterruptedError:
                            raise
                        except Exception:
                            logger.exception('Elephant detection model could not be loaded')
                            update_job(db_path, job_id, status='failed', phase='failed',
                                       error='大象检测模型加载失败，请检查网络和模型运行环境后重试。')
                            return
                    image = loader(disk_path)
                    if image is None:
                        raise ValueError('Original image could not be decoded')
                    try:
                        update_job(db_path, job_id, phase='detecting')
                        detections = detector.detect(image)
                        update_job(db_path, job_id, phase='verifying_focus')
                        result = score_elephants(image, detections, options['min_area_percent'], detector=focus)
                    finally:
                        image.close()
                    if signature != source_signature(disk_path):
                        raise ValueError('Original image changed during detection')
                    if not save_result(db_path, path, result, signature, version):
                        raise FileNotFoundError('Photo removed during detection')
                    succeeded += 1
            except InterruptedError:
                update_job(db_path, job_id, status='cancelled', phase='cancelled')
                return
            except ElephantLandmarkLoadError:
                logger.exception('Elephant landmark model could not be loaded')
                update_job(db_path, job_id, status='failed', phase='failed',
                           error='大象头眼模型加载失败，请检查网络和模型运行环境后重试。')
                return
            except Exception:
                logger.exception('Elephant detection failed for one photo')
                save_result(db_path, path, {'status': 'error'}, None, version,
                            error='无法读取原图或完成检测，请检查原图并重试。')
                failed += 1
            completed += 1
            update_job(db_path, job_id, completed=completed, succeeded=succeeded, skipped=skipped, failed=failed)
        update_job(db_path, job_id, status='completed', phase='completed')
    except BaseException:
        update_job(db_path, job_id, status='failed', phase='failed', error='检测任务意外停止，可重新选择照片后继续。')
        raise


def main():
    parser = argparse.ArgumentParser(description='Selected-photo elephant detection')
    parser.add_argument('--db', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    os.environ['DB_PATH'], os.environ['FACET_CONFIG'] = args.db, args.config
    from facet import LibraryLock
    try:
        with LibraryLock(args.db, kind='elephant detection'):
            run_job(args.db, args.job_id)
    except Exception:
        logger.exception('Elephant detection worker failed')
        update_job(args.db, args.job_id, status='failed', phase='failed',
                   error='无法启动检测任务，可能有其他照片库任务正在运行。请稍后重试。')
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
