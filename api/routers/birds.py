"""只对选中的可见照片运行鸟类检测，并提供持久进度和局部预览。"""

import json
import subprocess
import sys
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from api.auth import CurrentUser, get_optional_user, require_authenticated, require_edition
from api.config import _CONFIG_PATH, invalidate_stats_cache
from api.database import get_db
from api.db_helpers import get_visibility_clause, select_in_chunks
from db.connection import DEFAULT_DB_PATH
from models.bird_detector import DEFAULT_MIN_AREA_PERCENT
from processing.bird_detection import TERMINAL, get_job, source_signature, update_job

router = APIRouter(prefix='/api/birds', tags=['birds'])
_processes = {}


def _watch_process(job_id, process):
    """即使用户关闭进度窗口，也回收子进程并记录意外退出。"""
    process.wait()
    _processes.pop(job_id, None)
    job = get_job(DEFAULT_DB_PATH, job_id)
    if job and job['status'] not in TERMINAL:
        update_job(DEFAULT_DB_PATH, job_id, status='failed', phase='failed',
                   error='检测进程已停止，可重新选择照片后继续。')
    invalidate_stats_cache()


class BirdStartBody(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=10000)
    tiled: bool = True
    recompute: bool = False
    min_area_percent: float = Field(default=DEFAULT_MIN_AREA_PERCENT, ge=0, le=100, allow_inf_nan=False)

    @field_validator('paths')
    @classmethod
    def validate_paths(cls, paths):
        if any(not path.strip() for path in paths):
            raise ValueError('Photo paths must not be empty')
        return list(dict.fromkeys(paths))


class BirdJob(BaseModel):
    id: str
    status: str
    phase: str
    total: int
    completed: int
    succeeded: int
    failed: int
    skipped: int
    cancel_requested: int
    error: str | None = None


class BirdCancelBody(BaseModel):
    confirm: bool


class BirdPhotoResult(BaseModel):
    status: str | None = None
    score: float | None = None
    count: int | None = None
    best_index: int | None = None
    bbox: list[float] | None = None
    width: int | None = None
    height: int | None = None
    birds: list[dict] = []
    error: str | None = None
    stale: bool = False
    outdated: bool = False
    min_area_percent: float | None = None
    eligible_count: int | None = None
    excluded_small_count: int | None = None
    focus_unverified_count: int | None = None


def _visible_photo(path, user):
    clause, params = get_visibility_clause(user.user_id if user else None)
    with get_db() as conn:
        row = conn.execute(f'SELECT * FROM photos WHERE path=? AND {clause}', [path, *params]).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail='Photo not found')
    return dict(row)


def _owned_job(job_id, user):
    job = get_job(DEFAULT_DB_PATH, job_id)
    if not job or (job['user_id'] or '') != (user.user_id or ''):
        raise HTTPException(status_code=404, detail='Job not found')
    process = _processes.get(job_id)
    if process and process.poll() is not None:
        _processes.pop(job_id, None)
    if job['status'] not in TERMINAL:
        import psutil
        alive = False
        if job['pid']:
            try:
                alive = psutil.Process(job['pid']).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                pass
        if not alive and (job['pid'] or time.time() - job['created_at'] > 60):
            update_job(DEFAULT_DB_PATH, job_id, status='failed', phase='failed',
                       error='检测进程已停止，可重新选择照片后继续。')
            job = get_job(DEFAULT_DB_PATH, job_id)
    return job


@router.post('/jobs', response_model=BirdJob)
def start_bird_job(body: BirdStartBody, user: CurrentUser = Depends(require_edition)):
    from api.routers.scan import _recompute_conflict_detail, _cross_process_job_env

    conflict = _recompute_conflict_detail()
    if conflict:
        raise HTTPException(status_code=409, detail=conflict)
    clause, params = get_visibility_clause(user.user_id)
    with get_db() as conn:
        visible = list(select_in_chunks(
            conn, f'SELECT path FROM photos WHERE path IN ({{placeholders}}) AND {clause}',
            body.paths, after=params,
        ))
        if len(visible) != len(body.paths):
            raise HTTPException(status_code=404, detail='One or more photos were not found')
        job_id = uuid.uuid4().hex
        now = time.time()
        conn.execute(
            'INSERT INTO bird_detection_jobs (id,user_id,status,phase,total,paths_json,options_json,created_at,updated_at) '
            'VALUES (?,?,?,?,?,?,?,?,?)',
            (job_id, user.user_id, 'queued', 'preparing', len(body.paths), json.dumps(body.paths),
             json.dumps({'tiled': body.tiled, 'recompute': body.recompute,
                         'min_area_percent': body.min_area_percent}), now, now),
        )
        conn.commit()
    project = Path(__file__).resolve().parents[2]
    log_dir = Path(DEFAULT_DB_PATH).resolve().parent / '.facet_cache' / 'birds'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'{job_id}.log').open('ab') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'processing.bird_detection', '--db', str(Path(DEFAULT_DB_PATH).resolve()),
                 '--config', str(Path(_CONFIG_PATH).resolve()), '--job-id', job_id],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, env=_cross_process_job_env(),
            )
        _processes[job_id] = process
        update_job(DEFAULT_DB_PATH, job_id, pid=process.pid)
        threading.Thread(target=_watch_process, args=(job_id, process), daemon=True).start()
    except OSError:
        update_job(DEFAULT_DB_PATH, job_id, status='failed', phase='failed', error='无法启动后台检测进程。')
        raise HTTPException(status_code=500, detail='Could not start bird detection')
    return get_job(DEFAULT_DB_PATH, job_id)


@router.get('/jobs/{job_id}', response_model=BirdJob)
def bird_job_status(job_id: str, user: CurrentUser = Depends(require_edition)):
    job = _owned_job(job_id, user)
    if job['status'] in TERMINAL:
        invalidate_stats_cache()
    return job


@router.post('/jobs/{job_id}/cancel', response_model=BirdJob)
def cancel_bird_job(job_id: str, body: BirdCancelBody, user: CurrentUser = Depends(require_edition)):
    if not body.confirm:
        raise HTTPException(status_code=400, detail='Cancellation must be confirmed')
    job = _owned_job(job_id, user)
    if job['status'] not in TERMINAL:
        update_job(DEFAULT_DB_PATH, job_id, cancel_requested=1)
    return get_job(DEFAULT_DB_PATH, job_id)


@router.get('/photo', response_model=BirdPhotoResult)
def bird_photo(path: str = Query(...), user: CurrentUser = Depends(require_authenticated)):
    from models.bird_detector import RESULT_VERSION_PREFIX
    row = _visible_photo(path, user)
    result = json.loads(row.get('bird_detection_json') or '{}')
    result['status'] = row.get('bird_status')
    result['error'] = row.get('bird_error')
    result['outdated'] = bool(row.get('bird_version') and not row['bird_version'].startswith(RESULT_VERSION_PREFIX))
    if result['outdated']:
        result['score'] = None
    if result['status'] and row.get('bird_source_signature'):
        from api.path_validation import resolve_photo_disk_path
        try:
            result['stale'] = source_signature(resolve_photo_disk_path(path)) != row.get('bird_source_signature')
        except (OSError, HTTPException):
            result['stale'] = True
    return result


@router.get('/preview')
def bird_preview(path: str = Query(...), index: int | None = Query(None, ge=0),
                 user: CurrentUser | None = Depends(get_optional_user)):
    from api.path_validation import resolve_photo_disk_path
    from utils.image_loading import load_image_from_path
    from PIL import ImageDraw

    row = _visible_photo(path, user)
    disk_path = resolve_photo_disk_path(path)
    if source_signature(disk_path) != row.get('bird_source_signature'):
        raise HTTPException(status_code=409, detail='Original image changed; run bird detection again')
    result = json.loads(row.get('bird_detection_json') or '{}')
    birds = result.get('birds', [])
    if index is not None and index >= len(birds):
        raise HTTPException(status_code=404, detail='Bird not found')
    image, _ = load_image_from_path(disk_path, use_thumbnail=False)
    if image is None:
        raise HTTPException(status_code=404, detail='Original image unavailable')
    try:
        original_width, original_height = image.size
        if index is not None:
            box = birds[index]['bbox']
            rect = (int(box[0] * image.width), int(box[1] * image.height),
                    int(box[2] * image.width), int(box[3] * image.height))
            cropped = image.crop(rect)
            image.close()
            image = cropped
            image.thumbnail((960, 960))
            draw = ImageDraw.Draw(image)
            for outline in birds[index].get('mask_outline', []):
                points = [((x * original_width - rect[0]) * image.width / (rect[2] - rect[0]),
                           (y * original_height - rect[1]) * image.height / (rect[3] - rect[1])) for x, y in outline]
                if len(points) > 2:
                    draw.line(points + [points[0]], fill='#16a34a', width=2)
        else:
            image.thumbnail((1280, 1280))
            draw = ImageDraw.Draw(image)
            for i, bird in enumerate(birds):
                box = bird['bbox']
                color = '#16a34a' if i == result.get('best_index') else '#f97316'
                rect = (box[0] * image.width, box[1] * image.height,
                        box[2] * image.width, box[3] * image.height)
                draw.rectangle(rect, outline=color, width=3)
                for outline in bird.get('mask_outline', []):
                    points = [(x * image.width, y * image.height) for x, y in outline]
                    if len(points) > 2:
                        draw.line(points + [points[0]], fill=color, width=2)
                draw.text((rect[0] + 3, rect[1] + 3), str(i + 1), fill=color, stroke_width=1, stroke_fill='white')
        out = BytesIO()
        image.convert('RGB').save(out, format='JPEG', quality=90)
        return Response(out.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})
    finally:
        image.close()
