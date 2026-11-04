"""汇总照片库作业和当前用户的鸟类检测任务，不返回文件路径或后台日志。"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import CurrentUser, require_edition
from api.database import get_db
from db.connection import DEFAULT_DB_PATH
from processing.bird_detection import TERMINAL

router = APIRouter(prefix='/api/tasks', tags=['tasks'])


class BackgroundTask(BaseModel):
    id: str
    kind: str
    status: str
    phase: str | None = None
    current: int | None = None
    total: int | None = None
    started_at: float | None = None
    failed: int | None = None
    skipped: int | None = None
    succeeded: int | None = None
    error: str | None = None


class BackgroundTasksResponse(BaseModel):
    tasks: list[BackgroundTask]


def _library_tasks(bird_pids=()):
    from facet import library_job_holder
    from api.routers.scan import _scan_lock, _scan_state, _configured_scan_stale_seconds

    # 只读取标量和进度副本；绝不把扫描目录、当前文件或日志带入统一列表。
    with _scan_lock:
        state = dict(_scan_state)
        progress = dict(state.get('progress') or {})
    holder = library_job_holder(DEFAULT_DB_PATH)
    tasks = []
    local_pid = getattr(state.get('process'), 'pid', None)
    if state.get('kind') and state.get('started_at'):
        running = state['running']
        tasks.append(BackgroundTask(
            id=f"library:{local_pid}:{state['started_at']}", kind=state['kind'],
            status='running' if running else ('completed' if state['exit_code'] == 0 else 'failed'),
            started_at=state['started_at'], phase=progress.get('phase'),
            current=progress.get('current'), total=progress.get('total'),
        ))
    bird_holder = holder and holder.get('kind') == 'bird detection' and holder.get('pid') in bird_pids
    if holder and not bird_holder and not (state['running'] and local_pid == holder.get('pid')):
        tasks.append(BackgroundTask(
            id=f"library:{holder.get('pid')}:{holder.get('started_at')}",
            kind='birds' if holder.get('kind') == 'bird detection' else (holder.get('kind') or 'maintenance'),
            status='running',
            started_at=holder.get('started_at'),
        ))

    # 兼容没有照片库作业锁的旧扫描进程，并补足其他服务进程无法读取的扫描进度。
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, processed_files, total_files, failed_files, "
            "CAST(strftime('%s', started_at) AS REAL) AS started_at "
            "FROM scan_runs WHERE status='running' AND finished_at IS NULL "
            "AND COALESCE(heartbeat_at, started_at) >= datetime('now', ?) "
            "ORDER BY id DESC LIMIT 1", (f'-{int(_configured_scan_stale_seconds())} seconds',),
        ).fetchone()
    if row:
        scan = next((task for task in tasks if task.kind == 'scan' and task.status == 'running'), None)
        if scan is None:
            scan = BackgroundTask(id=f"scan:{row['id']}", kind='scan', status='running',
                                  started_at=row['started_at'])
            tasks.append(scan)
        if scan.current is None:
            scan.current, scan.total = row['processed_files'], row['total_files']
            scan.failed = row['failed_files']
    return tasks


@router.get('', response_model=BackgroundTasksResponse)
def background_tasks(user: CurrentUser = Depends(require_edition)):
    from api.routers.birds import _owned_job

    with get_db() as conn:
        # 活动任务全部保留，另取最近 50 个结束任务；不依赖浏览器保存的单个任务编号。
        columns = 'id,status,phase,total,completed,succeeded,failed,skipped,cancel_requested,error,created_at,pid'
        owner = "COALESCE(user_id, '') = ?"
        params = (user.user_id or '',)
        rows = conn.execute(
            f'SELECT {columns} FROM bird_detection_jobs WHERE {owner} '
            "AND status NOT IN ('completed','cancelled','failed')", params,
        ).fetchall()
        rows += conn.execute(
            f'SELECT {columns} FROM bird_detection_jobs WHERE {owner} '
            "AND status IN ('completed','cancelled','failed') ORDER BY created_at DESC LIMIT 50", params,
        ).fetchall()
    tasks = _library_tasks({row['pid'] for row in rows if row['status'] not in TERMINAL})
    for row in rows:
        job = dict(row)
        if job['status'] not in TERMINAL:
            # 复用单任务查询的存活检查，避免服务重启后把已退出进程永远显示为运行中。
            job.update(_owned_job(job['id'], user))
        tasks.append(BackgroundTask(
            id=job['id'], kind='birds',
            status='stopping' if job['cancel_requested'] and job['status'] not in TERMINAL else job['status'],
            phase=job['phase'], current=job['completed'], total=job['total'],
            started_at=job['created_at'], succeeded=job['succeeded'], failed=job['failed'],
            skipped=job['skipped'], error=job['error'],
        ))
    active = {'queued', 'running', 'stopping'}
    tasks.sort(key=lambda task: (task.status not in active, -(task.started_at or 0), task.id))
    return BackgroundTasksResponse(tasks=tasks)
