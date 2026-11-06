"""汇总照片库作业和当前用户的鸟类检测任务，不返回文件路径或后台日志。"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import CurrentUser, is_multi_user_enabled, require_edition
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
    can_terminate: bool = False


class BackgroundTasksResponse(BaseModel):
    tasks: list[BackgroundTask]


class TaskTerminateBody(BaseModel):
    confirm: bool


def _library_task_id(state):
    return f"library:{getattr(state.get('process'), 'pid', None)}:{state.get('started_at')}"


def _can_control_library(state, user):
    """只控制本服务实际启动的进程；多用户安装还须核对发起人。"""
    process = state.get('process')
    return bool(user and callable(getattr(process, 'terminate', None)) and (
        not is_multi_user_enabled() or user.is_superadmin
        or ('user_id' in state and state['user_id'] == user.user_id)
    ))


def _library_tasks(bird_pids=(), user=None):
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
            id=_library_task_id(state), kind=state['kind'],
            status=('stopping' if running else 'cancelled') if state.get('termination_requested') else
                   ('running' if running else ('completed' if state['exit_code'] == 0 else 'failed')),
            started_at=state['started_at'], phase=progress.get('phase'),
            current=progress.get('current'), total=progress.get('total'),
            can_terminate=running and not state.get('termination_requested', False) and _can_control_library(state, user),
        ))
    bird_holder = holder and holder.get('kind') in ('bird detection', 'elephant detection') and holder.get('pid') in bird_pids
    if holder and not bird_holder and not (state['running'] and local_pid == holder.get('pid')):
        tasks.append(BackgroundTask(
            id=f"library:{holder.get('pid')}:{holder.get('started_at')}",
            kind={'bird detection': 'birds', 'elephant detection': 'elephants'}.get(holder.get('kind'), holder.get('kind') or 'maintenance'),
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
        scan = next((task for task in tasks if task.kind == 'scan' and task.status in ('running', 'stopping')), None)
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
        rows = []
        for kind, table in (('birds', 'bird_detection_jobs'), ('elephants', 'elephant_detection_jobs')):
            active_rows = conn.execute(
                f'SELECT {columns} FROM {table} WHERE {owner} '
                "AND status NOT IN ('completed','cancelled','failed')", params,
            ).fetchall()
            finished_rows = conn.execute(
                f'SELECT {columns} FROM {table} WHERE {owner} '
                "AND status IN ('completed','cancelled','failed') ORDER BY created_at DESC LIMIT 50", params,
            ).fetchall()
            rows.extend({**dict(row), '_kind': kind} for row in [*active_rows, *finished_rows])
    tasks = _library_tasks({row['pid'] for row in rows if row['status'] not in TERMINAL}, user)
    for row in rows:
        job = dict(row)
        if job['status'] not in TERMINAL:
            # 复用单任务查询的存活检查，避免服务重启后把已退出进程永远显示为运行中。
            if job['_kind'] == 'elephants':
                from api.routers.elephants import _owned_job as owned_elephant_job
                job.update(owned_elephant_job(job['id'], user))
            else:
                job.update(_owned_job(job['id'], user))
        tasks.append(BackgroundTask(
            id=job['id'], kind=job['_kind'],
            status='stopping' if job['cancel_requested'] and job['status'] not in TERMINAL else job['status'],
            phase=job['phase'], current=job['completed'], total=job['total'],
            started_at=job['created_at'], succeeded=job['succeeded'], failed=job['failed'],
            skipped=job['skipped'], error=job['error'],
            can_terminate=job['status'] not in TERMINAL and not job['cancel_requested'],
        ))
    active = {'queued', 'running', 'stopping'}
    tasks.sort(key=lambda task: (task.status not in active, -(task.started_at or 0), task.id))
    return BackgroundTasksResponse(tasks=tasks)


@router.post('/{task_id}/terminate', response_model=BackgroundTask)
def terminate_task(task_id: str, body: TaskTerminateBody, user: CurrentUser = Depends(require_edition)):
    """终止已核对身份的任务，保留已提交结果；不接受客户端提供的进程编号。"""
    if not body.confirm:
        raise HTTPException(status_code=400, detail='Termination must be confirmed')
    if task_id.startswith(('library:', 'scan:')):
        from api.routers.scan import _scan_lock, _scan_state

        with _scan_lock:
            if task_id != _library_task_id(_scan_state):
                raise HTTPException(status_code=409, detail='This task cannot be controlled by this service')
            if not _can_control_library(_scan_state, user):
                raise HTTPException(status_code=403, detail='Task control is not permitted')
            process = _scan_state['process']
            if _scan_state['running'] and not _scan_state.get('termination_requested'):
                if process.poll() is not None:
                    raise HTTPException(status_code=409, detail='Task has already stopped; refresh the list')
                _scan_state['termination_requested'] = True
                try:
                    process.terminate()
                except OSError:
                    _scan_state['termination_requested'] = False
                    raise HTTPException(status_code=409, detail='Task could not be terminated; refresh the list')
    else:
        from api.routers.birds import BirdCancelBody, cancel_bird_job

        from processing.elephant_detection import get_job as get_elephant_job

        if get_elephant_job(DEFAULT_DB_PATH, task_id):
            from api.routers.elephants import ElephantCancelBody, cancel_elephant_job
            cancel_elephant_job(task_id, ElephantCancelBody(confirm=True), user)
        else:
            cancel_bird_job(task_id, BirdCancelBody(confirm=True), user)
    # 使用统一列表中的状态，避免操作返回值与刷新后的状态出现分歧。
    task = next((task for task in background_tasks(user).tasks if task.id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail='Task not found')
    return task
