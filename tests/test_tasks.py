"""统一任务列表的权限、持久任务恢复、去重与进度脱敏验证。"""

import os
import sqlite3
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api import create_app
from api.auth import CurrentUser, require_authenticated
from api.routers import birds, scan, tasks
from db.schema import init_database


@pytest.fixture()
def task_db(tmp_path, monkeypatch):
    path = str(tmp_path / 'tasks.db')
    init_database(path)

    @contextmanager
    def get_db():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr(tasks, 'get_db', get_db)
    monkeypatch.setattr(tasks, 'DEFAULT_DB_PATH', path)
    monkeypatch.setattr(birds, 'DEFAULT_DB_PATH', path)
    monkeypatch.setattr(birds, '_processes', {})
    monkeypatch.setattr(scan, '_scan_state', dict(
        running=False, kind=None, process=None, output_lines=[], started_at=None,
        directories=[], exit_code=None, progress=None,
    ))
    monkeypatch.setattr('facet.library_job_holder', lambda db: None)
    return get_db


def add_bird(get_db, job_id, *, owner='test', status='running', pid=None, cancel=0, created=None):
    with get_db() as conn:
        conn.execute(
            'INSERT INTO bird_detection_jobs '
            '(id,user_id,status,phase,total,completed,paths_json,options_json,created_at,updated_at,pid,cancel_requested) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            (job_id, owner, status, 'detecting', 10, 3, '["/private/photo.jpg"]', '{}',
             created or time.time(), time.time(), pid if pid is not None else os.getpid(), cancel),
        )
        conn.commit()


def test_lists_persisted_jobs_without_browser_id_and_hides_other_owners(task_db, edition_client):
    add_bird(task_db, 'active', cancel=1)
    add_bird(task_db, 'finished', status='completed')
    add_bird(task_db, 'other-user', owner='other')
    response = edition_client.get('/api/tasks')
    assert response.status_code == 200, response.text
    result = response.json()['tasks']
    assert [task['id'] for task in result] == ['active', 'finished']
    assert result[0]['status'] == 'stopping'
    assert result[0]['current'] == 3 and result[0]['total'] == 10
    assert '/private/photo.jpg' not in response.text
    assert 'user_id' not in response.text and 'paths_json' not in response.text


def test_dead_bird_worker_becomes_failed(task_db, edition_client):
    add_bird(task_db, 'dead', pid=2_000_000_000)
    response = edition_client.get('/api/tasks')
    assert response.status_code == 200, response.text
    job = response.json()['tasks'][0]
    assert job['status'] == 'failed' and job['error']


def test_active_jobs_are_not_lost_to_recent_history_limit(task_db, edition_client):
    for index in range(55):
        add_bird(task_db, f'finished-{index}', status='completed')
    add_bird(task_db, 'old-active', created=1)
    result = edition_client.get('/api/tasks').json()['tasks']
    assert len(result) == 51
    assert result[0]['id'] == 'old-active'


def test_bird_library_lock_does_not_duplicate_the_persisted_task(task_db, edition_client, monkeypatch):
    add_bird(task_db, 'active')
    monkeypatch.setattr('facet.library_job_holder', lambda db: {
        'pid': os.getpid(), 'kind': 'bird detection', 'started_at': time.time(),
    })
    result = edition_client.get('/api/tasks').json()['tasks']
    assert len(result) == 1 and result[0]['id'] == 'active'


def test_local_scan_progress_excludes_files_logs_and_duplicate_lock(task_db, edition_client, monkeypatch):
    from types import SimpleNamespace
    scan._scan_state.update(running=True, kind='scan', process=SimpleNamespace(pid=123), started_at=100,
                            progress={'phase': 'scoring', 'current': 4, 'total': 20,
                                      'current_file': '/secret.jpg'}, output_lines=['secret log'])
    monkeypatch.setattr('facet.library_job_holder', lambda db: {'pid': 123, 'kind': 'scan', 'started_at': 100})
    response = edition_client.get('/api/tasks')
    assert response.status_code == 200, response.text
    result = response.json()['tasks']
    assert len(result) == 1
    assert (result[0]['kind'], result[0]['current'], result[0]['total']) == ('scan', 4, 20)
    assert 'secret' not in response.text


def test_external_job_retains_kind_with_unknown_progress(task_db, edition_client, monkeypatch):
    monkeypatch.setattr('facet.library_job_holder', lambda db: {'pid': 123, 'kind': 'retrain', 'started_at': 100})
    result = edition_client.get('/api/tasks').json()['tasks']
    assert len(result) == 1
    assert result[0]['kind'] == 'retrain' and result[0]['status'] == 'running'
    assert result[0]['current'] is None and result[0]['total'] is None


def test_legacy_scan_with_fresh_heartbeat_is_visible(task_db, edition_client):
    with task_db() as conn:
        conn.execute("INSERT INTO scan_runs (mode,total_files,processed_files,heartbeat_at) "
                     "VALUES ('single-pass',100,7,datetime('now'))")
        conn.commit()
    result = edition_client.get('/api/tasks').json()['tasks']
    assert len(result) == 1
    assert result[0]['kind'] == 'scan' and result[0]['current'] == 7


def test_requires_edition_permission(task_db):
    app = create_app()
    app.dependency_overrides[require_authenticated] = lambda: CurrentUser(user_id='viewer')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CurrentUser, 'is_edition', property(lambda self: False))
        assert TestClient(app).get('/api/tasks').status_code == 403


def test_empty_when_no_jobs(task_db, edition_client):
    assert edition_client.get('/api/tasks').json() == {'tasks': []}


def test_terminates_only_owned_bird_job_and_keeps_progress(task_db, edition_client):
    add_bird(task_db, 'owned')
    add_bird(task_db, 'other', owner='other-user')
    response = edition_client.post('/api/tasks/owned/terminate', json={'confirm': True})
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'stopping'
    assert response.json()['current'] == 3 and response.json()['can_terminate'] is False
    with task_db() as conn:
        assert conn.execute("SELECT cancel_requested FROM bird_detection_jobs WHERE id='owned'").fetchone()[0] == 1
        assert conn.execute("SELECT cancel_requested FROM bird_detection_jobs WHERE id='other'").fetchone()[0] == 0
    assert edition_client.post('/api/tasks/other/terminate', json={'confirm': True}).status_code == 404


def test_termination_requires_confirmation_and_edition_access(task_db, edition_client, regular_client):
    add_bird(task_db, 'active')
    assert edition_client.post('/api/tasks/active/terminate', json={'confirm': False}).status_code == 400
    assert edition_client.post('/api/tasks/active/terminate').status_code == 422
    assert regular_client.post('/api/tasks/active/terminate', json={'confirm': True}).status_code == 403


def test_finishing_bird_task_does_not_get_restarted_or_lose_results(task_db, edition_client):
    add_bird(task_db, 'done', status='completed')
    response = edition_client.post('/api/tasks/done/terminate', json={'confirm': True})
    assert response.status_code == 200
    assert response.json()['status'] == 'completed' and response.json()['current'] == 3
    assert response.json()['can_terminate'] is False


def managed_library_process(kind='recompute'):
    process = SimpleNamespace(pid=123, poll=Mock(return_value=None), terminate=Mock())
    scan._scan_state.update(running=True, kind=kind, process=process, started_at=100,
                            user_id='test', termination_requested=False,
                            progress={'current': 4, 'total': 20})
    return process


@pytest.mark.parametrize('kind', ['scan', 'recompute', 'panoramas'])
def test_managed_library_task_terminates_once_and_reports_stopped_state(task_db, edition_client, kind):
    process = managed_library_process(kind)
    task = edition_client.get('/api/tasks').json()['tasks'][0]
    assert task['can_terminate'] is True
    for _ in range(2):
        response = edition_client.post(f"/api/tasks/{task['id']}/terminate", json={'confirm': True})
        assert response.status_code == 200, response.text
        assert response.json()['status'] == 'stopping' and response.json()['current'] == 4
    process.terminate.assert_called_once()
    scan._scan_state.update(running=False, exit_code=-15)
    task = edition_client.get('/api/tasks').json()['tasks'][0]
    assert task['status'] == 'cancelled' and task['can_terminate'] is False


def test_stale_or_external_task_id_never_signals_a_process(task_db, edition_client):
    process = managed_library_process()
    for task_id in ('library:123:99', 'library:987:100', 'scan:1'):
        assert edition_client.post(f'/api/tasks/{task_id}/terminate', json={'confirm': True}).status_code == 409
    process.terminate.assert_not_called()


def test_other_user_cannot_terminate_library_task_in_multi_user_install(task_db, edition_client, monkeypatch):
    process = managed_library_process()
    scan._scan_state['user_id'] = 'other-user'
    monkeypatch.setattr(tasks, 'is_multi_user_enabled', lambda: True)
    task = edition_client.get('/api/tasks').json()['tasks'][0]
    assert task['can_terminate'] is False
    assert edition_client.post(f"/api/tasks/{task['id']}/terminate", json={'confirm': True}).status_code == 403
    process.terminate.assert_not_called()


def test_process_exit_or_signal_failure_does_not_leave_task_stopping(task_db, edition_client):
    process = managed_library_process()
    process.poll.return_value = 0
    url = '/api/tasks/library:123:100/terminate'
    assert edition_client.post(url, json={'confirm': True}).status_code == 409
    process.terminate.assert_not_called()
    process.poll.return_value = None
    process.terminate.side_effect = PermissionError('cannot signal')
    assert edition_client.post(url, json={'confirm': True}).status_code == 409
    assert scan._scan_state['termination_requested'] is False
