"""Manual culling reuses photo group/review fields; negative IDs are user-set groups."""
import time
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import CurrentUser, require_edition
from api.database import get_db
from api.db_helpers import (
    get_visibility_clause, select_in_chunks, set_photos_rejected,
    is_multi_user_enabled, record_culling_decision, trigger_auto_retrain,
)

router = APIRouter(tags=['manual_culling'])
MAX_MEMBERS = 500


class ManualGroupBody(BaseModel):
    paths: list[str] = Field(min_length=2, max_length=MAX_MEMBERS)


def _clear_rejection(conn, paths, user_id):
    if user_id and is_multi_user_enabled():
        conn.executemany(
            'INSERT INTO user_preferences (user_id, photo_path, is_rejected) VALUES (?, ?, 0) '
            'ON CONFLICT(user_id, photo_path) DO UPDATE SET is_rejected = 0',
            [(user_id, p) for p in paths])
    else:
        conn.executemany('UPDATE photos SET is_rejected = 0 WHERE path = ?', [(p,) for p in paths])


def _members(conn, group_id, user_id):
    vis, params = get_visibility_clause(user_id)
    rows = conn.execute(
        f'SELECT path, burst_reviewed FROM photos WHERE burst_group_id = ? AND {vis} '
        'ORDER BY path', [group_id] + params).fetchall()
    total = conn.execute('SELECT COUNT(*) FROM photos WHERE burst_group_id = ?',
                         [group_id]).fetchone()[0]
    # Never read or confirm only the visible part of a shared group.
    if len(rows) != total:
        return []
    return rows


@router.post('/api/culling-groups/manual')
def save_manual_group(body: ManualGroupBody, user: CurrentUser = Depends(require_edition)):
    from api.routers.burst_culling import _invalidate_culling_groups_cache
    user_id = user.user_id if user else None
    paths = set(body.paths)
    with get_db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        vis, params = get_visibility_clause(user_id)
        visible = {r[0] for r in select_in_chunks(
            conn, f'SELECT path FROM photos WHERE path IN ({{placeholders}}) AND {vis}',
            paths, after=params)}
        if visible != paths:
            raise HTTPException(404, 'Some photos are missing or inaccessible')
        if len(paths) < 2:
            raise HTTPException(400, 'A group needs at least two distinct photos')
        # Automatic detection owns nonnegative IDs. Millisecond IDs remain
        # exact in the browser, and the transaction + minimum prevents collisions.
        minimum = conn.execute('SELECT COALESCE(MIN(burst_group_id), 0) FROM photos').fetchone()[0]
        group_id = min(-int(time.time() * 1000), minimum - 1)
        conn.executemany(
            'UPDATE photos SET burst_group_id = ?, is_burst_lead = 0, '
            'burst_reviewed = 0, similarity_reviewed = 0 WHERE path = ?',
            [(group_id, p) for p in paths])
        _clear_rejection(conn, paths, user_id)
        conn.execute("DELETE FROM stats_cache WHERE key LIKE 'similarity_groups_%' OR key LIKE 'scenes_%'")
        conn.commit()
    _invalidate_culling_groups_cache()
    return {'group_id': group_id, 'count': len(paths)}


def fetch_manual_groups(conn, user, album_id=None, date_from=None, date_to=None):
    """Manual sets stay whole, including previously rejected/reviewed photos."""
    from api.routers.burst_culling import _format_group, _culling_photo_columns
    from api.db_helpers import album_filter_clause, time_window_clauses
    user_id = user.user_id if user else None
    groups = []
    for row in conn.execute('SELECT DISTINCT burst_group_id FROM photos '
                            'WHERE burst_group_id < 0 AND burst_reviewed = 0'):
        group_id = row[0]
        members = _members(conn, group_id, user_id)
        paths = [r['path'] for r in members]
        if len(paths) < 2:
            continue
        # Scope filters include a whole set only when all its members match;
        # never confirm a truncated group or silently decide hidden members.
        album_sql, album_params = album_filter_clause(album_id)
        clauses, window_params = time_window_clauses(date_from, date_to)
        scope = ' AND '.join([album_sql, *clauses])
        rows = select_in_chunks(
            conn, f'SELECT {_culling_photo_columns("photos.path", burst=True)} '
            f'FROM photos WHERE photos.path IN ({{placeholders}}) AND {scope}',
            paths, after=album_params + window_params)
        photos = [dict(r) for r in rows]
        if len(photos) != len(paths):
            continue
        group = _format_group(photos, group_id)
        group.update(group_id=group_id, type='manual', reason='manual')
        groups.append(group)
    return groups


def confirm_manual_group(body, user):
    from api.routers.burst_culling import _invalidate_culling_groups_cache, _sequence_members
    user_id = user.user_id if user else None
    with get_db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if body.group_id >= 0:
            raise HTTPException(404, 'Manual group not found')
        members = _members(conn, body.group_id, user_id)
        paths = {r['path'] for r in members}
        if not paths:
            raise HTTPException(404, 'Manual group not found or inaccessible')
        if (set(body.paths) != paths or len(paths) < 2
                or any(r['burst_reviewed'] for r in members)):
            raise HTTPException(409, 'Group membership changed; refresh the group')
        keep = set(body.keep_paths)
        if not keep <= paths:
            raise HTTPException(400, 'Keep paths must belong to the group')
        rejected = paths - keep
        set_photos_rejected(conn, list(rejected), user_id)
        _clear_rejection(conn, keep, user_id)
        from api.routers.burst_culling import _mark_burst_reviewed, _mark_similarity_reviewed
        _mark_burst_reviewed(conn, list(keep), list(rejected))
        vis, params = get_visibility_clause(user_id)
        _mark_similarity_reviewed(conn, list(paths), vis, params)
        members = select_in_chunks(conn, 'SELECT path, sequence_kind FROM photos '
                                   'WHERE path IN ({placeholders})', paths)
        pairs = record_culling_decision(conn, keep, rejected, _sequence_members(members),
                                user_id=user_id, group_type='manual', commit=False)
        conn.execute("DELETE FROM stats_cache WHERE key LIKE 'similarity_groups_%' OR key LIKE 'scenes_%'")
        conn.commit()
        from db import DEFAULT_DB_PATH
        trigger_auto_retrain(DEFAULT_DB_PATH, user_id, pairs, conn=conn)
    _invalidate_culling_groups_cache()
    return {'status': 'ok', 'kept': len(keep), 'rejected': len(rejected)}
