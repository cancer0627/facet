"""Manual culling reuses photo group/review fields; negative IDs are user-set groups."""
import time
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import CurrentUser, require_edition
from api.database import get_db
from api.db_helpers import (
    get_visibility_clause, select_in_chunks, set_photos_rejected,
    record_culling_decision, trigger_auto_retrain,
    require_active_culling_photos, is_multi_user_enabled,
)

router = APIRouter(tags=['manual_culling'])
MAX_MEMBERS = 500


class ManualGroupBody(BaseModel):
    paths: list[str] = Field(min_length=2, max_length=MAX_MEMBERS)


def _restore_selected_rejections(conn, paths, user_id):
    """仅在用户主动建组时，恢复本次明确选择的淘汰照片。"""
    if user_id and is_multi_user_enabled():
        conn.executemany(
            'UPDATE user_preferences SET is_rejected = 0 '
            'WHERE user_id = ? AND photo_path = ? AND is_rejected = 1',
            [(user_id, p) for p in paths])
    else:
        conn.executemany(
            'UPDATE photos SET is_rejected = 0 WHERE path = ? AND is_rejected = 1',
            [(p,) for p in paths])


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
        _restore_selected_rejections(conn, paths, user_id)
        conn.executemany('DELETE FROM photo_sequence_overrides WHERE photo_path=?', [(p,) for p in paths])
        # Automatic detection owns nonnegative IDs. Millisecond IDs remain
        # exact in the browser, and the transaction + minimum prevents collisions.
        minimum = conn.execute('SELECT COALESCE(MIN(group_id), 0) FROM photos').fetchone()[0]
        group_id = min(-int(time.time() * 1000), minimum - 1)
        conn.executemany(
            "UPDATE photos SET group_id = ?, group_type='manual_similar', group_source='manual', "
            'sequence_kind=NULL, sequence_ev_offset=NULL, is_sequence_lead=0, is_burst_lead = 0, '
            'burst_reviewed = 0, similarity_reviewed = 0 WHERE path = ?',
            [(group_id, p) for p in paths])
        conn.execute("DELETE FROM stats_cache WHERE key LIKE 'similarity_groups_%' OR key LIKE 'scenes_%'")
        conn.commit()
    _invalidate_culling_groups_cache()
    return {'group_id': group_id, 'count': len(paths)}


def fetch_manual_groups(conn, user, album_id=None, date_from=None, date_to=None, exclude_rejected=True):
    """默认只显示至少有两张未淘汰照片的手动组，成员关联保持不变。"""
    from api.routers.burst_culling import _format_group, _culling_photo_columns
    from api.db_helpers import album_filter_clause, time_window_clauses
    user_id = user.user_id if user else None
    from api.routers.burst_culling import _get_rejected_paths
    rejected = _get_rejected_paths(conn, user_id) if exclude_rejected else set()
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
        photos = [p for p in photos if p['path'] not in rejected]
        if len(photos) < 2:
            continue
        group = _format_group(photos, group_id)
        group.update(group_id=group_id, type='manual_similar', reason='manual', member_paths=paths)
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
        from api.routers.burst_culling import _get_rejected_paths
        active = paths - _get_rejected_paths(conn, user_id)
        submitted = set(body.paths)
        if (submitted not in (paths, active) or len(submitted) < 2
                or any(r['burst_reviewed'] for r in members)):
            raise HTTPException(409, 'Group membership changed; refresh the group')
        keep = set(body.keep_paths)
        if not keep <= submitted:
            raise HTTPException(400, 'Keep paths must belong to the group')
        require_active_culling_photos(conn, submitted, user_id)
        rejected = submitted - keep
        set_photos_rejected(conn, list(rejected), user_id)
        from api.routers.burst_culling import _mark_burst_reviewed, _mark_similarity_reviewed
        _mark_burst_reviewed(conn, list(keep), list(paths - keep))
        vis, params = get_visibility_clause(user_id)
        _mark_similarity_reviewed(conn, list(paths), vis, params)
        members = select_in_chunks(conn, 'SELECT path, sequence_kind FROM photos '
                                   'WHERE path IN ({placeholders})', paths)
        pairs = record_culling_decision(conn, keep, rejected, _sequence_members(members),
                                user_id=user_id, group_type='manual_similar', commit=False)
        conn.execute("DELETE FROM stats_cache WHERE key LIKE 'similarity_groups_%' OR key LIKE 'scenes_%'")
        conn.commit()
        from db import DEFAULT_DB_PATH
        trigger_auto_retrain(DEFAULT_DB_PATH, user_id, pairs, conn=conn)
    _invalidate_culling_groups_cache()
    return {'status': 'ok', 'kept': len(keep), 'rejected': len(rejected)}
