"""统一编号下的选片列表、确认和人工类型修正。"""
from collections import defaultdict
from fastapi import HTTPException

from api.db_helpers import (
    get_visibility_clause, album_filter_clause, time_window_clauses, select_in_chunks,
    require_active_culling_photos, set_photos_rejected, record_culling_decision,
)
from db.photo_groups import GROUP_TYPES, SEQUENCE_TYPES, invalidate_groups


def is_unified(conn):
    return any(r[1] == 'group_id' for r in conn.execute('PRAGMA table_info(photos)'))


def fetch_groups(conn, user, group_by='all', exclude_rejected=True, album_id=None, date_from=None, date_to=None):
    from api.routers.burst_culling import _culling_photo_columns, _format_group, _get_rejected_paths
    uid = user.user_id if user else None
    vis, params = get_visibility_clause(uid)
    rows = conn.execute(f'SELECT {_culling_photo_columns("photos.path", burst=True)}, '
                        f'group_id,group_type,group_source,burst_reviewed,similarity_reviewed '
                        f'FROM photos WHERE group_id IS NOT NULL AND {vis}', params).fetchall()
    members = defaultdict(list)
    for row in rows:
        members[row['group_id']].append(dict(row))
    totals = {r[0]:r[1] for r in conn.execute('SELECT group_id,COUNT(*) FROM photos WHERE group_id IS NOT NULL GROUP BY group_id')}
    album, album_params = album_filter_clause(album_id)
    window, window_params = time_window_clauses(date_from,date_to)
    scope = ' AND '.join([album,*window])
    scoped = {r[0] for r in conn.execute(f'SELECT path FROM photos WHERE {scope}',album_params+window_params)}
    rejected = _get_rejected_paths(conn,uid) if exclude_rejected else set()
    result = []
    if group_by == 'scene':
        group_by = 'all'
    if group_by == 'manual':
        group_by = 'manual_similar'
    for gid, photos in members.items():
        kind = photos[0]['group_type']
        if kind not in GROUP_TYPES or group_by not in ('all',kind):
            continue
        paths = {p['path'] for p in photos}
        if len(photos) != totals[gid] or not paths <= scoped:
            continue
        review = 'burst_reviewed'
        shown = [p for p in photos if p['path'] not in rejected and not p[review]]
        if len(shown) < 2:
            continue
        group = _format_group(shown,gid)
        group.update(group_id=gid,type=kind,reason=kind,member_paths=sorted(paths),
                     group_source=photos[0]['group_source'])
        result.append(group)
    return result


def checked_members(conn, gid, uid):
    vis, params = get_visibility_clause(uid)
    rows = conn.execute(f'SELECT path,group_type,group_source,burst_reviewed,similarity_reviewed,sequence_kind '
                        f'FROM photos WHERE group_id=? AND {vis}',[gid]+params).fetchall()
    total = conn.execute('SELECT COUNT(*) FROM photos WHERE group_id=?',(gid,)).fetchone()[0]
    if not rows or len(rows) != total:
        raise HTTPException(404,'Group not found or inaccessible')
    return rows


def confirm_group(conn, body, user):
    from api.routers.burst_culling import _get_rejected_paths, _sequence_members
    uid = user.user_id if user else None
    conn.execute('BEGIN IMMEDIATE')
    rows = checked_members(conn,body.group_id,uid)
    kind = rows[0]['group_type']
    requested = 'manual_similar' if body.type == 'manual' else body.type
    if kind != requested:
        raise HTTPException(409,'Group type changed; refresh the group')
    rejected = _get_rejected_paths(conn,uid)
    review = 'burst_reviewed'
    pending = {r['path'] for r in rows if r['path'] not in rejected and not r[review]}
    submitted, keep = set(body.paths), set(body.keep_paths)
    if submitted != pending or len(pending)<2:
        raise HTTPException(409,'Group membership or photo status changed; refresh the group')
    if not keep <= submitted:
        raise HTTPException(400,'Keep paths must belong to the group')
    require_active_culling_photos(conn,submitted,uid)
    reject = submitted-keep
    set_photos_rejected(conn,list(reject),uid)
    conn.executemany('UPDATE photos SET burst_reviewed=1,similarity_reviewed=1 WHERE path=?',[(p,) for p in submitted])
    if kind == 'burst':
        conn.executemany('UPDATE photos SET is_burst_lead=? WHERE path=?',[(int(p in keep),p) for p in submitted])
    pairs = record_culling_decision(conn,keep,reject,_sequence_members(rows),
                                   user_id=uid,group_type=kind,commit=False)
    invalidate_groups(conn)
    conn.commit()
    return {'status':'ok','kept':len(keep),'rejected':len(reject)}, pairs


def change_group_type(conn, body, user):
    """人工修改类型后保留原成员与编号，自动检测不再覆盖。"""
    uid = user.user_id if user else None
    conn.execute('BEGIN IMMEDIATE')
    rows = checked_members(conn,body.group_id,uid)
    if {r['path'] for r in rows} != set(body.paths):
        raise HTTPException(409,'Group membership changed; refresh the group')
    if body.type not in GROUP_TYPES:
        raise HTTPException(400,'Unknown group type')
    lead = sorted(body.paths)[len(set(body.paths))//2]
    conn.executemany(
        "UPDATE photos SET group_type=?,group_source='manual',sequence_kind=?,sequence_ev_offset=NULL,"
        'is_sequence_lead=? WHERE path=?',
        [(body.type,body.type if body.type in SEQUENCE_TYPES else None,
          int(p==lead and body.type in SEQUENCE_TYPES),p) for p in set(body.paths)])
    conn.executemany('DELETE FROM photo_sequence_overrides WHERE photo_path=?',[(p,) for p in set(body.paths)])
    invalidate_groups(conn)
    conn.commit()
    return {'status':'ok','group_id':body.group_id,'type':body.type}
