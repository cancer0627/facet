"""手动解散只清空已有组编号，不增加数据库表。"""
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from unittest.mock import patch

from api.routers import burst_culling as culling
from api.routers import manual_culling as manual
from tests.test_manual_culling import db, create


def snapshot(conn):
    return [dict(r) for r in conn.execute('SELECT * FROM photos ORDER BY path')]


def dismiss(group, paths=('a', 'b')):
    return culling.dismiss_culling_group(culling.CullingDismissBody(
        type='manual', group_id=group['group_id'], paths=list(paths)), None)


def test_dissolve_only_changes_group_id(db):
    group = create()
    before = snapshot(db)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    with patch.object(culling, 'get_db', manual.get_db):
        dismiss(group)
    for row in before:
        if row['path'] in ('a', 'b'):
            row['burst_group_id'] = None
            row['group_id'] = row['group_type'] = row['group_source'] = None
    assert snapshot(db) == before
    assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == tables
    assert not db.execute("SELECT name FROM sqlite_master WHERE name='culling_dismissals'").fetchall()


def test_stale_membership_cannot_dissolve(db):
    group = create(['a', 'b', 'c'])
    before = snapshot(db)
    with patch.object(culling, 'get_db', manual.get_db):
        with pytest.raises(HTTPException) as exc:
            dismiss(group)
    assert exc.value.status_code == 409
    assert snapshot(db) == before


def test_partial_visibility_cannot_dissolve(db):
    group = create()
    before = snapshot(db)
    with patch.object(culling, 'get_db', manual.get_db), patch.object(
            manual, 'get_visibility_clause', return_value=("path != ?", ['b'])):
        with pytest.raises(HTTPException) as exc:
            dismiss(group)
    assert exc.value.status_code == 404
    assert snapshot(db) == before


@pytest.mark.parametrize('kind', ['burst', 'similar', 'scene', 'bracket', 'panorama', 'hdr_panorama'])
def test_automatic_group_dissolution_not_supported(kind):
    with pytest.raises(ValidationError):
        culling.CullingDismissBody(type=kind, group_id=5, paths=['a', 'b'])


def test_external_rejections_refresh_automatic_groups_even_with_warm_cache(db):
    import asyncio
    db.execute("UPDATE photos SET group_id=5, group_type='burst', group_source='auto', is_rejected=0, burst_reviewed=0 "
               "WHERE path IN ('a', 'b', 'c')")
    db.commit()

    def feed():
        return asyncio.run(culling.api_culling_groups(
            page=1, per_page=20, similarity_threshold=.85, seed=0, exclude_rejected=True,
            sort='easiest', direction='', group_by='burst', album_id=None,
            date_from=None, date_to=None, user=None))

    with patch.object(culling, 'get_db', manual.get_db), patch.object(
            culling, 'get_visibility_clause', return_value=('1=1', [])), patch.object(
            culling, 'attach_keeper_probs'):
        culling._invalidate_culling_groups_cache()
        assert feed()['total_groups'] == 1
        db.execute("UPDATE photos SET is_rejected=1 WHERE path IN ('b', 'c')")
        db.commit()
        assert feed()['total_groups'] == 0
        culling._invalidate_culling_groups_cache()
