"""普通选片共享淘汰状态，不得通过其他分组恢复已淘汰照片。"""
import asyncio
import sqlite3
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from api.db_helpers import require_active_culling_photos
from api.routers import burst_culling as culling, manual_culling as manual, scenes
from tests.test_manual_culling import db, create


def snapshot(conn):
    return [tuple(r) for r in conn.execute('SELECT * FROM photos ORDER BY path')]


@pytest.mark.parametrize('kind', ['manual', 'burst', 'similar', 'bracket', 'panorama', 'hdr_panorama'])
@pytest.mark.parametrize('rejected_path', ['a', 'b'])
def test_stale_confirmation_is_atomic_and_records_no_training(db, kind, rejected_path):
    group_id = -5 if kind == 'manual' else 5
    sequence_kind = kind if kind in ('bracket', 'panorama', 'hdr_panorama') else None
    db.execute("UPDATE photos SET group_id=?, group_type=?, group_source='auto', burst_reviewed=0, similarity_reviewed=0, "
               "sequence_kind=? WHERE path IN ('a', 'b', 'c')", [group_id, 'manual_similar' if kind == 'manual' else kind, sequence_kind])
    db.execute('UPDATE photos SET is_rejected=1 WHERE path=?', [rejected_path])
    db.commit()
    before = snapshot(db)
    body = culling.CullingConfirmBody(group_id=group_id, type=kind,
                                     paths=['a', 'b', 'c'], keep_paths=['a'])

    async def same_thread(fn, *args):
        return fn(*args)

    with patch.object(culling, 'get_db', manual.get_db), \
            patch.object(scenes, 'get_db', manual.get_db), \
            patch.object(culling.asyncio, 'to_thread', side_effect=same_thread):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(culling._dispatch_culling_confirm(body, None))
    assert exc.value.status_code == 409
    assert snapshot(db) == before
    assert db.execute('SELECT COUNT(*) FROM comparisons').fetchone()[0] == 0


def test_creating_manual_group_restores_only_explicitly_selected_photos(db):
    db.execute("UPDATE photos SET is_rejected=1 WHERE path='b'")
    db.commit()
    create(['a', 'b', 'c'])
    assert dict(db.execute('SELECT path, is_rejected FROM photos')) == {
        'a': 0, 'b': 0, 'c': 0, 'd': 1}
    assert {tuple(r) for r in db.execute('SELECT star_rating, is_favorite FROM photos')} == {(4, 1)}
    feed = manual.fetch_manual_groups(db, None)
    assert {p['path'] for p in feed[0]['photos']} == {'a', 'b', 'c'}


def test_group_creation_failure_rolls_back_restoration(db):
    db.execute("UPDATE photos SET is_rejected=1 WHERE path='b'")
    db.commit()
    before = snapshot(db)
    with patch.object(manual.time, 'time', side_effect=RuntimeError('creation failed')):
        with pytest.raises(RuntimeError):
            create(['a', 'b', 'c'])
    assert snapshot(db) == before


def test_inaccessible_selection_does_not_restore_any_photo(db):
    db.execute("UPDATE photos SET is_rejected=1 WHERE path='a'")
    db.commit()
    before = snapshot(db)
    with patch.object(manual, 'get_visibility_clause', return_value=("path != ?", ['b'])):
        with pytest.raises(HTTPException) as exc:
            create(['a', 'b', 'c'])
    assert exc.value.status_code == 404
    assert snapshot(db) == before


def test_status_check_is_scoped_to_the_current_user(db):
    db.execute("INSERT INTO user_preferences (user_id, photo_path, is_rejected) VALUES ('alice', 'a', 1)")
    db.commit()
    with patch('api.db_helpers.is_multi_user_enabled', return_value=True), \
            patch('api.db_helpers.get_visibility_clause', return_value=('1=1', [])):
        require_active_culling_photos(db, ['a'], 'bob')
        db.rollback()
        with pytest.raises(HTTPException) as exc:
            require_active_culling_photos(db, ['a'], 'alice')
        assert exc.value.status_code == 409
        db.rollback()


def test_status_check_holds_write_lock_until_confirmation_commits(db):
    require_active_culling_photos(db, ['a'], None)
    path = db.execute('PRAGMA database_list').fetchone()[2]
    other = sqlite3.connect(path, timeout=0)
    try:
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            other.execute("UPDATE photos SET is_rejected=1 WHERE path='a'")
        db.rollback()
        other.execute("UPDATE photos SET is_rejected=1 WHERE path='a'")
        other.commit()
    finally:
        other.close()
