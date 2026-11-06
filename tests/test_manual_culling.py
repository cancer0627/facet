"""Manual sets start fresh even when their photos were reviewed in older sets."""
import asyncio
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from db.schema import init_database
from api.routers import manual_culling as manual
from api.routers.burst_culling import CullingConfirmBody


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / 'manual.db')
    init_database(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    for p in ['a', 'b', 'c', 'd']:
        conn.execute('INSERT INTO photos (path, filename, burst_reviewed, similarity_reviewed, '
                     'is_rejected, star_rating, is_favorite) VALUES (?, ?, 1, 1, 0, 4, 1)', [p, p])
    conn.execute("UPDATE photos SET is_rejected = 1 WHERE path='d'")
    conn.commit()

    @contextmanager
    def context():
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise

    with patch.object(manual, 'get_db', context), \
            patch.object(manual, 'get_visibility_clause', return_value=('1=1', [])), \
            patch('api.db_helpers.is_multi_user_enabled', return_value=False), \
            patch.object(manual, 'trigger_auto_retrain'):
        yield conn
    conn.close()


def create(paths=None, user=None):
    return manual.save_manual_group(manual.ManualGroupBody(paths=paths or ['a', 'b']), user)


def body(group, paths=None, keep=None):
    return CullingConfirmBody(group_id=group['group_id'], type='manual',
                             paths=paths or ['a', 'b'], keep_paths=['a'] if keep is None else keep)


def test_reviewed_photos_enter_manual_group_without_losing_ratings(db):
    group = create()
    feed = manual.fetch_manual_groups(db, None)
    assert feed[0]['group_id'] == group['group_id'] < 0
    assert {p['path'] for p in feed[0]['photos']} == {'a', 'b'}
    assert 'keep_paths' not in feed[0]
    assert dict(db.execute("SELECT star_rating, is_favorite, burst_reviewed, similarity_reviewed, "
                           "is_rejected FROM photos WHERE path='a'").fetchone()) == {
        'star_rating': 4, 'is_favorite': 1, 'burst_reviewed': 0, 'similarity_reviewed': 0, 'is_rejected': 0}


def test_new_group_does_not_inherit_old_confirmation_or_keep_choices(db):
    old = create()
    manual.confirm_manual_group(body(old), None)
    assert manual.fetch_manual_groups(db, None) == []
    new = create(['a', 'b', 'c'])
    assert new['group_id'] != old['group_id']
    feed = manual.fetch_manual_groups(db, None)
    assert len(feed) == 1
    assert feed[0]['group_id'] == new['group_id']
    assert {p['path'] for p in feed[0]['photos']} == {'a', 'b', 'c'}
    assert 'keep_paths' not in feed[0]
    manual.confirm_manual_group(body(new, ['a', 'b', 'c'], ['b']), None)
    assert dict(db.execute('SELECT path, is_rejected FROM photos').fetchall()) == {
        'a': 1, 'b': 0, 'c': 1, 'd': 1}


def test_duplicate_paths_do_not_create_duplicate_members(db):
    group = create(['a', 'a', 'b'])
    assert group['count'] == 2
    with pytest.raises(HTTPException) as exc:
        create(['a', 'a'])
    assert exc.value.status_code == 400
    assert db.execute('SELECT COUNT(DISTINCT burst_group_id) FROM photos WHERE burst_group_id < 0').fetchone()[0] == 1


def test_stale_member_snapshot_rejected_without_changing_any_photo(db):
    group = create()
    with pytest.raises(HTTPException) as exc:
        manual.confirm_manual_group(body(group, paths=['a', 'b', 'c']), None)
    assert exc.value.status_code == 409
    assert db.execute('SELECT SUM(is_rejected) FROM photos').fetchone()[0] == 1


def test_confirmed_group_cannot_be_submitted_again(db):
    group = create()
    manual.confirm_manual_group(body(group), None)
    with pytest.raises(HTTPException) as exc:
        manual.confirm_manual_group(body(group, keep=['b']), None)
    assert exc.value.status_code == 409
    assert dict(db.execute('SELECT path, is_rejected FROM photos').fetchall())['a'] == 0


def test_keep_paths_outside_group_rejected(db):
    group = create()
    with pytest.raises(HTTPException) as exc:
        manual.confirm_manual_group(body(group, keep=['d']), None)
    assert exc.value.status_code == 400


def test_hidden_or_missing_photo_makes_entire_write_fail(db):
    with patch.object(manual, 'get_visibility_clause', return_value=("photos.path != ?", ['b'])):
        with pytest.raises(HTTPException) as exc:
            create()
    assert exc.value.status_code == 404
    assert db.execute('SELECT COUNT(*) FROM photos WHERE burst_group_id < 0').fetchone()[0] == 0


def test_partial_visibility_cannot_confirm_or_read(db):
    group = create()
    with patch.object(manual, 'get_visibility_clause', return_value=("photos.path != ?", ['b'])):
        assert manual.fetch_manual_groups(db, None) == []
        with pytest.raises(HTTPException) as exc:
            manual.confirm_manual_group(body(group), None)
        assert exc.value.status_code == 404


def test_regrouping_restores_only_current_users_selected_rejections(db):
    for user_id in ('alice', 'bob'):
        db.executemany('INSERT INTO user_preferences (user_id, photo_path, is_rejected) '
                       'VALUES (?, ?, 1)', [(user_id, p) for p in ('a', 'b')])
    db.execute("INSERT INTO user_preferences (user_id, photo_path, is_rejected) VALUES ('alice', 'c', 1)")
    db.commit()
    with patch.object(manual, 'is_multi_user_enabled', return_value=True):
        create(user=SimpleNamespace(user_id='alice'))
    prefs = {(r[0], r[1]): r[2] for r in db.execute(
        'SELECT user_id, photo_path, is_rejected FROM user_preferences')}
    assert prefs == {('alice', 'a'): 0, ('alice', 'b'): 0, ('alice', 'c'): 1,
                     ('bob', 'a'): 1, ('bob', 'b'): 1}


def test_group_scope_never_truncates_members(db):
    create()
    db.execute("UPDATE photos SET date_taken='2026:11:01 12:00:00' WHERE path='a'")
    db.execute("UPDATE photos SET date_taken='2026:10:01 12:00:00' WHERE path='b'")
    assert manual.fetch_manual_groups(db, None, date_from='2026:11:01 00:00:00') == []


def test_unified_confirm_dispatches_manual_before_sequence_guard(db):
    from api.routers.burst_culling import _dispatch_culling_confirm
    group = create()
    # sqlite's fixture connection owns this thread.
    async def in_thread(fn, *args):
        return fn(*args)
    with patch('api.routers.burst_culling.asyncio.to_thread', side_effect=in_thread), \
            patch('api.routers.burst_culling.get_db', manual.get_db):
        result = asyncio.run(_dispatch_culling_confirm(body(group), None))
    assert result['kept'] == 1


def test_regroup_moves_members_and_stale_old_page_cannot_confirm_them(db):
    old = create(['a', 'b', 'c'])
    new = create(['a', 'b'])
    assert new['group_id'] != old['group_id']
    assert dict(db.execute('SELECT path, burst_group_id FROM photos')) == {
        'a': new['group_id'], 'b': new['group_id'], 'c': old['group_id'], 'd': None}
    with pytest.raises(HTTPException) as exc:
        manual.confirm_manual_group(body(old, ['a', 'b', 'c']), None)
    assert exc.value.status_code == 409
    assert {g['group_id'] for g in manual.fetch_manual_groups(db, None)} == {new['group_id']}


def test_new_groups_need_no_extra_tables_or_columns(db):
    assert not db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'manual_culling_%'").fetchall()
    create()
    assert not db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'manual_culling_%'").fetchall()
    assert not any('manual' in r[1] for r in db.execute('PRAGMA table_info(photos)'))


def test_manual_members_stay_visible_and_out_of_automatic_burst_feed(db):
    from api.db_helpers import HIDE_BURSTS_SQL, HIDDEN_BURST_SQL
    from api.routers.burst_culling import _query_burst_groups
    create()
    assert db.execute(f'SELECT COUNT(*) FROM photos WHERE burst_group_id < 0 AND {HIDE_BURSTS_SQL}').fetchone()[0] == 2
    assert db.execute(f'SELECT COUNT(*) FROM photos WHERE {HIDDEN_BURST_SQL}').fetchone()[0] == 0
    groups, count, _ = _query_burst_groups(db, '1=1', [])
    assert groups == [] and count == 0


def test_manual_members_stay_out_of_recomputed_similarity_groups(db):
    import struct
    from api.similarity_groups import compute_similarity_groups
    db.execute('UPDATE photos SET clip_embedding = ?, similarity_reviewed = 0, burst_reviewed = 0',
               [struct.pack('2f', 1.0, 0.0)])
    db.commit()
    create()
    from db.photo_groups import rebuild_similar_groups
    rebuild_similar_groups(db)
    with patch('api.db_helpers.get_visibility_clause', return_value=('1=1', [])):
        groups = compute_similarity_groups(db, threshold=0.9, min_size=2)
    assert groups
    assert all(not {'a', 'b'}.intersection(g['paths']) for g in groups)


def test_automatic_burst_detection_and_rescoring_preserve_manual_groups(db, tmp_path):
    from config_resolve import defaults_path
    from processing.scorer import process_bursts, _photos_upsert
    group = create()
    db.execute("UPDATE photos SET phash='ffff0000ffff0000', date_taken='2026:11:03 12:00:00'")
    db.commit()
    path = str(db.execute('PRAGMA database_list').fetchone()[2])
    process_bursts(path, config_path=defaults_path())
    state = [tuple(r) for r in db.execute('SELECT burst_group_id, burst_reviewed, is_burst_lead '
                                        "FROM photos WHERE path IN ('a', 'b')")]
    assert state == [(group['group_id'], 0, 0)] * 2
    db.execute(_photos_upsert('INSERT OR REPLACE INTO photos (path, filename, aggregate) VALUES (?, ?, ?)'),
               ['a', 'rescored.jpg', 9.0])
    assert db.execute("SELECT burst_group_id FROM photos WHERE path='a'").fetchone()[0] == group['group_id']


@pytest.mark.parametrize('remaining', [0, 1, 2])
def test_default_feed_excludes_rejected_and_hides_finished_groups(db, remaining):
    group = create(['a', 'b', 'c'])
    for p in ['a', 'b', 'c'][remaining:]:
        db.execute('UPDATE photos SET is_rejected = 1 WHERE path = ?', [p])
    db.commit()
    before = [tuple(r) for r in db.execute('SELECT * FROM photos')]
    feed = manual.fetch_manual_groups(db, None)
    if remaining < 2:
        assert feed == []
    else:
        assert {p['path'] for p in feed[0]['photos']} == {'a', 'b'}
        assert set(feed[0]['member_paths']) == {'a', 'b', 'c'}
    assert [tuple(r) for r in db.execute('SELECT * FROM photos')] == before
    # 显式查看已淘汰照片时仍可复核，不因读取清空分组。
    assert manual.fetch_manual_groups(db, None, exclude_rejected=False)[0]['group_id'] == group['group_id']


def test_confirm_filtered_manual_group_preserves_hidden_rejections(db):
    group = create(['a', 'b', 'c'])
    db.execute("UPDATE photos SET is_rejected = 1 WHERE path='c'")
    db.commit()
    with patch.object(manual, 'record_culling_decision', return_value=0) as record:
        result = manual.confirm_manual_group(body(group, ['a', 'b'], ['a']), None)
    assert result == {'status': 'ok', 'kept': 1, 'rejected': 1}
    assert dict(db.execute('SELECT path, is_rejected FROM photos')) == {'a': 0, 'b': 1, 'c': 1, 'd': 1}
    assert record.call_args.args[2] == {'b'}
    assert manual.fetch_manual_groups(db, None) == []


def test_manual_filter_uses_current_users_rejection(db):
    create(['a', 'b', 'c'])
    for user_id, rejected in [('alice', ('b', 'c')), ('bob', ('c',))]:
        db.executemany('INSERT INTO user_preferences (user_id, photo_path, is_rejected) VALUES (?, ?, 1)',
                       [(user_id, p) for p in rejected])
    db.commit()
    with patch('api.routers.burst_culling.is_multi_user_enabled', return_value=True), \
            patch('api.db_helpers.is_multi_user_enabled', return_value=True):
        assert manual.fetch_manual_groups(db, SimpleNamespace(user_id='alice')) == []
        feed = manual.fetch_manual_groups(db, SimpleNamespace(user_id='bob'))
        assert {p['path'] for p in feed[0]['photos']} == {'a', 'b'}
