"""统一编号、迁移、优先级和人工分组保护的实际数据库验证。"""
import sqlite3
from unittest.mock import patch
from types import SimpleNamespace
import pytest

from db.schema import init_database
from db.photo_groups import assign_auto_groups, rebuild_similar_groups
from api import unified_culling as unified
from api.routers import manual_culling as manual
from api.routers.burst_culling import CullingConfirmBody, GroupTypeBody


@pytest.fixture
def conn(tmp_path):
    p=str(tmp_path/'unified.db');init_database(p)
    c=sqlite3.connect(p);c.row_factory=sqlite3.Row
    for p in 'abcdefghij':
        c.execute('INSERT INTO photos(path,filename,star_rating,is_favorite,is_rejected) VALUES (?,?,4,1,0)',(p,p))
    c.commit()
    with patch('api.db_helpers.get_visibility_clause',return_value=('1=1',[])), \
         patch('api.unified_culling.get_visibility_clause',return_value=('1=1',[])), \
         patch('api.routers.burst_culling.trigger_auto_retrain'):
        yield c
    c.close()


def test_one_stored_id_with_read_only_legacy_aliases(conn):
    columns={r[1]:r[6] for r in conn.execute('PRAGMA table_xinfo(photos)')}
    assert columns['group_id']==0
    assert columns['burst_group_id']==columns['sequence_group_id']==2
    with pytest.raises(sqlite3.OperationalError,match='generated'):
        conn.execute("UPDATE photos SET sequence_group_id=3 WHERE path='a'")


def test_detection_priority_and_manual_groups_cannot_be_overwritten(conn):
    conn.execute("UPDATE photos SET group_id=-1,group_type='manual_similar',group_source='manual' WHERE path IN ('a','b')")
    for kind,paths in [('similar','abcdefghij'),('burst','abcdefgh'),('bracket','abcdef'),('panorama','abcd'),('hdr_panorama','abcd')]:
        assign_auto_groups(conn,[{'kind':kind,'paths':list(paths)}],[kind])
    rows={r['path']:r for r in conn.execute('SELECT * FROM photos')}
    assert rows['a']['group_type']==rows['b']['group_type']=='manual_similar'
    assert rows['c']['group_type']==rows['d']['group_type']=='hdr_panorama'
    assert rows['e']['group_type']==rows['f']['group_type']=='bracket'
    assert rows['g']['group_type']==rows['h']['group_type']=='burst'
    assert rows['i']['group_type']==rows['j']['group_type']=='similar'
    ids={rows[p]['group_id'] for p in 'acegi'}
    assert len(ids)==5
    assert all(rows[p]['is_rejected']==0 and rows[p]['star_rating']==4 and rows[p]['is_favorite']==1 for p in rows)


def test_changing_panorama_to_burst_keeps_members_and_blocks_redetection(conn):
    assigned=assign_auto_groups(conn,[{'kind':'panorama','paths':['a','b','c']}],['panorama']);gid=assigned[0]['group_id']
    conn.commit()
    unified.change_group_type(conn,GroupTypeBody(group_id=gid,type='burst',paths=['a','b','c']),None)
    assign_auto_groups(conn,[{'kind':'hdr_panorama','paths':['a','b','c','d','e']}],['hdr_panorama'])
    assert [tuple(r) for r in conn.execute("SELECT group_id,group_type,group_source FROM photos WHERE path IN ('a','b','c')")] == [(gid,'burst','manual')]*3


def test_list_has_no_overlapping_members_and_scenes_are_not_a_type(conn):
    assign_auto_groups(conn,[{'kind':'burst','paths':['a','b','c']}],['burst'])
    assign_auto_groups(conn,[{'kind':'similar','paths':['c','d','e']}],['similar'])
    conn.commit()
    groups=unified.fetch_groups(conn,None)
    paths=[p['path'] for g in groups for p in g['photos']]
    assert len(paths)==len(set(paths))==5
    assert {g['type'] for g in groups}=={'burst','similar'}
    assert {g['group_id'] for g in unified.fetch_groups(conn,None,'scene')}=={g['group_id'] for g in groups}


def test_confirm_marks_all_views_done_and_never_restores_rejections(conn):
    gid=assign_auto_groups(conn,[{'kind':'burst','paths':['a','b','c']}],['burst'])[0]['group_id'];conn.commit()
    body=CullingConfirmBody(group_id=gid,type='burst',paths=['a','b','c'],keep_paths=['a'])
    result,_=unified.confirm_group(conn,body,None)
    assert result['rejected']==2
    assert unified.fetch_groups(conn,None)==[]
    before=[tuple(r) for r in conn.execute('SELECT * FROM photos')]
    with pytest.raises(Exception):
        unified.confirm_group(conn,body,None)
    conn.rollback()
    assert [tuple(r) for r in conn.execute('SELECT * FROM photos')]==before


def test_migration_resolves_old_id_collisions_without_changing_personal_attributes(tmp_path):
    p=str(tmp_path/'old.db')
    c=sqlite3.connect(p)
    c.executescript('''CREATE TABLE photos(path TEXT PRIMARY KEY, filename TEXT, burst_group_id INTEGER,
      sequence_group_id INTEGER, sequence_kind TEXT, is_rejected INTEGER,star_rating INTEGER,is_favorite INTEGER);
      INSERT INTO photos VALUES ('a','a',-1,1,'panorama',1,4,1);
      INSERT INTO photos VALUES ('b','b',-1,1,'panorama',0,3,0);
      INSERT INTO photos VALUES ('c','c',1,1,'panorama',0,2,1);
      INSERT INTO photos VALUES ('d','d',1,1,'panorama',1,5,1);
      INSERT INTO photos VALUES ('e','e',1,NULL,NULL,0,4,0);
      INSERT INTO photos VALUES ('f','f',1,NULL,NULL,0,1,0);''')
    c.close();init_database(p);init_database(p)
    c=sqlite3.connect(p)
    rows=c.execute('SELECT path,group_id,group_type,is_rejected,star_rating,is_favorite FROM photos ORDER BY path').fetchall()
    assert rows[0][1:3]==(-1,'manual_similar') and rows[1][1:3]==(-1,'manual_similar')
    assert rows[2][2]==rows[3][2]=='panorama'
    assert rows[4][1:3]==rows[5][1:3]==(1,'burst')
    assert rows[2][1] not in (-1,1)
    assert [(r[0],*r[3:]) for r in rows]==[('a',1,4,1),('b',0,3,0),('c',0,2,1),('d',1,5,1),('e',0,4,0),('f',0,1,0)]
    c.close()


def test_same_auto_members_reuse_id_and_manual_type_change_rejects_stale_members(conn):
    candidate={'kind':'burst','paths':['a','b','c']}
    first=assign_auto_groups(conn,[candidate],['burst'])[0]['group_id'];conn.commit()
    second=assign_auto_groups(conn,[candidate],['burst'])[0]['group_id'];conn.commit()
    assert first==second
    before=[tuple(r) for r in conn.execute('SELECT * FROM photos')]
    with pytest.raises(Exception) as exc:
        unified.change_group_type(conn,GroupTypeBody(group_id=first,type='panorama',paths=['a','b']),None)
    assert exc.value.status_code==409
    conn.rollback()
    assert [tuple(r) for r in conn.execute('SELECT * FROM photos')]==before


def test_auto_cull_uses_unique_ids_and_skips_manual_similar(conn):
    from api.routers import burst_culling as culling
    conn.execute("UPDATE photos SET group_id=-1,group_type='manual_similar',group_source='manual' WHERE path IN ('a','b')")
    assign_auto_groups(conn,[{'kind':'burst','paths':['c','d']}],['burst']);conn.commit()
    with patch.object(culling,'get_visibility_clause',return_value=('1=1',[])):
        groups=culling._collect_auto_cull_groups(conn,None,'all',None,None,None)
    assert len(groups)==1 and groups[0]['type']=='burst'
    assert {p['path'] for p in groups[0]['photos']}=={'c','d'}


def test_group_type_change_requires_full_visibility(conn):
    gid=assign_auto_groups(conn,[{'kind':'panorama','paths':['a','b','c']}],['panorama'])[0]['group_id'];conn.commit()
    before=[tuple(r) for r in conn.execute('SELECT * FROM photos')]
    with patch.object(unified,'get_visibility_clause',return_value=("path!='c'",[])):
        with pytest.raises(Exception) as exc:
            unified.change_group_type(conn,GroupTypeBody(group_id=gid,type='burst',paths=['a','b','c']),None)
    assert exc.value.status_code==404
    conn.rollback()
    assert [tuple(r) for r in conn.execute('SELECT * FROM photos')]==before


def test_similarity_rebuild_respects_existing_memberships_and_processing_limit(conn):
    import struct
    conn.execute("UPDATE photos SET group_id=-1,group_type='manual_similar',group_source='manual' WHERE path IN ('a','b')")
    assign_auto_groups(conn,[{'kind':'burst','paths':['c','d']}],['burst'])
    conn.execute("UPDATE photos SET clip_embedding=? WHERE path IN ('a','b','c','d','e','f','g')",(struct.pack('2f',1.,0.),))
    rebuild_similar_groups(conn,max_photos=2)
    rows={r['path']:r['group_type'] for r in conn.execute('SELECT path,group_type FROM photos')}
    assert rows['a']==rows['b']=='manual_similar'
    assert rows['c']==rows['d']=='burst'
    assert rows['e']==rows['f']=='similar'
    assert rows['g'] is None
