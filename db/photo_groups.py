"""统一选片归属：每张照片只保存一个组编号；旧字段仅作只读兼容。"""
import json
from collections import defaultdict

AUTO_PRIORITY = {'hdr_panorama': 0, 'panorama': 1, 'bracket': 2, 'burst': 3, 'similar': 4}
GROUP_TYPES = ('manual_similar', *AUTO_PRIORITY)
SEQUENCE_TYPES = ('hdr_panorama', 'panorama', 'bracket')
LEGACY_COLUMNS = {
    'burst_group_id': "INTEGER GENERATED ALWAYS AS (CASE WHEN group_type IN ('burst','manual_similar') THEN group_id END) VIRTUAL",
    'sequence_group_id': "INTEGER GENERATED ALWAYS AS (CASE WHEN group_type IN ('hdr_panorama','panorama','bracket') THEN group_id END) VIRTUAL",
}


def next_group_id(conn):
    return max(0, conn.execute('SELECT COALESCE(MAX(group_id), 0) FROM photos').fetchone()[0]) + 1


def invalidate_groups(conn):
    conn.execute("DELETE FROM stats_cache WHERE key LIKE 'similarity_groups_%' OR key LIKE 'scenes_%'")


def assign_auto_groups(conn, candidates, kinds):
    """按优先级写入检测结果；人工分组及更高优先级的归属不受影响。"""
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    kinds = set(kinds)
    current = {r[0]: dict(zip(('path','group_id','group_type','group_source'),r)) for r in conn.execute(
        'SELECT path, group_id, group_type, group_source FROM photos')}
    old = defaultdict(set)
    for row in current.values():
        if row['group_id'] is not None and row['group_type'] in kinds and row['group_source'] != 'manual':
            old[(row['group_type'], row['group_id'])].add(row['path'])
    stable = {(kind, frozenset(paths)): gid for (kind, gid), paths in old.items()}
    next_id = next_group_id(conn)
    conn.execute('SAVEPOINT assign_photo_groups')
    try:
        for kind in kinds:
            conn.execute("UPDATE photos SET group_id=NULL, group_type=NULL, group_source=NULL, "
                         "sequence_kind=NULL, sequence_ev_offset=NULL, is_sequence_lead=0 "
                         "WHERE group_type=? AND COALESCE(group_source,'auto') != 'manual'", (kind,))
        assigned = []
        claimed = set()
        for candidate in sorted(candidates, key=lambda c: AUTO_PRIORITY[c['kind']]):
            kind = candidate['kind']
            paths = []
            for path in dict.fromkeys(candidate['paths']):
                row = current.get(path)
                if row is None or path in claimed or row['group_source'] == 'manual':
                    continue
                previous = row['group_type']
                if previous not in kinds and previous in AUTO_PRIORITY and AUTO_PRIORITY[previous] < AUTO_PRIORITY[kind]:
                    continue
                paths.append(path)
            if len(paths) < 2:
                continue
            gid = stable.get((kind, frozenset(paths)))
            if gid is None:
                gid, next_id = next_id, next_id + 1
            lead = candidate.get('lead') if candidate.get('lead') in paths else paths[len(paths)//2]
            offsets = candidate.get('offsets', {})
            conn.executemany(
                'UPDATE photos SET group_id=?, group_type=?, group_source=\'auto\', '
                'sequence_kind=?, sequence_ev_offset=?, is_sequence_lead=?, is_burst_lead=? WHERE path=?',
                [(gid, kind, kind if kind in SEQUENCE_TYPES else None, offsets.get(path),
                  int(path == lead and kind in SEQUENCE_TYPES), int(path == lead), path) for path in paths])
            claimed.update(paths)
            assigned.append({'group_id': gid, 'kind': kind, 'paths': paths, 'lead': lead})
        # 较高优先级取走成员后，自动组不足两张则解除残余关系；人工组不作隐式修改。
        conn.execute("UPDATE photos SET group_id=NULL, group_type=NULL, group_source=NULL "
                     "WHERE COALESCE(group_source,'auto') != 'manual' AND group_id IN "
                     '(SELECT group_id FROM photos WHERE group_id IS NOT NULL GROUP BY group_id HAVING COUNT(*)<2)')
        invalidate_groups(conn)
        conn.execute('RELEASE assign_photo_groups')
        return assigned
    except BaseException:
        conn.execute('ROLLBACK TO assign_photo_groups')
        conn.execute('RELEASE assign_photo_groups')
        raise


def migrate_photo_groups(conn):
    """将旧连拍／手动／序列编号合并为全库唯一编号，不改照片个人属性。"""
    columns = {r[1]: r for r in conn.execute('PRAGMA table_xinfo(photos)')}
    if all(name in columns and columns[name][6] for name in LEGACY_COLUMNS):
        return
    conn.execute('SAVEPOINT unified_photo_groups')
    try:
        for name, definition in [('group_id', 'INTEGER'), ('group_type', 'TEXT'), ('group_source', 'TEXT')]:
            if name not in columns:
                conn.execute(f'ALTER TABLE photos ADD COLUMN {name} {definition}')
        rows = conn.execute('SELECT path, burst_group_id, sequence_group_id, sequence_kind FROM photos').fetchall()
        groups = defaultdict(list)
        for path, burst, sequence, kind in rows:
            if burst is not None and burst < 0:
                groups[('manual_similar', burst)].append(path)
            elif kind in SEQUENCE_TYPES and sequence is not None:
                groups[(kind, sequence)].append(path)
            elif burst is not None:
                groups[('burst', burst)].append(path)
        forced = {r[0]:r[1] for r in conn.execute("SELECT photo_path,sequence_kind FROM photo_sequence_overrides WHERE sequence_kind IS NOT NULL")} if conn.execute("SELECT 1 FROM sqlite_master WHERE name='photo_sequence_overrides'").fetchone() else {}
        next_id = max([0] + [int(burst) for _, burst, _, _ in rows if burst is not None]) + 1
        for (kind, old_id), paths in groups.items():
            if kind in ('manual_similar', 'burst'):
                gid = old_id
            else:
                gid, next_id = next_id, next_id + 1
            source = 'manual' if kind == 'manual_similar' or any(forced.get(p) == kind for p in paths) else 'auto'
            conn.executemany('UPDATE photos SET group_id=?, group_type=?, group_source=? WHERE path=?',
                             [(gid, kind, source, p) for p in paths])
        # 相似组旧编号为临时编号，仅合并未被更高优先级占用的成员。
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='stats_cache'").fetchone():
            caches = conn.execute("SELECT value FROM stats_cache WHERE key LIKE 'similarity_groups_%_None_None_None_10k' ORDER BY updated_at DESC").fetchall()
            available = {r[0] for r in conn.execute('SELECT path FROM photos WHERE group_id IS NULL')}
            for (value,) in caches:
                try:
                    candidates = json.loads(value)
                except (ValueError, TypeError):
                    continue
                for candidate in candidates:
                    paths = set(candidate.get('paths', [])) & available
                    if len(paths) < 2:
                        continue
                    conn.executemany("UPDATE photos SET group_id=?,group_type='similar',group_source='auto' WHERE path=?", [(next_id,p) for p in paths])
                    next_id += 1
                    available -= paths
        # 先移除引用旧列的索引，再将旧编号替换为虚拟兼容列；不保存第二份编号。
        for name, sql in conn.execute("SELECT name,sql FROM sqlite_master WHERE type='index' AND tbl_name='photos'").fetchall():
            if sql and any(legacy in sql for legacy in LEGACY_COLUMNS):
                conn.execute('DROP INDEX "' + name.replace('"','""') + '"')
        for name, definition in LEGACY_COLUMNS.items():
            if name in columns and columns[name][6]:
                continue
            if name in columns and not columns[name][6]:
                conn.execute(f'ALTER TABLE photos DROP COLUMN {name}')
            conn.execute(f'ALTER TABLE photos ADD COLUMN {name} {definition}')
        # 被手动组接管的照片不再同时带序列分组信息。
        conn.execute("UPDATE photos SET sequence_kind=NULL WHERE group_type NOT IN ('hdr_panorama','panorama','bracket') OR group_type IS NULL")
        conn.execute('UPDATE photos SET burst_reviewed=MAX(COALESCE(burst_reviewed,0),COALESCE(similarity_reviewed,0)), '
                     'similarity_reviewed=MAX(COALESCE(burst_reviewed,0),COALESCE(similarity_reviewed,0))')
        invalidate_groups(conn)
        conn.execute('RELEASE unified_photo_groups')
    except BaseException:
        conn.execute('ROLLBACK TO unified_photo_groups')
        conn.execute('RELEASE unified_photo_groups')
        raise


def rebuild_similar_groups(conn, threshold=0.85, max_group_size=50, max_photos=10000):
    """仅对没有更高优先级归属的照片形成相似组，编号持久保存。"""
    import numpy as np
    from utils.embedding import bytes_to_normalized_embedding, filter_uniform_embeddings
    from utils.union_find import UnionFind
    rows = conn.execute("SELECT path,clip_embedding FROM photos WHERE clip_embedding IS NOT NULL "
                        "AND (group_id IS NULL OR (group_type='similar' AND COALESCE(group_source,'auto')!='manual')) "
                        'ORDER BY date_taken DESC,path LIMIT ?', (max_photos,)).fetchall()
    embeddings, valid = [], []
    for i, row in enumerate(rows):
        embedding = bytes_to_normalized_embedding(row[1])
        if embedding is not None:
            embeddings.append(embedding)
            valid.append(i)
    embeddings, valid = filter_uniform_embeddings(embeddings, valid)
    candidates = []
    if len(embeddings) >= 2:
        matrix = np.stack(embeddings)
        uf = UnionFind(len(matrix))
        for start in range(0, len(matrix), 500):
            scores = matrix[start:start+500] @ matrix.T
            for offset, score in enumerate(scores):
                i = start + offset
                for j in np.flatnonzero(score[i+1:] >= threshold) + i + 1:
                    uf.union(i, int(j))
        groups = defaultdict(list)
        for i, row_index in enumerate(valid):
            groups[uf.find(i)].append(rows[row_index][0])
        candidates = [{'kind':'similar','paths': paths} for paths in groups.values() if 2 <= len(paths) <= max_group_size]
    return assign_auto_groups(conn, candidates, ['similar'])
