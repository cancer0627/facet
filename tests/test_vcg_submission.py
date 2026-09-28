"""VCG submission scoring, persistence and gallery ordering."""

import sqlite3
from contextlib import asynccontextmanager
from unittest import mock

import aiosqlite
import numpy as np
from fastapi.testclient import TestClient

from analyzers.vcg_suitability import (
    VCG_SCORE_VERSION,
    calculate_vcg_submission_score,
    score_vcg_suitability,
)
from api import create_app
from api.auth import get_optional_user
from db.schema import init_database
from processing.scorer import Facet, _photos_upsert


def _async_conn_factory(db_path):
    @asynccontextmanager
    async def factory():
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        try:
            yield conn
        finally:
            await conn.close()

    return factory


def _sync_conn_factory(db_path):
    from contextlib import contextmanager

    @contextmanager
    def factory():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return factory


def _gallery_client(db_path):
    app = create_app()
    app.dependency_overrides[get_optional_user] = lambda: None
    return app, TestClient(app)


def test_vcg_score_uses_embedding_and_falls_back_to_aggregate():
    axis = np.array([1.0, 0.0], dtype=np.float32)
    embedding = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    suitability = score_vcg_suitability(embedding, axis)

    assert suitability == 10.0
    assert calculate_vcg_submission_score(8.0, suitability) == 8.7
    assert calculate_vcg_submission_score(8.0, None) == 8.0
    assert score_vcg_suitability(b"invalid", axis) is None


def test_scan_score_payload_contains_vcg_fields(monkeypatch):
    facet = Facet.__new__(Facet)
    monkeypatch.setattr(facet, "_get_vcg_axis", lambda: np.array([1.0, 0.0], dtype=np.float32))
    result = {
        "aggregate": 8.0,
        "clip_embedding": np.array([1.0, 0.0], dtype=np.float32).tobytes(),
    }

    facet.populate_vcg_scores(result)

    assert result["vcg_suitability_score"] == 10.0
    assert result["vcg_submission_score"] == 8.7
    assert result["vcg_score_version"] == VCG_SCORE_VERSION
    assert result["vcg_scored_at"]


def test_existing_database_migrates_vcg_columns_as_null_and_adds_index(tmp_path):
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE photos (path TEXT PRIMARY KEY, aggregate REAL)")
    conn.execute("INSERT INTO photos (path, aggregate) VALUES ('/old.jpg', 7.5)")
    conn.commit()
    conn.close()

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT aggregate, vcg_suitability_score, vcg_submission_score, "
        "vcg_score_version, vcg_scored_at FROM photos WHERE path = '/old.jpg'"
    ).fetchone()
    indexes = {r[1] for r in conn.execute("PRAGMA index_list(photos)")}
    conn.close()
    assert row == (7.5, None, None, None, None)
    assert "idx_photos_vcg_submission_score" in indexes


def test_upsert_without_vcg_columns_does_not_clear_existing_score(tmp_path):
    db_path = str(tmp_path / "preserve.db")
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO photos (path, aggregate, vcg_suitability_score, vcg_submission_score) "
        "VALUES ('/a.jpg', 6.0, 8.0, 6.7)"
    )
    sql = _photos_upsert(
        "INSERT OR REPLACE INTO photos (path, aggregate) VALUES (:path, :aggregate)"
    )
    conn.execute(sql, {"path": "/a.jpg", "aggregate": 7.0})
    row = conn.execute(
        "SELECT aggregate, vcg_suitability_score, vcg_submission_score "
        "FROM photos WHERE path = '/a.jpg'"
    ).fetchone()
    conn.close()
    assert row == (7.0, 8.0, 6.7)


def test_recompute_refreshes_vcg_submission_score(tmp_path):
    db_path = str(tmp_path / "recompute.db")
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO photos (path, aggregate, exposure_score, vcg_suitability_score) "
        "VALUES ('/a.jpg', 6.0, 5.0, 9.0)"
    )
    conn.commit()
    conn.close()

    Facet(db_path=db_path, lightweight=True).update_all_aggregates(use_embeddings=False)

    conn = sqlite3.connect(db_path)
    aggregate, suitability, submission, version, scored_at = conn.execute(
        "SELECT aggregate, vcg_suitability_score, vcg_submission_score, "
        "vcg_score_version, vcg_scored_at FROM photos WHERE path = '/a.jpg'"
    ).fetchone()
    conn.close()
    assert submission == calculate_vcg_submission_score(aggregate, suitability)
    assert version == VCG_SCORE_VERSION
    assert scored_at


def test_gallery_vcg_sort_paginates_and_sinks_nulls_in_both_directions(tmp_path):
    db_path = str(tmp_path / "gallery.db")
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO photos (path, filename, aggregate, vcg_suitability_score, "
        "vcg_submission_score) VALUES (?, ?, 5.0, ?, ?)",
        [
            ("/g/high.jpg", "high.jpg", 8.0, 9.0),
            ("/g/low.jpg", "low.jpg", 3.0, 2.0),
            ("/g/unscored.jpg", "unscored.jpg", None, None),
        ],
    )
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
    conn.close()

    app, client = _gallery_client(db_path)
    with (
        mock.patch("api.routers.gallery.get_db", _sync_conn_factory(db_path)),
        mock.patch("api.routers.gallery.get_async_db", _async_conn_factory(db_path)),
        mock.patch("api.db_helpers._existing_columns_cache", cols),
        mock.patch.dict("api.config._count_cache", {}, clear=True),
    ):
        desc = client.get(
            "/api/photos?sort=vcg_submission_score&sort_direction=DESC&per_page=2&page=1"
        )
        asc = client.get(
            "/api/photos?sort=vcg_submission_score&sort_direction=ASC&per_page=2&page=1"
        )
        asc_page_2 = client.get(
            "/api/photos?sort=vcg_submission_score&sort_direction=ASC&per_page=2&page=2"
        )

    assert desc.status_code == asc.status_code == asc_page_2.status_code == 200
    assert [p["path"] for p in desc.json()["photos"]] == ["/g/high.jpg", "/g/low.jpg"]
    assert [p["path"] for p in asc.json()["photos"]] == ["/g/low.jpg", "/g/high.jpg"]
    assert [p["path"] for p in asc_page_2.json()["photos"]] == ["/g/unscored.jpg"]
    assert desc.json()["photos"][0]["vcg_suitability_score"] == 8.0
    assert desc.json()["photos"][0]["vcg_submission_score"] == 9.0


def test_regular_album_vcg_ascending_sort_sinks_nulls(tmp_path):
    db_path = str(tmp_path / "album.db")
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO albums (id, name) VALUES (1, 'VCG')")
    conn.executemany(
        "INSERT INTO photos (path, filename, vcg_submission_score) VALUES (?, ?, ?)",
        [
            ("/a/high.jpg", "high.jpg", 9.0),
            ("/a/low.jpg", "low.jpg", 2.0),
            ("/a/null.jpg", "null.jpg", None),
        ],
    )
    conn.executemany(
        "INSERT INTO album_photos (album_id, photo_path, position) VALUES (1, ?, ?)",
        [("/a/high.jpg", 0), ("/a/null.jpg", 1), ("/a/low.jpg", 2)],
    )
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
    conn.close()

    app, client = _gallery_client(db_path)
    with (
        mock.patch("api.routers.albums.get_async_db", _async_conn_factory(db_path)),
        mock.patch("api.db_helpers._existing_columns_cache", cols),
    ):
        response = client.get(
            "/api/albums/1/photos?sort=vcg_submission_score&sort_direction=ASC"
        )

    assert response.status_code == 200
    assert [p["path"] for p in response.json()["photos"]] == [
        "/a/low.jpg", "/a/high.jpg", "/a/null.jpg"
    ]
