"""Tests for the critique API router (api/routers/critique.py).

GET /api/critique is async (Topic 2 step 7), so endpoint-level tests patch
get_async_db with a real aiosqlite-backed temp DB rather than a MagicMock
(a MagicMock is not awaitable). VLM inference (_get_vlm_critique) is patched
with a plain stub since it is wrapped in asyncio.to_thread in the handler.
"""

import json
import sqlite3
from contextlib import ExitStack, asynccontextmanager, contextmanager
from unittest import mock

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from api import create_app
from api.auth import CurrentUser, get_optional_user


# Columns selected by the critique endpoint (must all exist in the test DB).
_CRITIQUE_COLS = [
    'path', 'category', 'aggregate', 'aesthetic', 'quality_score', 'tech_sharpness',
    'face_quality', 'eye_sharpness', 'face_sharpness', 'comp_score',
    'exposure_score', 'color_score', 'contrast_score', 'isolation_bonus',
    'noise_sigma', 'dynamic_range_stops', 'leading_lines_score',
    'power_point_score', 'aesthetic_iaa', 'face_quality_iqa', 'liqe_score',
    'subject_sharpness', 'subject_prominence', 'subject_placement',
    'bg_separation', 'mean_saturation', 'mean_luminance',
    'form_symmetry', 'form_balance', 'form_edge_entropy',
    'form_fractal', 'color_harmony',
    'distortion_attributes', 'skin_tone_delta', 'skin_tone_cast',
    'face_ratio', 'face_count', 'is_monochrome', 'is_blink',
    'is_silhouette', 'is_group_portrait',
    'highlight_clipped', 'shadow_clipped', 'tags', 'shutter_speed',
    'focal_length', 'f_stop', 'iso',
]

_PERSONALIZED_COLS = [
    *_CRITIQUE_COLS,
    'vcg_suitability_score', 'vcg_submission_score', 'vcg_score_version',
    'config_version', 'scanned_at', 'vcg_scored_at',
]

_CRITIQUE_SCHEMA = (
    "CREATE TABLE photos (path TEXT PRIMARY KEY, "
    + ", ".join(f"{c} TEXT" for c in _PERSONALIZED_COLS if c != 'path')
    + ", is_rejected INTEGER DEFAULT 0"
    + ", thumbnail BLOB, vlm_critique TEXT, vlm_critique_language TEXT, vlm_critique_translated TEXT, personalized_suggestions TEXT);"
)

_VLM_TEST_COLS = set(_PERSONALIZED_COLS) | {
    'thumbnail', 'vlm_critique', 'vlm_critique_language', 'vlm_critique_translated',
    'personalized_suggestions',
}


def _make_db(path, photos):
    conn = sqlite3.connect(path)
    conn.executescript(_CRITIQUE_SCHEMA)
    for p in photos:
        cols = list(p.keys())
        placeholders = ", ".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO photos ({', '.join(cols)}) VALUES ({placeholders})",
            [p[c] for c in cols],
        )
    conn.commit()
    conn.close()


def _async_conn_factory(db_path):
    @asynccontextmanager
    async def factory():
        c = await aiosqlite.connect(db_path)
        c.row_factory = aiosqlite.Row
        try:
            yield c
        finally:
            await c.close()
    return factory


def _fake_vis(user_id, table_alias=None):
    return ("1=1", [])


def _make_photo(**overrides):
    """Return a photo dict with sensible defaults for critique tests."""
    defaults = {
        "path": "/photos/test.jpg",
        "category": "landscape",
        "aggregate": 7.5,
        "vcg_suitability_score": 6.5,
        "vcg_submission_score": 7.15,
        "vcg_score_version": "vcg-stock-v1-65-35",
        "config_version": "test-config",
        "scanned_at": "2026-09-28T00:00:00+00:00",
        "vcg_scored_at": "2026-09-28T00:00:00+00:00",
        "aesthetic": 8.0,
        "quality_score": None,
        "tech_sharpness": 7.0,
        "face_quality": None,
        "eye_sharpness": None,
        "face_sharpness": None,
        "comp_score": 6.5,
        "exposure_score": 7.2,
        "color_score": 6.8,
        "contrast_score": 7.1,
        "isolation_bonus": None,
        "noise_sigma": 2.5,
        "dynamic_range_stops": 8.0,
        "leading_lines_score": 5.0,
        "power_point_score": 4.5,
        "aesthetic_iaa": 6.9,
        "face_quality_iqa": None,
        "liqe_score": 7.3,
        "subject_sharpness": 7.8,
        "subject_prominence": 6.0,
        "subject_placement": 5.5,
        "bg_separation": 6.2,
        "mean_saturation": 0.45,
        "mean_luminance": 0.52,
        "distortion_attributes": None,
        "skin_tone_delta": None,
        "skin_tone_cast": None,
        "face_ratio": None,
        "face_count": 0,
        "is_monochrome": 0,
        "is_blink": 0,
        "highlight_clipped": 0,
        "shadow_clipped": 0,
        "tags": '["landscape", "mountain"]',
        "shutter_speed": None,
    }
    defaults.update(overrides)
    return defaults


def _personalized_payload(photo, lang='zh', aggregate_action='旧综合建议'):
    from api.routers.critique import _PERSONALIZED_SUGGESTIONS_VERSION, _personalized_context_hash

    return {
        'version': _PERSONALIZED_SUGGESTIONS_VERSION,
        'lang': lang,
        'context_hash': _personalized_context_hash(photo, lang),
        'generated_at': '2026-09-28T00:00:00+00:00',
        'aggregate_suggestions': [{'action': aggregate_action, 'reason': '当前画面问题'}],
        'vcg_suggestions': [{'action': '旧申请建议', 'reason': '申请场景需要'}],
    }


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Endpoint-level tests (mock _build_rule_critique for isolation)
# ---------------------------------------------------------------------------


class TestCritiqueEndpoint:
    """Tests for GET /api/critique — endpoint routing and guards."""

    def test_critique_disabled(self, client):
        """When show_critique=False the endpoint returns 403."""
        with mock.patch(
            "api.routers.critique.VIEWER_CONFIG",
            {"features": {"show_critique": False}},
        ):
            resp = client.get("/api/critique", params={"path": "/photos/test.jpg"})

        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"].lower()

    def test_photo_not_found(self, client, tmp_path):
        """Unknown path returns 404."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [])  # empty -> path not found

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
        ):
            resp = client.get("/api/critique", params={"path": "/photos/missing.jpg"})

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


    def test_rule_critique_success(self, client, tmp_path):
        """Rule mode returns breakdown, strengths, weaknesses, and category."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        fake_result = {
            "category": "landscape",
            "category_reason": {"reason_key": "matched", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [
                {"metric": "Aesthetic Quality", "metric_key": "aesthetic", "value": 8.0, "weight": 0.35, "contribution": 2.80},
            ],
            "strengths": [{"metric_key": "aesthetic", "value": 8.0}],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_result),
        ):
            resp = client.get("/api/critique", params={"path": "/photos/test.jpg"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["category"] == "landscape"
        assert body["aggregate"] == 7.5
        assert len(body["breakdown"]) == 1
        assert isinstance(body["strengths"], list)
        assert isinstance(body["weaknesses"], list)


    def test_vlm_mode_unavailable(self, client, tmp_path):
        """When mode=vlm but VLM returns nothing, vlm_available=False in response.

        _get_vlm_critique is the blocking inference call (wrapped in
        asyncio.to_thread by the handler); patch it with a plain stub returning
        None to simulate an unavailable VLM.
        """
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        fake_rule = {
            "category": "landscape",
            "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_rule),
            mock.patch("api.routers.critique._get_vlm_critique", return_value=None),
        ):
            resp = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("vlm_available") is False
        assert "vlm_critique" not in body

    def test_vlm_mode_available(self, client, tmp_path):
        """When mode=vlm and VLM returns text, it appears as vlm_critique."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        fake_rule = {
            "category": "landscape",
            "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_rule),
            mock.patch("api.routers.critique._get_vlm_critique", return_value="A lovely landscape."),
        ):
            resp = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["vlm_critique"] == "A lovely landscape."
        assert body["vlm_source"] == "generated"
        assert "vlm_available" not in body

    def test_vlm_critique_cached_on_second_call(self, client, tmp_path):
        """The generated critique is persisted and reused without a second inference."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        fake_rule = {
            "category": "landscape",
            "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }
        vlm_stub = mock.MagicMock(return_value="Cached critique text.")

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_rule),
            mock.patch("api.routers.critique._get_vlm_critique", vlm_stub),
        ):
            first = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})
            second = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})

        assert first.json()["vlm_source"] == "generated"
        assert second.json()["vlm_critique"] == "Cached critique text."
        assert second.json()["vlm_source"] == "cached"
        assert vlm_stub.call_count == 1

        conn = sqlite3.connect(db)
        stored = conn.execute("SELECT vlm_critique FROM photos WHERE path = '/photos/test.jpg'").fetchone()[0]
        conn.close()
        assert stored == "Cached critique text."

    def test_vlm_refresh_regenerates(self, client, tmp_path):
        """refresh=true bypasses the cache and re-runs inference."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])
        conn = sqlite3.connect(db)
        conn.execute("UPDATE photos SET vlm_critique = 'Stale text.' WHERE path = '/photos/test.jpg'")
        conn.commit()
        conn.close()

        fake_rule = {
            "category": "landscape",
            "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }
        vlm_stub = mock.MagicMock(return_value="Fresh text.")

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_rule),
            mock.patch("api.routers.critique._get_vlm_critique", vlm_stub),
        ):
            resp = client.get(
                "/api/critique",
                params={"path": "/photos/test.jpg", "mode": "vlm", "refresh": "true"},
            )

        assert resp.json()["vlm_critique"] == "Fresh text."
        assert vlm_stub.call_count == 1

    def test_vlm_critique_uses_requested_language_and_persists_it(self, client, tmp_path):
        """The current viewer language reaches generation and the cache records it."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        fake_rule = {
            "category": "landscape",
            "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_rule),
            mock.patch("api.routers.critique._get_vlm_critique", return_value="Un beau paysage.") as vlm_stub,
        ):
            resp = client.get(
                "/api/critique",
                params={"path": "/photos/test.jpg", "mode": "vlm", "lang": "fr"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["vlm_critique"] == "Un beau paysage."
        assert body["vlm_source"] == "generated"
        assert vlm_stub.call_args.args[-1] == "fr"

        conn = sqlite3.connect(db)
        stored = conn.execute(
            "SELECT vlm_critique, vlm_critique_language, vlm_critique_translated "
            "FROM photos WHERE path = '/photos/test.jpg'"
        ).fetchone()
        conn.close()
        assert stored[0] == "Un beau paysage."
        assert stored[1] == "fr"
        assert stored[2] is None

    def test_vlm_critique_regenerates_when_cached_language_differs(self, client, tmp_path):
        """A cached critique in another locale is not returned as the current locale."""
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE photos SET vlm_critique = ?, vlm_critique_language = ? "
            "WHERE path = ?",
            ("English cache.", "en", "/photos/test.jpg"),
        )
        conn.commit()
        conn.close()

        fake_rule = {
            "category": "landscape",
            "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
            "aggregate": 7.5,
            "breakdown": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "penalties": {},
        }
        vlm_stub = mock.MagicMock(return_value="中文点评缓存替换文本。")

        with (
            mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
            mock.patch("api.auth.VIEWER_CONFIG", {"password": "", "edition_password": ""}),
            mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
            mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
            mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
            mock.patch("api.routers.critique._build_rule_critique", return_value=fake_rule),
            mock.patch("api.routers.critique._get_vlm_critique", vlm_stub),
        ):
            resp = client.get(
                "/api/critique",
                params={"path": "/photos/test.jpg", "mode": "vlm", "lang": "zh"},
            )

        assert resp.json()["vlm_critique"] == "中文点评缓存替换文本。"
        assert resp.json()["vlm_source"] == "generated"
        assert vlm_stub.call_args.args[-1] == "zh"


class TestPersonalizedSuggestions:
    """Structured personalized advice is independently cached and refreshed."""

    _RULE = {
        "category": "landscape",
        "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
        "aggregate": 7.5,
        "breakdown": [
            {"metric": "Composition", "metric_key": "comp_score", "value": 6.5,
             "weight": 0.2, "contribution": 1.3},
        ],
        "strengths": [], "weaknesses": [], "suggestions": [], "penalties": {},
    }

    @contextmanager
    def _endpoint_patches(self, db, *, edition=True, show_vlm=True, generated=None):
        generator = mock.MagicMock(return_value=generated)
        with ExitStack() as stack:
            stack.enter_context(mock.patch(
                "api.routers.critique.VIEWER_CONFIG",
                {"features": {"show_critique": True, "show_vlm_critique": show_vlm}},
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique.get_async_db", _async_conn_factory(db),
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique.get_visibility_clause", _fake_vis,
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS,
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique.is_edition_authenticated", return_value=edition,
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique.resolve_personalized_vlm_config",
                return_value={"model_path": "/m"},
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique._build_rule_critique", return_value=dict(self._RULE),
            ))
            stack.enter_context(mock.patch(
                "api.routers.critique._get_personalized_suggestions", generator,
            ))
            yield generator

    def test_prompt_contains_scores_formula_and_safety_limits(self):
        from api.routers.critique import _build_personalized_prompt

        prompt = _build_personalized_prompt(_make_photo(), self._RULE, 'zh').lower()

        assert '7.5/10' in prompt
        assert '7.15/10' in prompt
        assert 'visual china' in prompt
        assert '65% aggregate + 35% suitability' in prompt
        assert 'return only valid json' in prompt
        assert 'aggregate_action' in prompt
        assert 'vcg_action' in prompt
        assert 'at most 3 items' in prompt
        assert 'do not predict any numeric score increase' in prompt
        assert 'do not claim to know official visual china review rules' in prompt

    def test_parser_accepts_plain_and_fenced_json_but_rejects_invalid_output(self):
        from api.routers.critique import _parse_personalized_suggestions

        plain = '{"aggregate_suggestions":[{"action":"Crop","reason":"Remove edge clutter"}],"vcg_suggestions":[]}'
        compact = (
            '{"aggregate_action":"Crop","aggregate_reason":"Remove edge clutter",'
            '"vcg_action":"Reduce highlights","vcg_reason":"Protect bright areas"}'
        )
        fenced = '```json\n' + plain + '\n```'

        assert _parse_personalized_suggestions(plain)['aggregate_suggestions'][0]['action'] == 'Crop'
        assert _parse_personalized_suggestions(fenced)['vcg_suggestions'] == []
        parsed_compact = _parse_personalized_suggestions(compact)
        assert parsed_compact['vcg_suggestions'][0]['action'] == 'Reduce highlights'
        assert _parse_personalized_suggestions('not json') is None
        assert _parse_personalized_suggestions('{"aggregate_suggestions":[]}') is None

    def test_valid_cache_hit_skips_vlm(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        photo = _make_photo()
        _make_db(db, [photo])
        payload = _personalized_payload(photo)
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE photos SET personalized_suggestions = ? WHERE path = ?",
            [json.dumps(payload, ensure_ascii=False), photo['path']],
        )
        conn.commit()
        conn.close()

        with self._endpoint_patches(db, edition=False, generated=None) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": photo['path'], "lang": "zh"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is True
        assert body["source"] == "cached"
        assert body["aggregate_suggestions"][0]["action"] == "旧综合建议"
        generator.assert_not_called()

    def test_missing_cache_generates_and_persists(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        photo = _make_photo()
        _make_db(db, [photo])
        generated = {
            "aggregate_suggestions": [{"action": "拍摄：改变机位", "reason": "主体边缘杂乱"}],
            "vcg_suggestions": [{"action": "后期：压低高光", "reason": "高光区域过亮"}],
        }

        with self._endpoint_patches(db, generated=generated) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": photo['path'], "lang": "zh"},
            )

        assert resp.status_code == 200
        assert resp.json()["source"] == "generated"
        assert resp.json()["aggregate_suggestions"] == generated["aggregate_suggestions"]
        generator.assert_called_once()
        conn = sqlite3.connect(db)
        stored = conn.execute(
            "SELECT personalized_suggestions FROM photos WHERE path = ?", [photo['path']]
        ).fetchone()[0]
        conn.close()
        assert json.loads(stored)["vcg_suggestions"] == generated["vcg_suggestions"]

    def test_refresh_replaces_valid_cache(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        photo = _make_photo()
        _make_db(db, [photo])
        old_payload = _personalized_payload(photo)
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE photos SET personalized_suggestions = ? WHERE path = ?",
            [json.dumps(old_payload), photo['path']],
        )
        conn.commit()
        conn.close()
        generated = {
            "aggregate_suggestions": [{"action": "新综合建议", "reason": "新的综合问题"}],
            "vcg_suggestions": [{"action": "新申请建议", "reason": "新的申请问题"}],
        }

        with self._endpoint_patches(db, generated=generated) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": photo['path'], "lang": "zh", "refresh": "true"},
            )

        assert resp.json()["source"] == "generated"
        assert resp.json()["aggregate_suggestions"] == generated["aggregate_suggestions"]
        generator.assert_called_once()
        conn = sqlite3.connect(db)
        stored = json.loads(conn.execute(
            "SELECT personalized_suggestions FROM photos WHERE path = ?", [photo['path']]
        ).fetchone()[0])
        conn.close()
        assert stored["aggregate_suggestions"] == generated["aggregate_suggestions"]

    def test_score_context_change_regenerates_stale_cache(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        photo = _make_photo()
        _make_db(db, [photo])
        payload = _personalized_payload(photo)
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE photos SET personalized_suggestions = ?, aggregate = ? WHERE path = ?",
            [json.dumps(payload), 6.0, photo['path']],
        )
        conn.commit()
        conn.close()
        generated = {
            "aggregate_suggestions": [{"action": "重新生成", "reason": "评分上下文已变化"}],
            "vcg_suggestions": [],
        }

        with self._endpoint_patches(db, generated=generated) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": photo['path'], "lang": "zh"},
            )

        assert resp.json()["source"] == "generated"
        assert resp.json()["aggregate_score"] == 6.0
        generator.assert_called_once()

    def test_missing_score_is_explicitly_unavailable(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo(aggregate=None)])

        with self._endpoint_patches(db, generated=None) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": "/photos/test.jpg", "lang": "zh"},
            )

        assert resp.json() == {
            "available": False,
            "source": "unavailable",
            "aggregate_score": None,
            "vcg_submission_score": 7.15,
            "aggregate_suggestions": [],
            "vcg_suggestions": [],
            "generated_at": None,
            "reason": "score_unavailable",
            "warning": None,
        }
        generator.assert_not_called()

    def test_no_edition_access_is_explicitly_unavailable(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        with self._endpoint_patches(db, edition=False, generated=None) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": "/photos/test.jpg", "lang": "zh"},
            )

        assert resp.json()["reason"] == "edition_required"
        assert resp.json()["available"] is False
        generator.assert_not_called()

    def test_no_vlm_is_explicitly_unavailable(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])

        with self._endpoint_patches(db, show_vlm=False, generated=None) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": "/photos/test.jpg", "lang": "zh"},
            )

        assert resp.json()["reason"] == "vlm_unavailable"
        assert resp.json()["available"] is False
        generator.assert_not_called()

    def test_refresh_failure_keeps_existing_cache(self, client, tmp_path):
        db = str(tmp_path / "critique.db")
        photo = _make_photo()
        _make_db(db, [photo])
        payload = _personalized_payload(photo)
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE photos SET personalized_suggestions = ? WHERE path = ?",
            [json.dumps(payload), photo['path']],
        )
        conn.commit()
        conn.close()

        with self._endpoint_patches(db, generated=None) as generator:
            resp = client.get(
                "/api/personalized_suggestions",
                params={"path": photo['path'], "lang": "zh", "refresh": "true"},
            )

        body = resp.json()
        assert body["source"] == "cached"
        assert body["warning"] == "refresh_failed"
        assert body["aggregate_suggestions"][0]["action"] == "旧综合建议"
        generator.assert_called_once()
        conn = sqlite3.connect(db)
        stored = json.loads(conn.execute(
            "SELECT personalized_suggestions FROM photos WHERE path = ?", [photo['path']]
        ).fetchone()[0])
        conn.close()
        assert stored["aggregate_suggestions"][0]["action"] == "旧综合建议"


def _client_for(user):
    """Build a (TestClient, app) whose get_optional_user yields ``user``."""
    app = create_app()
    app.dependency_overrides[get_optional_user] = lambda: user
    return TestClient(app), app


class TestCritiqueVlmEditionGate:
    """F5': VLM critique generation is edition-only; cached reads stay open."""

    _RULE = {
        "category": "landscape",
        "category_reason": {"reason_key": "default", "category": "landscape", "details": []},
        "aggregate": 7.5, "breakdown": [], "strengths": [], "weaknesses": [],
        "suggestions": [], "penalties": {},
    }

    def test_generation_denied_for_non_edition(self, tmp_path):
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])
        vlm_stub = mock.MagicMock(return_value="should not run")
        client, app = _client_for(CurrentUser(user_id="u1", role="user"))
        try:
            with (
                mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
                mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
                mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
                mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
                mock.patch("api.routers.critique._build_rule_critique", return_value=dict(self._RULE)),
                mock.patch("api.routers.critique._get_vlm_critique", vlm_stub),
                mock.patch("api.auth.is_multi_user_enabled", return_value=True),
            ):
                resp = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("vlm_available") is False
        assert "vlm_critique" not in body
        vlm_stub.assert_not_called()
        conn = sqlite3.connect(db)
        stored = conn.execute(
            "SELECT vlm_critique FROM photos WHERE path = '/photos/test.jpg'"
        ).fetchone()[0]
        conn.close()
        assert stored is None

    def test_cached_read_open_to_non_edition(self, tmp_path):
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])
        conn = sqlite3.connect(db)
        conn.execute("UPDATE photos SET vlm_critique = 'Cached.' WHERE path = '/photos/test.jpg'")
        conn.commit()
        conn.close()
        vlm_stub = mock.MagicMock(return_value="should not run")
        client, app = _client_for(CurrentUser(user_id="u1", role="user"))
        try:
            with (
                mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
                mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
                mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
                mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
                mock.patch("api.routers.critique._build_rule_critique", return_value=dict(self._RULE)),
                mock.patch("api.routers.critique._get_vlm_critique", vlm_stub),
                mock.patch("api.auth.is_multi_user_enabled", return_value=True),
            ):
                resp = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json()["vlm_critique"] == "Cached."
        vlm_stub.assert_not_called()

    def test_generation_allowed_for_edition_admin(self, tmp_path):
        db = str(tmp_path / "critique.db")
        _make_db(db, [_make_photo()])
        client, app = _client_for(
            CurrentUser(user_id="admin", role="admin", edition_authenticated=True)
        )
        try:
            with (
                mock.patch("api.routers.critique.VIEWER_CONFIG", {"features": {"show_critique": True, "show_vlm_critique": True}}),
                mock.patch("api.routers.critique.get_async_db", _async_conn_factory(db)),
                mock.patch("api.routers.critique.get_visibility_clause", _fake_vis),
                mock.patch("api.routers.critique.get_existing_columns", return_value=_VLM_TEST_COLS),
                mock.patch("api.routers.critique._build_rule_critique", return_value=dict(self._RULE)),
                mock.patch("api.routers.critique._get_vlm_critique", return_value="Fresh critique."),
                mock.patch("api.auth.is_multi_user_enabled", return_value=True),
            ):
                resp = client.get("/api/critique", params={"path": "/photos/test.jpg", "mode": "vlm"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json()["vlm_critique"] == "Fresh critique."


# ---------------------------------------------------------------------------
# Direct _build_rule_critique tests (mock ScoringConfig)
# ---------------------------------------------------------------------------


class TestBuildRuleCritique:
    """Unit tests for _build_rule_critique with mocked ScoringConfig."""

    def _mock_scoring_config(self, weights, category_config=None):
        """Return a mock ScoringConfig class whose instances return *weights*."""
        instance = mock.MagicMock()
        instance.get_weights.return_value = weights
        instance.get_category_config.return_value = category_config or {}
        cls = mock.MagicMock(return_value=instance)
        return cls

    def test_rule_critique_with_penalties(self):
        """Photo with is_blink=1 and high noise_sigma produces penalties."""
        from api.routers.critique import _build_rule_critique

        weights = {
            "aesthetic": 0.35,
            "tech_sharpness": 0.25,
            "composition": 0.20,
            "noise": 0.10,
            "exposure": 0.10,
        }
        photo = _make_photo(
            is_blink=1,
            noise_sigma=10.0,
            aesthetic=8.0,
            tech_sharpness=6.0,
            comp_score=5.5,
            exposure_score=7.0,
        )

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        penalties = result["penalties"]
        assert penalties.get("blink") is True
        assert "noise" in penalties
        assert penalties["noise"] < 0  # negative penalty value

    def test_critique_fields(self):
        """Verify all expected keys are present in the result."""
        from api.routers.critique import _build_rule_critique

        weights = {
            "aesthetic": 0.40,
            "tech_sharpness": 0.30,
            "composition": 0.15,
            "exposure": 0.15,
        }
        photo = _make_photo()

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        expected_keys = {
            "category",
            "category_reason",
            "aggregate",
            "breakdown",
            "strengths",
            "weaknesses",
            "suggestions",
            "penalties",
            "distortions",
        }
        assert expected_keys == set(result.keys())

        # Verify sub-structure types
        assert isinstance(result["breakdown"], list)
        assert isinstance(result["strengths"], list)
        assert isinstance(result["weaknesses"], list)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["penalties"], dict)
        assert isinstance(result["distortions"], list)
        assert isinstance(result["category_reason"], dict)
        assert result["aggregate"] == photo["aggregate"]
        assert result["category"] == photo["category"]

    def test_breakdown_item_structure(self):
        """Each breakdown item has metric, metric_key, value, weight, contribution."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}
        photo = _make_photo(aesthetic=8.5, comp_score=6.0)

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert len(result["breakdown"]) >= 1
        item = result["breakdown"][0]
        assert set(item.keys()) == {"metric", "metric_key", "value", "weight", "contribution"}
        assert isinstance(item["value"], float)
        assert isinstance(item["weight"], float)

    def test_strengths_and_weaknesses_classification(self):
        """High-scoring metrics appear in strengths, low-scoring in weaknesses."""
        from api.routers.critique import _build_rule_critique

        weights = {
            "aesthetic": 0.40,
            "tech_sharpness": 0.30,
            "composition": 0.30,
        }
        photo = _make_photo(aesthetic=9.0, tech_sharpness=3.0, comp_score=9.5)

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        strength_keys = [s["metric_key"] for s in result["strengths"]]
        weakness_keys = [w["metric_key"] for w in result["weaknesses"]]

        assert "aesthetic" in strength_keys
        assert "comp_score" in strength_keys
        assert "tech_sharpness" in weakness_keys

    def test_suggestions_exist_without_old_weakness_threshold(self):
        """A scored photo still gets next actions when no metric passes the old
        value<5 and weight>5% weakness gate."""
        from api.routers.critique import _build_rule_critique

        weights = {
            "aesthetic": 0.58,
            "composition": 0.18,
            "tech_sharpness": 0.04,
            "dynamic_range": 0.02,
        }
        photo = _make_photo(
            aesthetic=6.2,
            comp_score=7.6,
            tech_sharpness=0.4,
            dynamic_range_stops=2.0,
        )

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert result["weaknesses"] == []
        assert result["suggestions"] == ["aesthetic", "comp_score", "tech_sharpness"]

    def test_low_weight_metric_can_receive_suggestion(self):
        """Low-weight metrics are valid advice candidates even though they do
        not qualify as headline weaknesses."""
        from api.routers.critique import _build_rule_critique

        weights = {"tech_sharpness": 0.04}
        photo = _make_photo(tech_sharpness=0.4)

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert result["weaknesses"] == []
        assert result["suggestions"] == ["tech_sharpness"]

    def test_all_weighted_score_metrics_have_suggestion_copy(self):
        """A weighted score must not silently lose its recommendation."""
        from api.routers.critique import SUGGESTIONS, WEIGHT_TO_COLUMN

        excluded = {"mean_saturation"}
        assert set(WEIGHT_TO_COLUMN.values()) - excluded <= set(SUGGESTIONS)

    def test_no_penalties_for_clean_photo(self):
        """A photo with no issues produces an empty penalties dict."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}
        photo = _make_photo(is_blink=0, noise_sigma=1.5, highlight_clipped=0, shadow_clipped=0)

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert result["penalties"] == {}

    def test_distortions_parsed_from_json_column(self):
        """The stored distortion_attributes JSON surfaces as attribute keys."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}
        photo = _make_photo(distortion_attributes=(
            '[{"attribute": "motion_blur", "confidence": 0.91},'
            ' {"attribute": "haze", "confidence": 0.7}]'
        ))

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert result["distortions"] == ["motion_blur", "haze"]

    def test_distortions_empty_and_malformed_json(self):
        """NULL and malformed columns degrade to an empty list, never raise."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            assert _build_rule_critique(_make_photo())["distortions"] == []
            broken = _make_photo(distortion_attributes="not-json")
            assert _build_rule_critique(broken)["distortions"] == []

    def test_skin_tone_cast_surfaces_penalty(self):
        """A stored cast surfaces as an advisory penalty from the stored fields
        alone — the compute-time cast decision is trusted, never re-gated (F20)."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}
        photo = _make_photo(skin_tone_delta=18.0, skin_tone_cast="green")

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert result["penalties"]["skin_tone"] == {"cast": "green", "delta": 18.0}

    def test_skin_tone_cast_surfaces_below_old_threshold(self):
        """A cast set at compute time still surfaces when the delta is below the
        old hardcoded 12.0 gate — proving _check_penalties no longer re-gates."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}
        photo = _make_photo(skin_tone_delta=8.0, skin_tone_cast="magenta")

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert result["penalties"]["skin_tone"] == {"cast": "magenta", "delta": 8.0}

    def test_skin_tone_no_cast_no_penalty(self):
        """A stored delta without a cast stays advisory-silent (the cast column
        is NULL below the compute-time threshold, so nothing is surfaced)."""
        from api.routers.critique import _build_rule_critique

        weights = {"aesthetic": 0.50, "composition": 0.50}
        photo = _make_photo(skin_tone_delta=5.0, skin_tone_cast=None)

        with mock.patch("config.ScoringConfig", self._mock_scoring_config(weights)):
            result = _build_rule_critique(photo)

        assert "skin_tone" not in result["penalties"]


# ---------------------------------------------------------------------------
# VLM config resolution (regression: the critique used to gate on a
# nonexistent models.vlm_tagger block and the raw 'auto' vram_profile,
# leaving mode=vlm permanently unavailable with the shipped config)
# ---------------------------------------------------------------------------


def _models_config(vram_profile, tagging_model="qwen3.5-2b", model_path="/models/qwen"):
    return {
        "models": {
            "vram_profile": vram_profile,
            "profiles": {
                "legacy": {"tagging_model": "clip"},
                "16gb": {"tagging_model": tagging_model},
                "24gb": {"tagging_model": "qwen3.5-4b"},
            },
            "qwen3_5_2b": {"model_path": model_path},
            "qwen3_5_4b": {"model_path": model_path},
        }
    }


class TestResolveVlmConfig:
    """resolve_vlm_config must follow the active profile's tagging model."""

    def test_explicit_vlm_profile_resolves(self):
        from api.model_cache import resolve_vlm_config

        with mock.patch("api.config._FULL_CONFIG", _models_config("16gb")):
            cfg = resolve_vlm_config()

        assert cfg == {"model_path": "/models/qwen"}

    def test_clip_profile_returns_none(self):
        from api.model_cache import resolve_vlm_config

        with mock.patch("api.config._FULL_CONFIG", _models_config("legacy")):
            assert resolve_vlm_config() is None

    def test_missing_model_path_returns_none(self):
        from api.model_cache import resolve_vlm_config

        with mock.patch("api.config._FULL_CONFIG", _models_config("16gb", model_path="")):
            assert resolve_vlm_config() is None

    def test_auto_profile_resolves_via_hardware_detection(self):
        from api.model_cache import resolve_vlm_config

        with (
            mock.patch("api.config._FULL_CONFIG", _models_config("auto")),
            mock.patch("api.model_cache._resolved_profile", None),
            mock.patch("config.ScoringConfig.suggest_vram_profile", return_value=("16gb", 15.8, "detected")),
        ):
            cfg = resolve_vlm_config()

        assert cfg == {"model_path": "/models/qwen"}

    def test_auto_profile_without_gpu_returns_none(self):
        from api.model_cache import resolve_vlm_config

        with (
            mock.patch("api.config._FULL_CONFIG", _models_config("auto")),
            mock.patch("api.model_cache._resolved_profile", None),
            mock.patch("config.ScoringConfig.suggest_vram_profile", return_value=("legacy", None, "no gpu")),
        ):
            assert resolve_vlm_config() is None


class TestBuildVlmPrompt:
    """The prompt template is filled with the full breakdown, penalties, and EXIF."""

    def test_prompt_injects_breakdown_penalties_and_exif(self):
        from api.routers.critique import _build_vlm_prompt

        rule = {
            "category": "portrait",
            "aggregate": 6.4,
            "breakdown": [
                {"metric": "Aesthetic Quality", "metric_key": "aesthetic", "value": 7.1, "weight": 0.3, "contribution": 2.1},
                {"metric": "Noise Level", "metric_key": "noise_sigma", "value": 2.5, "weight": 0.1, "contribution": 0.3},
            ],
            "penalties": {"blink": True},
        }
        photo = _make_photo(f_stop=2.8, shutter_speed="1/250", iso=400, focal_length=85)

        prompt = _build_vlm_prompt(rule, photo)

        assert "portrait" in prompt
        assert "6.4/10" in prompt
        assert "- Aesthetic Quality: 7.1 (weight 30%)" in prompt
        assert "- Noise Level: 2.5 (weight 10%, lower is better)" in prompt
        assert "Penalties applied: blink." in prompt
        assert "f/2.8, 1/250s, ISO 400, 85mm" in prompt

        zh_prompt = _build_vlm_prompt(rule, photo, "zh")
        assert "Respond entirely in Simplified Chinese (zh)" in zh_prompt

    def test_prompt_handles_missing_data(self):
        from api.routers.critique import _build_vlm_prompt

        rule = {"category": "", "aggregate": None, "breakdown": [], "penalties": {}}
        photo = _make_photo(f_stop=None, shutter_speed=None, iso=None, focal_length=None)

        prompt = _build_vlm_prompt(rule, photo)

        assert "no per-metric data available" in prompt
        assert "Camera settings: unknown." in prompt

    def test_default_prompt_is_laddered_and_multidimensional(self):
        """The shipped default prompt asks for the AesBench four-ability ladder,
        keeps the 3-section wire format, covers the four aesthetic dimensions and
        instructs the model to reconcile the pixels with the injected metrics."""
        from api.routers.critique import _build_vlm_prompt

        rule = {
            "category": "portrait",
            "aggregate": 6.4,
            "breakdown": [
                {"metric": "Composition", "metric_key": "comp_score", "value": 5.0, "weight": 0.2, "contribution": 1.0},
            ],
            "penalties": {},
        }
        prompt = _build_vlm_prompt(rule, _make_photo()).lower()

        for section in ("observation:", "assessment:", "suggestions:"):
            assert section in prompt

        assert "perceive" in prompt
        assert "feel" in prompt

        for dimension in ("composition", "color & light", "focus", "subject & moment"):
            assert dimension in prompt

        assert "confirm or contradict" in prompt
        assert "never restate the numbers" in prompt
        assert "respond entirely in english (en)" in prompt

        assert "composition: 5.0" in prompt


class TestGetVlmCritique:
    """_get_vlm_critique plumbing: the configured token budget reaches the model
    and a new-structure reply passes through verbatim (no server-side parsing)."""

    _RULE = {"category": "portrait", "aggregate": 6.4, "breakdown": [], "penalties": {}}

    _NEW_FORMAT_REPLY = (
        "Observation: A backlit portrait of a child on a beach at golden hour, "
        "framed slightly left of center.\n"
        "Assessment: Composition leans on a clean rule-of-thirds placement and the "
        "warm light flatters the skin, matching the metrics; focus is soft on the "
        "eyes, contradicting the sharpness score; the moment feels tender.\n"
        "Suggestions: Nail focus on the near eye; add a touch of fill or a reflector; "
        "in editing, lift shadows and add local contrast on the face."
    )

    def _patches(self, crit, tagger, full_config):
        return (
            mock.patch.object(crit, "VIEWER_CONFIG", {"features": {"show_vlm_critique": True}}),
            mock.patch.object(crit, "resolve_vlm_config", return_value={"model_path": "/m"}),
            mock.patch.object(crit, "get_or_load_vlm_tagger", return_value=tagger),
            mock.patch.object(crit, "_load_critique_image", return_value=object()),
            mock.patch.object(crit, "_FULL_CONFIG", full_config),
        )

    def test_max_new_tokens_plumbed_to_generate(self):
        from api.routers import critique as crit

        tagger = mock.MagicMock()
        tagger.generate.return_value = self._NEW_FORMAT_REPLY
        p = self._patches(crit, tagger, {"critique": {"vlm": {"max_new_tokens": 384}}})

        with p[0], p[1], p[2], p[3], p[4]:
            out = crit._get_vlm_critique(_make_photo(), dict(self._RULE), b"thumb")

        assert out == self._NEW_FORMAT_REPLY
        assert tagger.generate.call_count == 1
        _, kwargs = tagger.generate.call_args
        assert kwargs["max_new_tokens"] == 384

    def test_default_token_budget_when_unset(self):
        """Absent config, the code default token budget (320) is used."""
        from api.routers import critique as crit

        tagger = mock.MagicMock()
        tagger.generate.return_value = self._NEW_FORMAT_REPLY
        p = self._patches(crit, tagger, {})  # no critique block -> fall back to 320

        with p[0], p[1], p[2], p[3], p[4]:
            crit._get_vlm_critique(_make_photo(), dict(self._RULE), b"thumb")

        _, kwargs = tagger.generate.call_args
        assert kwargs["max_new_tokens"] == 320

    def test_new_structure_reply_passes_through_verbatim(self):
        """The 3-section reply is returned unchanged (client renders it as-is);
        there is no server-side re-parsing that could drop a section."""
        from api.routers import critique as crit

        tagger = mock.MagicMock()
        tagger.generate.return_value = "  " + self._NEW_FORMAT_REPLY + "  "
        p = self._patches(crit, tagger, {"critique": {"vlm": {"max_new_tokens": 320}}})

        with p[0], p[1], p[2], p[3], p[4]:
            out = crit._get_vlm_critique(_make_photo(), dict(self._RULE), b"thumb")

        assert out == self._NEW_FORMAT_REPLY
        for section in ("Observation:", "Assessment:", "Suggestions:"):
            assert section in out
