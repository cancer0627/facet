"""Tests for the resumable AI critique batch initializer."""

from __future__ import annotations

import io
import json
import sqlite3

import pytest
from PIL import Image

import processing.critique_batch as critique_batch
from config import ScoringConfig
from db import init_database
from models.vlm_backend import VLMBackendError
from processing.critique_batch import (
    CritiqueBatchError,
    initialize_critiques,
)


class FakeTagger:
    def __init__(self, personalized_response=None, critique_response="观察：画面清晰。"):
        self.personalized_response = personalized_response or json.dumps({
            "aggregate_action": "收紧裁切",
            "aggregate_reason": "主体占比偏小",
            "vcg_action": "校正高光",
            "vcg_reason": "提升交付完整度",
        }, ensure_ascii=False)
        self.critique_response = critique_response
        self.prompts = []
        self.load_calls = 0
        self.unload_calls = 0

    def load(self):
        self.load_calls += 1

    def unload(self):
        self.unload_calls += 1

    def generate(self, image, prompt, max_new_tokens):
        self.prompts.append((prompt, max_new_tokens))
        if 'aggregate_action' in prompt:
            return self.personalized_response
        return self.critique_response


class FailingBackendTagger(FakeTagger):
    def generate(self, image, prompt, max_new_tokens):
        raise VLMBackendError("connection refused")


class ConcurrentWriteTagger(FakeTagger):
    """Simulate a viewer write while the batch is doing slow inference."""

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.external_write_done = False

    def generate(self, image, prompt, max_new_tokens):
        if not self.external_write_done:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE photos SET star_rating = 1 WHERE path = '/photos/a.jpg'"
                )
            self.external_write_done = True
        return super().generate(image, prompt, max_new_tokens)


def _thumbnail():
    image = Image.new('RGB', (16, 12), color=(80, 120, 160))
    out = io.BytesIO()
    image.save(out, format='JPEG')
    return out.getvalue()


@pytest.fixture()
def scoring_config():
    return ScoringConfig(validate=False)


@pytest.fixture()
def critique_db(tmp_path):
    db_path = str(tmp_path / 'critique-batch.db')
    init_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO photos (
                path, filename, category, aggregate,
                vcg_suitability_score, vcg_submission_score, vcg_score_version,
                config_version, scanned_at, vcg_scored_at, thumbnail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                '/photos/a.jpg', 'a.jpg', 'landscape', 7.5,
                6.5, 7.15, 'vcg-stock-v1-65-35',
                'test-config', '2026-10-01T00:00:00+00:00',
                '2026-10-01T00:00:00+00:00', _thumbnail(),
            ),
        )
    return db_path


def _cached_row(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute(
            "SELECT vlm_critique, vlm_critique_language, personalized_suggestions "
            "FROM photos WHERE path = '/photos/a.jpg'"
        ).fetchone())


def test_initializes_both_caches_and_rerun_skips(critique_db, scoring_config):
    tagger = FakeTagger()

    first = initialize_critiques(critique_db, scoring_config, 'zh', tagger=tagger)

    assert first.initialized == 1
    assert first.critiques_generated == 1
    assert first.suggestions_generated == 1
    assert first.incomplete == 0
    assert len(tagger.prompts) == 2
    row = _cached_row(critique_db)
    assert row['vlm_critique_language'] == 'zh'
    payload = json.loads(row['personalized_suggestions'])
    assert payload['lang'] == 'zh'
    assert payload['aggregate_suggestions'][0]['action'] == '收紧裁切'

    rerun_tagger = FakeTagger()
    second = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=rerun_tagger,
    )

    assert second.skipped == 1
    assert second.critiques_generated == 0
    assert second.suggestions_generated == 0
    assert rerun_tagger.prompts == []


def test_command_owned_tagger_is_loaded_once_and_unloaded(
        critique_db, scoring_config, monkeypatch):
    tagger = FakeTagger()
    monkeypatch.setattr(critique_batch, '_create_tagger', lambda _config: tagger)

    summary = initialize_critiques(critique_db, scoring_config, 'zh')

    assert summary.initialized == 1
    assert tagger.load_calls == 1
    assert tagger.unload_calls == 1
    assert len(tagger.prompts) == 2


def test_score_change_regenerates_only_personalized_cache(critique_db, scoring_config):
    initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=FakeTagger(),
    )
    with sqlite3.connect(critique_db) as conn:
        conn.execute("UPDATE photos SET aggregate = 6.8 WHERE path = '/photos/a.jpg'")

    tagger = FakeTagger()
    summary = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=tagger,
    )

    assert summary.initialized == 1
    assert summary.critiques_generated == 0
    assert summary.suggestions_generated == 1
    assert len(tagger.prompts) == 1
    assert 'aggregate_action' in tagger.prompts[0][0]


def test_both_scores_below_five_skip_all_ai_generation(
        critique_db, scoring_config):
    with sqlite3.connect(critique_db) as conn:
        conn.execute(
            "UPDATE photos SET aggregate = 4.9, vcg_submission_score = 4.8 "
            "WHERE path = '/photos/a.jpg'"
        )
    tagger = FakeTagger()

    summary = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=tagger,
    )

    assert summary.skipped == 1
    assert summary.incomplete == 0
    assert summary.critiques_generated == 0
    assert summary.suggestions_generated == 0
    assert tagger.load_calls == 0
    assert tagger.prompts == []
    row = _cached_row(critique_db)
    assert row['vlm_critique'] is None
    assert row['personalized_suggestions'] is None


@pytest.mark.parametrize(
    ('aggregate', 'vcg_submission_score'),
    [(5.0, 4.9), (4.9, 5.0)],
)
def test_score_equal_to_five_does_not_skip_ai_generation(
        critique_db, scoring_config, aggregate, vcg_submission_score):
    with sqlite3.connect(critique_db) as conn:
        conn.execute(
            "UPDATE photos SET aggregate = ?, vcg_submission_score = ? "
            "WHERE path = '/photos/a.jpg'",
            (aggregate, vcg_submission_score),
        )
    tagger = FakeTagger()

    summary = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=tagger,
    )

    assert summary.initialized == 1
    assert summary.incomplete == 0
    assert summary.critiques_generated == 1
    assert summary.suggestions_generated == 1
    assert len(tagger.prompts) == 2


def test_language_change_regenerates_both_caches(critique_db, scoring_config):
    initialize_critiques(
        critique_db, scoring_config, 'en', tagger=FakeTagger(),
    )

    tagger = FakeTagger()
    summary = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=tagger,
    )

    assert summary.initialized == 1
    assert summary.critiques_generated == 1
    assert summary.suggestions_generated == 1
    assert len(tagger.prompts) == 2
    assert _cached_row(critique_db)['vlm_critique_language'] == 'zh'


def test_partial_success_is_committed_and_resumable(critique_db, scoring_config):
    first_tagger = FakeTagger(personalized_response='not-json')
    first = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=first_tagger,
    )

    assert first.partial == 1
    assert first.incomplete == 1
    row = _cached_row(critique_db)
    assert row['vlm_critique'] == '观察：画面清晰。'
    assert row['personalized_suggestions'] is None

    second_tagger = FakeTagger()
    second = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=second_tagger,
    )

    assert second.initialized == 1
    assert second.critiques_generated == 0
    assert second.suggestions_generated == 1
    assert len(second_tagger.prompts) == 1


def test_backend_failure_is_clear_and_preserves_cache(critique_db, scoring_config):
    with pytest.raises(CritiqueBatchError, match='VLM backend is unavailable'):
        initialize_critiques(
            critique_db, scoring_config, 'zh', tagger=FailingBackendTagger(),
        )

    row = _cached_row(critique_db)
    assert row['vlm_critique'] is None
    assert row['personalized_suggestions'] is None


def test_viewer_write_during_inference_does_not_lock_batch(
        critique_db, scoring_config):
    # The previous implementation fetched 100 rows at a time from one cursor.
    # Keep a 101st row pending so that cursor still owns a read snapshot when
    # the simulated viewer commits its write during the first inference.
    with sqlite3.connect(critique_db) as conn:
        conn.executemany(
            """
            INSERT INTO photos (
                path, filename, category, aggregate,
                vcg_suitability_score, vcg_submission_score, vcg_score_version,
                config_version, scanned_at, vcg_scored_at, thumbnail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f'/photos/extra-{index:03d}.jpg', f'extra-{index:03d}.jpg',
                    'landscape', 7.5, 6.5, 7.15, 'vcg-stock-v1-65-35',
                    'test-config', '2026-10-01T00:00:00+00:00',
                    '2026-10-01T00:00:00+00:00', _thumbnail(),
                )
                for index in range(100)
            ],
        )
    tagger = ConcurrentWriteTagger(critique_db)

    summary = initialize_critiques(
        critique_db, scoring_config, 'zh', tagger=tagger,
    )

    assert tagger.external_write_done is True
    assert summary.initialized == 101
    assert summary.failed == 0
    assert summary.critiques_generated == 101
    assert summary.suggestions_generated == 101
