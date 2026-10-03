"""Batch initialization for cached VLM critiques and personalized suggestions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from tqdm import tqdm

from api.routers.critique import (
    _PERSONALIZED_COLUMNS,
    _build_rule_critique,
    _generate_personalized_suggestions,
    _generate_vlm_critique,
    _load_critique_image,
    _make_personalized_cache_payload,
    _read_personalized_cache,
    _read_vlm_cache,
    _score_value,
)
from db import get_connection
from models.vlm_backend import VLMBackendError, create_remote_vlm_tagger

logger = logging.getLogger("facet.critique_batch")

SUPPORTED_CRITIQUE_LANGUAGES = ('en', 'fr', 'de', 'it', 'es', 'pt', 'zh')
_AI_GENERATION_SCORE_THRESHOLD = 5.0

_MODEL_KEY_MAP = {
    'qwen3-vl-2b': 'qwen3_vl_2b',
    'qwen2.5-vl-7b': 'qwen2_5_vl_7b',
    'qwen3.5-2b': 'qwen3_5_2b',
    'qwen3.5-4b': 'qwen3_5_4b',
}


class CritiqueBatchError(RuntimeError):
    """Raised when the batch cannot continue because its VLM is unavailable."""


@dataclass(frozen=True)
class CritiqueBatchSummary:
    total: int = 0
    initialized: int = 0
    skipped: int = 0
    partial: int = 0
    failed: int = 0
    critiques_generated: int = 0
    suggestions_generated: int = 0

    @property
    def incomplete(self) -> int:
        return self.partial + self.failed


def _create_tagger(scoring_config):
    """Resolve one configured VLM instance, matching existing CLI behavior."""
    remote = create_remote_vlm_tagger(scoring_config.config, scoring_config)
    if remote is not None:
        return remote

    from models.vlm_tagger import VLMTagger

    models_config = scoring_config.get_model_config()
    tag_model = scoring_config.get_model_for_task('tagging')
    config_key = _MODEL_KEY_MAP.get(tag_model)
    model_config = models_config.get(config_key) if config_key else None
    if not model_config or not model_config.get('model_path'):
        profile = models_config.get('vram_profile', 'legacy')
        raise CritiqueBatchError(
            "VLM tagger is not available for profile "
            f"{profile} (tagging_model={tag_model}). Configure vlm_backend or a VLM profile."
        )
    return VLMTagger(model_config, scoring_config)


def _select_columns() -> str:
    columns = list(dict.fromkeys([
        *_PERSONALIZED_COLUMNS,
        'thumbnail',
        'vlm_critique',
        'vlm_critique_language',
        'personalized_suggestions',
    ]))
    return ', '.join(columns)


def initialize_critiques(db_path, scoring_config, lang, tagger=None):
    """Fill missing critique caches for every scored photo in ``db_path``.

    Valid caches are skipped independently. Each successful cache write is
    committed immediately so an interrupted run can resume without repeating
    completed inference.
    """
    if lang not in SUPPORTED_CRITIQUE_LANGUAGES:
        raise ValueError(f"Unsupported critique language: {lang}")

    owns_tagger = tagger is None
    tagger_loaded = False
    counts = {
        'total': 0,
        'initialized': 0,
        'skipped': 0,
        'partial': 0,
        'failed': 0,
        'critiques_generated': 0,
        'suggestions_generated': 0,
    }

    try:
        with get_connection(db_path) as conn:
            # Materialize only the small path list before inference.  Keeping a
            # SELECT cursor open while the VLM runs leaves this connection on
            # an old WAL snapshot for minutes at a time.  If the viewer writes
            # meanwhile, the later UPDATE cannot upgrade that stale snapshot
            # and SQLite raises SQLITE_BUSY_SNAPSHOT ("database is locked").
            # A short per-photo SELECT, closed before inference, lets every
            # cache write start from a current transaction instead.
            path_cursor = conn.execute(
                "SELECT path FROM photos WHERE aggregate IS NOT NULL ORDER BY path"
            )
            try:
                photo_paths = [row['path'] for row in path_cursor.fetchall()]
            finally:
                path_cursor.close()
            total = len(photo_paths)
            counts['total'] = total
            logger.info("Initializing AI critiques and personalized suggestions for %d photos...", total)

            with tqdm(total=total, desc="AI critique initialization") as progress:
                for path in photo_paths:
                    row_cursor = conn.execute(
                        f"SELECT {_select_columns()} FROM photos "
                        "WHERE path = ? AND aggregate IS NOT NULL",
                        (path,),
                    )
                    try:
                        row = row_cursor.fetchone()
                    finally:
                        row_cursor.close()
                    if row is None:
                        # The viewer may remove a photo while this long-running
                        # command is active.  It no longer needs initialization.
                        counts['skipped'] += 1
                        progress.update(1)
                        continue

                    photo = dict(row)
                    aggregate_score = _score_value(photo.get('aggregate'))
                    vcg_score = _score_value(photo.get('vcg_submission_score'))
                    if (aggregate_score is not None
                            and vcg_score is not None
                            and aggregate_score < _AI_GENERATION_SCORE_THRESHOLD
                            and vcg_score < _AI_GENERATION_SCORE_THRESHOLD):
                        counts['skipped'] += 1
                        progress.update(1)
                        continue

                    has_critique = bool(_read_vlm_cache(
                        photo.get('vlm_critique'),
                        photo.get('vlm_critique_language'),
                        lang,
                    ))
                    has_suggestions = bool(_read_personalized_cache(
                        photo.get('personalized_suggestions'), photo, lang,
                    ))

                    if has_critique and has_suggestions:
                        counts['skipped'] += 1
                        progress.update(1)
                        continue

                    try:
                        if tagger is None:
                            tagger = _create_tagger(scoring_config)
                        if not tagger_loaded:
                            tagger.load()
                            tagger_loaded = True

                        rule_critique = _build_rule_critique(photo, scoring_config)
                        image = _load_critique_image(path, photo.get('thumbnail'))

                        if not has_critique:
                            critique = _generate_vlm_critique(
                                tagger,
                                photo,
                                rule_critique,
                                image,
                                lang,
                                scoring_config.config,
                            )
                            if critique:
                                conn.execute(
                                    "UPDATE photos SET vlm_critique = ?, "
                                    "vlm_critique_language = ?, "
                                    "vlm_critique_translated = NULL WHERE path = ?",
                                    (critique, lang, path),
                                )
                                conn.commit()
                                has_critique = True
                                counts['critiques_generated'] += 1

                        if not has_suggestions:
                            if aggregate_score is not None and vcg_score is not None:
                                suggestions = _generate_personalized_suggestions(
                                    tagger,
                                    photo,
                                    rule_critique,
                                    image,
                                    lang,
                                    scoring_config.config,
                                )
                                if suggestions:
                                    payload = _make_personalized_cache_payload(
                                        photo, lang, suggestions,
                                    )
                                    conn.execute(
                                        "UPDATE photos SET personalized_suggestions = ? WHERE path = ?",
                                        (json.dumps(payload, ensure_ascii=False), path),
                                    )
                                    conn.commit()
                                    has_suggestions = True
                                    counts['suggestions_generated'] += 1
                            else:
                                logger.warning(
                                    "Personalized suggestions skipped for %s: required scores are missing",
                                    path,
                                )
                    except VLMBackendError as ex:
                        conn.rollback()
                        raise CritiqueBatchError(f"VLM backend is unavailable: {ex}") from ex
                    except Exception as ex:
                        # A failed UPDATE can leave the connection inside an
                        # aborted transaction.  Reset it so one photo cannot
                        # poison every remaining item in this resumable run.
                        conn.rollback()
                        logger.warning("Critique initialization failed for %s: %s", path, ex)

                    if has_critique and has_suggestions:
                        counts['initialized'] += 1
                    elif has_critique or has_suggestions:
                        counts['partial'] += 1
                    else:
                        counts['failed'] += 1
                    progress.update(1)
    finally:
        if owns_tagger and tagger is not None and tagger_loaded:
            tagger.unload()

    summary = CritiqueBatchSummary(**counts)
    logger.info(
        "AI critique initialization complete: total=%d, initialized=%d, skipped=%d, "
        "partial=%d, failed=%d, critiques_generated=%d, suggestions_generated=%d",
        summary.total,
        summary.initialized,
        summary.skipped,
        summary.partial,
        summary.failed,
        summary.critiques_generated,
        summary.suggestions_generated,
    )
    return summary
