"""Visual China submission suitability derived from cached image embeddings.

This signal is deliberately separate from Facet's general-purpose aggregate:
it estimates whether an image resembles a useful, publication-ready stock
photograph.  It does not make any copyright, property-release or model-release
decision; those remain manual checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from analyzers.aesthetic_clip import build_aesthetic_axis, score_embedding


VCG_AGGREGATE_WEIGHT = 0.65
VCG_SUITABILITY_WEIGHT = 0.35
VCG_SCORE_VERSION = "vcg-stock-v1-65-35"

VCG_POSITIVE_PROMPTS: tuple[str, ...] = (
    "a distinctive professional stock photograph with a clear subject",
    "a publication-ready editorial photograph capturing an authentic moment",
    "a commercially useful stock image with strong visual storytelling",
    "a technically clean photograph with intentional composition and broad usability",
    "a visually compelling documentary or creative stock photograph",
)

VCG_NEGATIVE_PROMPTS: tuple[str, ...] = (
    "a casual low-quality snapshot with no clear subject",
    "a blurry noisy badly exposed amateur photograph",
    "a generic repetitive image with weak visual storytelling",
    "a poorly composed photograph with distracting clutter",
    "a watermarked collage screenshot or text-heavy image",
)


def build_vcg_axis(text_encode: callable) -> np.ndarray:
    """Build the unit vector for the dedicated stock-suitability prompts."""
    return build_aesthetic_axis(
        text_encode,
        positive_prompts=VCG_POSITIVE_PROMPTS,
        negative_prompts=VCG_NEGATIVE_PROMPTS,
    )


def score_vcg_suitability(
    embedding: bytes | bytearray | memoryview | np.ndarray | None,
    axis: np.ndarray | None,
) -> float | None:
    """Return a 0-10 suitability score, or ``None`` for an invalid vector."""
    if embedding is None or axis is None:
        return None
    try:
        vector = (
            np.frombuffer(embedding, dtype=np.float32)
            if isinstance(embedding, (bytes, bytearray, memoryview))
            else np.asarray(embedding, dtype=np.float32)
        )
        if vector.ndim != 1 or vector.shape != axis.shape:
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        value = score_embedding(vector / norm, axis)
        return round(value, 2) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def calculate_vcg_submission_score(
    aggregate: float | None,
    suitability: float | None,
) -> float | None:
    """Blend aggregate and suitability; missing suitability falls back to aggregate."""
    try:
        aggregate_value = float(aggregate)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(aggregate_value):
        return None

    try:
        suitability_value = float(suitability) if suitability is not None else aggregate_value
    except (TypeError, ValueError):
        suitability_value = aggregate_value
    if not np.isfinite(suitability_value):
        suitability_value = aggregate_value

    score = (
        VCG_AGGREGATE_WEIGHT * aggregate_value
        + VCG_SUITABILITY_WEIGHT * suitability_value
    )
    return round(max(0.0, min(10.0, score)), 2)


def vcg_scored_at() -> str:
    """Return a stable UTC ISO timestamp for a score calculation."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
