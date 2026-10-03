"""
AI Critique router — rule-based and VLM-powered score explanations.

Provides per-photo analysis: score breakdown, strengths, weaknesses, suggestions.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import CurrentUser, get_optional_user, is_edition_authenticated
from api.config import VIEWER_CONFIG, _FULL_CONFIG
from api.database import get_async_db
from api.db_helpers import get_existing_columns, get_visibility_clause
from api.models.media import MediaCritiqueResponse, MediaPersonalizedSuggestionsResponse
from api.model_cache import (
    get_or_load_vlm_tagger,
    resolve_vlm_config,
    resolve_personalized_vlm_config,
    translation_target,
    vlm_generate_lock,
)

router = APIRouter(tags=["critique"])
logger = logging.getLogger(__name__)

# Score thresholds for strength/weakness classification
_STRENGTH_THRESHOLD = 7.5
_WEAKNESS_THRESHOLD = 5.0
# Noise thresholds
_NOISE_CLEAN_THRESHOLD = 3.0
_NOISE_HIGH_THRESHOLD = 8.0
_NOISE_PENALTY_THRESHOLD = 4.0

# Metric labels for human-readable output
METRIC_LABELS = {
    'aesthetic': 'Aesthetic Quality',
    'quality_score': 'Overall Image Quality',
    'tech_sharpness': 'Technical Sharpness',
    'face_quality': 'Face Quality',
    'eye_sharpness': 'Eye Sharpness',
    'face_sharpness': 'Face Sharpness',
    'comp_score': 'Composition',
    'exposure_score': 'Exposure',
    'color_score': 'Color',
    'contrast_score': 'Contrast',
    'isolation_bonus': 'Subject Isolation',
    'noise_sigma': 'Noise Level',
    'dynamic_range_stops': 'Dynamic Range',
    'leading_lines_score': 'Leading Lines',
    'power_point_score': 'Power Points',
    'aesthetic_iaa': 'Aesthetic (IAA)',
    'face_quality_iqa': 'Face Quality (IQA)',
    'liqe_score': 'LIQE Quality',
    'subject_sharpness': 'Subject Sharpness',
    'subject_prominence': 'Subject Prominence',
    'subject_placement': 'Subject Placement',
    'bg_separation': 'Background Separation',
    'mean_saturation': 'Saturation',
    'mean_luminance': 'Luminance',
    'form_symmetry': 'Symmetry',
    'form_balance': 'Visual Balance',
    'form_edge_entropy': 'Edge Entropy',
    'form_fractal': 'Fractal Complexity',
    'color_harmony': 'Color Harmony',
}

# Map config weight keys to DB column names
WEIGHT_TO_COLUMN = {
    'aesthetic': 'aesthetic',
    'quality': 'quality_score',
    'face_quality': 'face_quality',
    'face_sharpness': 'face_sharpness',
    'eye_sharpness': 'eye_sharpness',
    'tech_sharpness': 'tech_sharpness',
    'composition': 'comp_score',
    'exposure': 'exposure_score',
    'color': 'color_score',
    'contrast': 'contrast_score',
    'isolation': 'isolation_bonus',
    'dynamic_range': 'dynamic_range_stops',
    'leading_lines': 'leading_lines_score',
    'power_point': 'power_point_score',
    'aesthetic_iaa': 'aesthetic_iaa',
    'face_quality_iqa': 'face_quality_iqa',
    'liqe': 'liqe_score',
    'subject_sharpness': 'subject_sharpness',
    'subject_prominence': 'subject_prominence',
    'subject_placement': 'subject_placement',
    'bg_separation': 'bg_separation',
    'noise': 'noise_sigma',
    'saturation': 'mean_saturation',
    'symmetry': 'form_symmetry',
    'balance': 'form_balance',
    'edge_entropy': 'form_edge_entropy',
    'fractal': 'form_fractal',
    'color_harmony': 'color_harmony',
}

# Action templates keyed by metric name (used to build ranked suggestions)
SUGGESTIONS = {
    'aesthetic': 'Consider stronger visual impact through better lighting or subject matter',
    'quality_score': 'Inspect the image at 100% and correct the most visible blur, noise, or compression artifact first',
    'tech_sharpness': 'Use a faster shutter speed or tripod to improve sharpness',
    'face_quality': 'Ensure the face is well-lit and in focus',
    'eye_sharpness': 'Focus precisely on the eyes for portraits',
    'face_sharpness': 'Ensure the face region is sharp — avoid motion blur',
    'comp_score': 'Try applying compositional rules like rule of thirds or leading lines',
    'exposure_score': 'Adjust exposure to avoid clipping highlights or crushing shadows',
    'color_score': 'Consider white balance correction or more vibrant color grading',
    'contrast_score': 'Increase tonal contrast for more visual depth',
    'noise_sigma': 'Use a lower ISO or apply noise reduction',
    'dynamic_range_stops': 'Bracket exposures or use graduated filters for better dynamic range',
    'leading_lines_score': 'Look for natural lines that draw the eye into the frame',
    'power_point_score': 'Crop or reframe so the main subject sits closer to a rule-of-thirds intersection',
    'aesthetic_iaa': 'Simplify competing elements and strengthen the subject with more deliberate light and color',
    'face_quality_iqa': 'Correct face exposure and noise locally, then sharpen the eyes without oversharpening skin',
    'subject_sharpness': 'Ensure your main subject is the sharpest element in the frame',
    'subject_prominence': 'Give the subject more frame space or use a shallower depth of field',
    'subject_placement': 'Crop or reframe to move the subject toward a clear visual anchor such as a thirds intersection',
    'bg_separation': 'Use wider aperture or greater distance to separate subject from background',
    'liqe_score': 'Improve overall image quality — check for distortions or artifacts',
    'isolation_bonus': 'Use wider aperture to better isolate the subject from background',
    'form_symmetry': 'Center the subject or align strong shapes to balance the left and right halves of the frame',
    'form_balance': 'Recompose so the visual weight sits closer to the frame center or a balanced thirds position',
    'form_edge_entropy': 'Add more varied lines and textures — the dominant edges all run in the same direction',
    'form_fractal': 'Include richer detail or texture — the frame reads as visually sparse',
    'color_harmony': 'Adjust the palette toward a harmonic hue scheme such as complementary or analogous colors',
}

_NON_ACTIONABLE_METRICS = {'mean_saturation', 'mean_luminance'}

_PERSONALIZED_SUGGESTIONS_VERSION = 'personalized-v1'
_PERSONALIZED_SUGGESTION_LIMIT = 3

# Keep the rule critique projection stable. The personalized endpoint extends it
# with the score-version fields needed for cache invalidation.
_CRITIQUE_COLUMNS = [
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

_PERSONALIZED_COLUMNS = [
    *_CRITIQUE_COLUMNS,
    'vcg_suitability_score', 'vcg_submission_score', 'vcg_score_version',
    'config_version', 'scanned_at', 'vcg_scored_at',
]


def _build_category_trail(photo, matched_category, sc):
    """Build list of interesting rejected categories evaluated before the match.

    Only includes categories where the rejection is non-trivial (not just
    'no matching tags for a tag-only category').
    """
    from config.category_filter import CategoryFilter

    rejected = []
    for cat in sc.get_categories():
        name = cat.get('name')
        if name == matched_category:
            break

        filters = cat.get('filters', {})
        if not filters:
            continue

        cf = CategoryFilter(filters)
        mismatch = cf.explain_mismatch(photo)
        if mismatch is None:
            continue

        # Skip trivially irrelevant tag-only categories: if the only filters
        # are tag-related and none of the required tags appear in the photo
        filter_keys = set(filters.keys()) - {'tag_match_mode'}
        is_tag_only = filter_keys <= {'required_tags', 'excluded_tags'}
        if is_tag_only and mismatch['key'] == 'required_tags' and not mismatch.get('actual'):
            continue

        rejected.append({'category': name, 'mismatch': mismatch})
        if len(rejected) >= 5:
            break

    return rejected


def _build_category_reason(photo, category, sc):
    """Build structured category reason for i18n on the frontend."""
    cat_config = sc.get_category_config(category)
    if not cat_config:
        return {'reason_key': 'default', 'category': category or 'default', 'details': [], 'rejected': []}

    filters = cat_config.get('filters', {})
    details = []

    if 'face_ratio_min' in filters and photo.get('face_ratio'):
        details.append({
            'key': 'face_ratio',
            'value': round(photo['face_ratio'], 2),
            'threshold': filters['face_ratio_min'],
        })
    if 'face_count_min' in filters and photo.get('face_count'):
        details.append({
            'key': 'face_count',
            'value': photo['face_count'],
            'threshold': filters['face_count_min'],
        })
    if filters.get('is_monochrome') and photo.get('is_monochrome'):
        details.append({'key': 'monochrome'})
    if filters.get('is_silhouette') and photo.get('is_silhouette'):
        details.append({'key': 'silhouette'})
    if filters.get('required_tags'):
        tags = photo.get('tags', '') or ''
        matched = [t for t in filters['required_tags'] if t in tags]
        if matched:
            details.append({'key': 'tags', 'tags': matched})
    if 'luminance_max' in filters and photo.get('mean_luminance') is not None:
        details.append({
            'key': 'luminance',
            'value': round(photo['mean_luminance'], 2),
            'threshold': filters['luminance_max'],
        })
    if 'shutter_speed_min' in filters and photo.get('shutter_speed'):
        details.append({'key': 'long_exposure'})

    rejected = _build_category_trail(photo, category, sc)

    return {
        'reason_key': 'matched' if details else 'matched_generic',
        'category': category,
        'details': details,
        'rejected': rejected,
    }


def _calculate_breakdown(photo, sc, category):
    """Calculate score breakdown from photo metrics and category weights.

    Returns a sorted list of score contributions with metric details.
    """
    weights = sc.get_weights(category)
    if not weights:
        weights = sc.get_weights('')

    breakdown = []

    for weight_key, weight_val in weights.items():
        if weight_key in ('bonus', 'blink_penalty', 'noise_tolerance_multiplier',
                          'noise_penalty_max', 'noise_threshold', 'score_min', 'score_max',
                          'bimodality_threshold', 'bimodality_penalty',
                          'oversaturation_threshold', 'oversaturation_penalty',
                          'clipping_multiplier', 'noise_penalty_rate'):
            continue

        col = WEIGHT_TO_COLUMN.get(weight_key)
        if not col or weight_val <= 0:
            continue

        value = photo.get(col)
        if value is None:
            continue

        display_value = float(value)
        contribution = display_value * weight_val

        breakdown.append({
            'metric': METRIC_LABELS.get(col, weight_key),
            'metric_key': col,
            'value': round(display_value, 2),
            'weight': round(weight_val, 3),
            'contribution': round(contribution, 2),
        })

    breakdown.sort(key=lambda x: x['contribution'], reverse=True)
    return breakdown


def _suggestion_priority(item):
    """Estimate how much improving one metric can affect this photo's score.

    Suggestions explain this photo's weighted score, so a moderate score with a
    large weight can matter more than a very low diagnostic with a tiny weight.
    Noise is inverted (lower is better); all other critique metrics are scored
    on the usual 0-10 scale.
    """
    value = item['value']
    if item['metric_key'] == 'noise_sigma':
        improvement_room = max(0.0, value - _NOISE_CLEAN_THRESHOLD)
    else:
        improvement_room = max(0.0, 10.0 - value)
    return improvement_room * item['weight']


def _build_suggestions(breakdown, limit=3):
    """Return concrete suggestion keys for the highest-impact opportunities.

    Unlike weakness badges, suggestions are not hidden behind the old ``>5%``
    weight gate. That gate left photos with visibly weak, low-weight metrics
    without any next action. Every scored photo now receives up to ``limit``
    suggestions as long as it has an actionable weighted metric.
    """
    candidates = [
        item for item in breakdown
        if item['metric_key'] in SUGGESTIONS
        and item['metric_key'] not in _NON_ACTIONABLE_METRICS
        and _suggestion_priority(item) > 0
    ]
    candidates.sort(
        key=lambda item: (
            _suggestion_priority(item),
            item['weight'],
            -item['value'] if item['metric_key'] != 'noise_sigma' else item['value'],
        ),
        reverse=True,
    )
    return [item['metric_key'] for item in candidates[:limit]]


def _identify_strengths_weaknesses(breakdown):
    """Identify strengths and weaknesses from a score breakdown.

    Returns a (strengths, weaknesses, suggestions) tuple. Strengths and weaknesses
    are lists of dicts with metric_key and value; suggestions is a list of metric keys.
    """
    strengths = []
    weaknesses = []

    for item in breakdown:
        val = item['value']
        metric_key = item['metric_key']

        # Noise is inverted -- high noise_sigma is bad
        if metric_key == 'noise_sigma':
            if val < _NOISE_CLEAN_THRESHOLD:
                strengths.append({'metric_key': metric_key, 'value': round(val, 1)})
            elif val > _NOISE_HIGH_THRESHOLD:
                weaknesses.append({'metric_key': metric_key, 'value': round(val, 1)})
        elif metric_key in _NON_ACTIONABLE_METRICS:
            continue  # Not meaningful as strengths/weaknesses
        else:
            if val >= _STRENGTH_THRESHOLD:
                strengths.append({'metric_key': metric_key, 'value': round(val, 1)})
            elif val < _WEAKNESS_THRESHOLD and item['weight'] > 0.05:
                weaknesses.append({'metric_key': metric_key, 'value': round(val, 1)})

    return strengths, weaknesses, _build_suggestions(breakdown)


def _score_value(value):
    """Return a JSON/API-friendly score without changing missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_penalties(photo):
    """Check for scoring penalties (blink, noise, clipping, skin-tone cast).

    Returns a dict of penalty names to values. The skin-tone entry is advisory
    (it never enters the aggregate) and surfaces whenever the stored worst-face
    cast is present: ``compute_photo_skin_tone`` already applied the configured
    ``cast_delta_threshold`` once at recompute time (``skin_tone_cast`` is NULL
    below it), so the stored decision is trusted here rather than re-gated.
    """
    penalties = {}
    is_blink = photo.get('is_blink')
    is_blink_text = isinstance(is_blink, str) and is_blink.lower() in {'1', 'true'}
    if is_blink is True or is_blink == 1 or is_blink_text:
        penalties['blink'] = True
    noise_sigma = _score_value(photo.get('noise_sigma'))
    if noise_sigma is not None and noise_sigma > _NOISE_PENALTY_THRESHOLD:
        noise_penalty = min(1.5, max(0, (noise_sigma - _NOISE_PENALTY_THRESHOLD) * 0.3))
        if noise_penalty > 0:
            penalties['noise'] = round(-noise_penalty, 2)
    highlight_clipped = _score_value(photo.get('highlight_clipped'))
    if highlight_clipped is not None and highlight_clipped > 0:
        penalties['highlight_clipping'] = round(-highlight_clipped * 1.0, 2)
    shadow_clipped = _score_value(photo.get('shadow_clipped'))
    if shadow_clipped is not None and shadow_clipped > 0:
        penalties['shadow_clipping'] = round(-shadow_clipped * 0.5, 2)
    skin_delta = _score_value(photo.get('skin_tone_delta'))
    if skin_delta is not None and photo.get('skin_tone_cast'):
        penalties['skin_tone'] = {
            'cast': photo['skin_tone_cast'],
            'delta': round(float(skin_delta), 1),
        }
    return penalties


def _parse_distortions(raw):
    """Attribute keys from the stored distortion_attributes JSON column."""
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(entries, list):
        return []
    return [e['attribute'] for e in entries if isinstance(e, dict) and e.get('attribute')]


def _build_rule_critique(photo, scoring_config=None):
    """Build a rule-based critique from stored metrics."""
    if scoring_config is None:
        from api.config import server_scoring_config

        scoring_config = server_scoring_config()
    sc = scoring_config
    category = photo.get('category', '')

    breakdown = _calculate_breakdown(photo, sc, category)
    strengths, weaknesses, suggestions = _identify_strengths_weaknesses(breakdown)
    penalties = _check_penalties(photo)
    category_reason = _build_category_reason(photo, category, sc)

    return {
        'category': category or 'default',
        'category_reason': category_reason,
        'aggregate': photo.get('aggregate'),
        'breakdown': breakdown,
        'strengths': sorted(strengths, key=lambda x: x['value'], reverse=True)[:5],
        'weaknesses': sorted(weaknesses, key=lambda x: x['value'])[:5],
        'suggestions': suggestions,
        'penalties': penalties,
        'distortions': _parse_distortions(photo.get('distortion_attributes')),
    }


def _personalized_context(photo, lang):
    """Build the score context that determines whether a cache is still valid."""
    metric_columns = sorted(set(WEIGHT_TO_COLUMN.values()))
    return {
        'version': _PERSONALIZED_SUGGESTIONS_VERSION,
        'lang': lang or 'en',
        'category': photo.get('category'),
        'aggregate': _score_value(photo.get('aggregate')),
        'vcg_submission_score': _score_value(photo.get('vcg_submission_score')),
        'vcg_suitability_score': _score_value(photo.get('vcg_suitability_score')),
        'vcg_score_version': photo.get('vcg_score_version'),
        'config_version': photo.get('config_version'),
        'scanned_at': photo.get('scanned_at'),
        'vcg_scored_at': photo.get('vcg_scored_at'),
        'metrics': {column: _score_value(photo.get(column)) for column in metric_columns},
        'penalties': _check_penalties(photo),
    }


def _personalized_context_hash(photo, lang):
    context = json.dumps(
        _personalized_context(photo, lang),
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return hashlib.sha256(context.encode('utf-8')).hexdigest()


def _normalise_personalized_suggestions(value):
    """Validate and cap one structured suggestion group from model output."""
    if not isinstance(value, list):
        return None

    normalized = []
    for item in value[:_PERSONALIZED_SUGGESTION_LIMIT]:
        if not isinstance(item, dict):
            return None
        action = item.get('action')
        reason = item.get('reason')
        if not isinstance(action, str) or not action.strip():
            return None
        if not isinstance(reason, str) or not reason.strip():
            return None
        normalized.append({
            'action': action.strip(),
            'reason': reason.strip(),
        })
    return normalized


def _parse_personalized_suggestions(raw):
    """Parse strict JSON returned either directly or in a Markdown code fence."""
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    aggregate = _normalise_personalized_suggestions(payload.get('aggregate_suggestions'))
    vcg = _normalise_personalized_suggestions(payload.get('vcg_suggestions'))
    if aggregate is None or vcg is None:
        # Small CPU VLMs（视觉语言模型）follow a compact object schema more
        # reliably than two nested arrays. Accept that internal shape and keep
        # the public API（应用程序接口）unchanged.
        compact_keys = (
            'aggregate_action', 'aggregate_reason', 'vcg_action', 'vcg_reason',
        )
        if not all(isinstance(payload.get(key), str) and payload[key].strip() for key in compact_keys):
            return None
        aggregate = [{
            'action': payload['aggregate_action'].strip(),
            'reason': payload['aggregate_reason'].strip(),
        }]
        vcg = [{
            'action': payload['vcg_action'].strip(),
            'reason': payload['vcg_reason'].strip(),
        }]
    return {
        'aggregate_suggestions': aggregate,
        'vcg_suggestions': vcg,
    }


def _read_personalized_cache(raw, photo, lang):
    """Return a cache only when its version, language and score context match."""
    if not raw:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        payload = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('version') != _PERSONALIZED_SUGGESTIONS_VERSION:
        return None
    if payload.get('lang') != (lang or 'en'):
        return None
    if payload.get('context_hash') != _personalized_context_hash(photo, lang):
        return None
    generated_at = payload.get('generated_at')
    if not isinstance(generated_at, str) or not generated_at:
        return None

    aggregate = _normalise_personalized_suggestions(payload.get('aggregate_suggestions'))
    vcg = _normalise_personalized_suggestions(payload.get('vcg_suggestions'))
    if aggregate is None or vcg is None:
        return None
    return {
        'aggregate_suggestions': aggregate,
        'vcg_suggestions': vcg,
        'generated_at': generated_at,
    }


def _read_vlm_cache(text, cached_language, lang):
    """Return a VLM critique only when its stored language matches."""
    requested_lang = lang or 'en'
    if not isinstance(text, str) or not text.strip():
        return None
    if cached_language == requested_lang:
        return text
    # Databases created before language-aware caching contain English text
    # without a marker. Preserve that compatibility only for English.
    if cached_language is None and requested_lang == 'en':
        return text
    return None


def _make_personalized_cache_payload(photo, lang, suggestions, generated_at=None):
    """Build the single cache shape shared by the API and batch initializer."""
    requested_lang = lang or 'en'
    return {
        'version': _PERSONALIZED_SUGGESTIONS_VERSION,
        'lang': requested_lang,
        'context_hash': _personalized_context_hash(photo, requested_lang),
        'generated_at': generated_at or datetime.now(timezone.utc).isoformat(),
        **suggestions,
    }


def _personalized_response(photo, source, suggestions=None, generated_at=None,
                           reason=None, warning=None):
    suggestions = suggestions or {}
    return {
        'available': source != 'unavailable',
        'source': source,
        'aggregate_score': _score_value(photo.get('aggregate')),
        'vcg_submission_score': _score_value(photo.get('vcg_submission_score')),
        'aggregate_suggestions': suggestions.get('aggregate_suggestions', []),
        'vcg_suggestions': suggestions.get('vcg_suggestions', []),
        'generated_at': generated_at,
        'reason': reason,
        'warning': warning,
    }


def _build_personalized_prompt(photo, rule_critique, lang, full_config=None):
    """Build the independent structured prompt for personalized suggestions."""
    full_config = _FULL_CONFIG if full_config is None else full_config
    breakdown_lines = []
    for item in rule_critique.get('breakdown', [])[:12]:
        suffix = ', lower is better' if item['metric_key'] == 'noise_sigma' else ''
        breakdown_lines.append(
            f"- {item['metric']}: {item['value']} (weight {item['weight']:.0%}{suffix})"
        )
    breakdown = '\n'.join(breakdown_lines) or '- no per-metric data available'

    penalties = rule_critique.get('penalties', {})
    penalty_text = json.dumps(penalties, ensure_ascii=False, sort_keys=True) if penalties else '{}'
    exif_parts = []
    if photo.get('f_stop'):
        exif_parts.append(f"f/{photo['f_stop']}")
    if photo.get('shutter_speed'):
        exif_parts.append(f"{photo['shutter_speed']}s")
    if photo.get('iso'):
        exif_parts.append(f"ISO {photo['iso']}")
    if photo.get('focal_length'):
        exif_parts.append(f"{photo['focal_length']}mm")
    exif = ', '.join(exif_parts) or 'unknown'

    aggregate = _score_value(photo.get('aggregate'))
    vcg_score = _score_value(photo.get('vcg_submission_score'))
    suitability = _score_value(photo.get('vcg_suitability_score'))
    category = photo.get('category') or 'photo'
    output_lang = lang or 'en'
    vlm_config = ((full_config.get('critique') or {}).get('vlm') or {})
    try:
        max_suggestions = int(vlm_config.get('personalized_max_items', _PERSONALIZED_SUGGESTION_LIMIT))
    except (TypeError, ValueError):
        max_suggestions = _PERSONALIZED_SUGGESTION_LIMIT
    max_suggestions = max(1, min(_PERSONALIZED_SUGGESTION_LIMIT, max_suggestions))
    suggestion_count_instruction = (
        "Return exactly one item in each array."
        if max_suggestions == 1
        else f"Return no more than {max_suggestions} item(s) in each array."
    )

    return (
        "You are an expert photography improvement coach. Inspect the current "
        f"{category} photograph itself and produce concrete, executable advice in "
        f"the requested language ({output_lang}). Facet's current aggregate score "
        f"is {aggregate if aggregate is not None else 'unavailable'}/10. Its internal "
        "Visual China suitability estimate is "
        f"{suitability if suitability is not None else 'unavailable'}/10, and its "
        f"internal Visual China submission estimate is {vcg_score if vcg_score is not None else 'unavailable'}/10. "
        "The submission estimate is calculated as 65% aggregate + 35% suitability; "
        "it is an internal estimate, not an official Visual China score.\n\n"
        "Score breakdown:\n"
        f"{breakdown}\n"
        f"Scoring penalties: {penalty_text}\n"
        f"Camera settings: {exif}\n\n"
        "Return only valid JSON with exactly four string fields and no Markdown or "
        "explanation outside the JSON object. Use this compact shape so the "
        "result is unambiguous:\n"
        '{"aggregate_action":"具体行动","aggregate_reason":"对应当前照片问题",'
        '"vcg_action":"具体行动","vcg_reason":"对应视觉中国申请场景"}\n'
        f"The API will place these into two suggestion groups, each with at most 3 items; {suggestion_count_instruction} "
        "Keep each action and reason concise, preferably under 20 words. Every action must be concrete and "
        "executable, and must clearly distinguish a shooting change from a "
        "post-processing change. Base every reason on visible pixels or the supplied "
        "score context. Do not predict any numeric score increase. Do not claim to "
        "know official Visual China review rules. Do not infer copyright, market "
        "demand, commercial value, or uniqueness from the image."
    )


@router.get("/api/critique", response_model=MediaCritiqueResponse, response_model_exclude_unset=True)
async def api_critique(
    path: str = Query(...),
    mode: str = Query("rule"),
    lang: Optional[str] = Query(None),
    refresh: bool = Query(False),
    user: Optional[CurrentUser] = Depends(get_optional_user),
):
    """Get AI critique for a photo's score.

    Modes:
      - rule: Fast rule-based analysis (always available)
      - vlm: VLM-powered natural language critique (requires GPU + a profile
        with a VLM tagging model). The generated text is cached per photo;
        ``refresh=true`` regenerates it. The generated response follows the
        requested viewer language in ``lang``.
    """
    if not VIEWER_CONFIG.get('features', {}).get('show_critique', True):
        raise HTTPException(status_code=403, detail="Critique feature is disabled")

    async with get_async_db() as conn:
        user_id = user.user_id if user else None
        vis_sql, vis_params = get_visibility_clause(user_id)

        # Select only columns needed for critique (avoid loading BLOB fields).
        col_str = ', '.join(_CRITIQUE_COLUMNS)
        cur = await conn.execute(
            f"SELECT {col_str} FROM photos WHERE path = ? AND {vis_sql}",
            [path] + vis_params
        )
        photo = await cur.fetchone()
        await cur.close()

        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")

        photo = dict(photo)
        result = _build_rule_critique(photo)

        if mode == 'vlm':
            await _attach_vlm_critique(conn, photo, result, lang, refresh, user)

        return result


@router.get(
    "/api/personalized_suggestions",
    response_model=MediaPersonalizedSuggestionsResponse,
    response_model_exclude_unset=True,
)
async def api_personalized_suggestions(
    path: str = Query(...),
    lang: Optional[str] = Query(None),
    refresh: bool = Query(False),
    user: Optional[CurrentUser] = Depends(get_optional_user),
):
    """Get score-context-aware personalized suggestions for one photo.

    Cached suggestions can be read without edition authentication. Generating or
    refreshing them is intentionally separate from the existing full VLM critique.
    """
    if not VIEWER_CONFIG.get('features', {}).get('show_critique', True):
        raise HTTPException(status_code=403, detail="Critique feature is disabled")

    requested_lang = lang or 'en'
    async with get_async_db() as conn:
        user_id = user.user_id if user else None
        vis_sql, vis_params = get_visibility_clause(user_id)
        col_str = ', '.join(_PERSONALIZED_COLUMNS)
        cur = await conn.execute(
            f"SELECT {col_str} FROM photos WHERE path = ? AND {vis_sql}",
            [path] + vis_params,
        )
        row = await cur.fetchone()
        await cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="Photo not found")

        photo = dict(row)
        aggregate_score = _score_value(photo.get('aggregate'))
        vcg_score = _score_value(photo.get('vcg_submission_score'))
        if aggregate_score is None or vcg_score is None:
            return _personalized_response(
                photo,
                'unavailable',
                reason='score_unavailable',
            )

        existing_cols = get_existing_columns()
        can_cache = 'personalized_suggestions' in existing_cols
        cached = None
        if can_cache:
            cur = await conn.execute(
                "SELECT personalized_suggestions FROM photos WHERE path = ?",
                [path],
            )
            cache_row = await cur.fetchone()
            await cur.close()
            if cache_row:
                cached = _read_personalized_cache(
                    cache_row['personalized_suggestions'], photo, requested_lang,
                )

        if cached and not refresh:
            return _personalized_response(
                photo,
                'cached',
                cached,
                generated_at=cached['generated_at'],
            )

        if not is_edition_authenticated(user):
            if cached:
                return _personalized_response(
                    photo,
                    'cached',
                    cached,
                    generated_at=cached['generated_at'],
                    warning='edition_required',
                )
            return _personalized_response(photo, 'unavailable', reason='edition_required')

        if not VIEWER_CONFIG.get('features', {}).get('show_vlm_critique', False):
            if cached:
                return _personalized_response(
                    photo,
                    'cached',
                    cached,
                    generated_at=cached['generated_at'],
                    warning='vlm_unavailable',
                )
            return _personalized_response(photo, 'unavailable', reason='vlm_unavailable')

        if not resolve_personalized_vlm_config():
            if cached:
                return _personalized_response(
                    photo,
                    'cached',
                    cached,
                    generated_at=cached['generated_at'],
                    warning='vlm_unavailable',
                )
            return _personalized_response(photo, 'unavailable', reason='vlm_unavailable')

        rule_critique = _build_rule_critique(photo)
        cur = await conn.execute("SELECT thumbnail FROM photos WHERE path = ?", [path])
        thumbnail_row = await cur.fetchone()
        await cur.close()
        thumbnail = thumbnail_row['thumbnail'] if thumbnail_row else None
        generated = await asyncio.to_thread(
            _get_personalized_suggestions,
            photo,
            rule_critique,
            thumbnail,
            requested_lang,
        )
        if not generated:
            if cached:
                return _personalized_response(
                    photo,
                    'cached',
                    cached,
                    generated_at=cached['generated_at'],
                    warning='refresh_failed',
                )
            return _personalized_response(photo, 'unavailable', reason='generation_failed')

        generated_at = datetime.now(timezone.utc).isoformat()
        cache_payload = _make_personalized_cache_payload(
            photo, requested_lang, generated, generated_at,
        )
        if can_cache:
            await conn.execute(
                "UPDATE photos SET personalized_suggestions = ? WHERE path = ?",
                [json.dumps(cache_payload, ensure_ascii=False), path],
            )
            await conn.commit()

        return _personalized_response(
            photo,
            'generated',
            generated,
            generated_at=generated_at,
        )


async def _attach_vlm_critique(conn, photo, result, lang, refresh, user):
    """Attach a cached or freshly generated VLM critique to the rule result."""
    path = photo['path']
    requested_lang = lang or 'en'
    existing_cols = get_existing_columns()
    can_cache = 'vlm_critique' in existing_cols
    has_language_cache = 'vlm_critique_language' in existing_cols
    text = None
    source = 'cached'

    if can_cache and not refresh:
        language_column = ', vlm_critique_language' if has_language_cache else ''
        cur = await conn.execute(
            f"SELECT vlm_critique, vlm_critique_translated{language_column} "
            "FROM photos WHERE path = ?", [path]
        )
        row = await cur.fetchone()
        await cur.close()
        if row:
            cached_language = row['vlm_critique_language'] if has_language_cache else None
            text = _read_vlm_cache(row['vlm_critique'], cached_language, requested_lang)
            if not text:
                # Preserve the old configured-target translation as a
                # compatibility fallback for pre-language-marker databases.
                target_lang = translation_target(requested_lang)
                if target_lang and row['vlm_critique_translated']:
                    text = row['vlm_critique_translated']

    if not text:
        if not is_edition_authenticated(user):
            result['vlm_available'] = False
            return

        cur = await conn.execute("SELECT thumbnail FROM photos WHERE path = ?", [path])
        row = await cur.fetchone()
        await cur.close()
        thumbnail = row['thumbnail'] if row else None
        # VLM inference is GPU/CPU-bound and blocking — run it off the
        # event loop so it never stalls other async requests.
        text = await asyncio.to_thread(
            _get_vlm_critique, photo, result, thumbnail, requested_lang,
        )
        source = 'generated'
        if text and can_cache:
            assignments = [
                "vlm_critique = ?",
                "vlm_critique_translated = NULL",
            ]
            params = [text]
            if has_language_cache:
                assignments.append("vlm_critique_language = ?")
                params.append(requested_lang)
            params.append(path)
            await conn.execute(
                f"UPDATE photos SET {', '.join(assignments)} WHERE path = ?",
                params,
            )
            await conn.commit()

    if not text:
        result['vlm_available'] = False
        return

    result['vlm_critique'] = text
    result['vlm_source'] = source


_DEFAULT_VLM_PROMPT = (
    "You are an expert photography critic reviewing a {category} photograph "
    "that Facet's metrics scored {aggregate}/10 overall.\n"
    "Measured metrics (0-10):\n{breakdown}\n"
    "{penalties}"
    "Camera settings: {exif}.\n\n"
    "Judge the picture from what you actually see. Treat the metrics as context "
    "to confirm or contradict from the pixels, never restate the numbers. First "
    "perceive the scene, then register the feeling it evokes, then judge it, then "
    "advise. Write exactly three titled sections in compact prose, no bullet "
    "lists:\n"
    "Observation: one or two sentences naming the subject, the moment and the "
    "framing you see.\n"
    "Assessment: three to four sentences giving a short verdict on each of "
    "composition, color & light, focus/depth-of-field & technical execution, and "
    "subject & moment; say where the pixels back the metrics or contradict them, "
    "and what the image makes a viewer feel.\n"
    "Suggestions: at most three concrete fixes for the weakest dimensions above, "
    "mixing shooting and editing advice."
)


def _build_vlm_prompt(rule_critique, photo, lang='en', full_config=None):
    """Fill the configured critique prompt template with the rule breakdown and EXIF."""
    full_config = _FULL_CONFIG if full_config is None else full_config
    vlm_cfg = full_config.get('critique', {}).get('vlm', {})
    template = vlm_cfg.get('prompt_template') or _DEFAULT_VLM_PROMPT

    lines = []
    for item in rule_critique.get('breakdown', [])[:12]:
        suffix = ', lower is better' if item['metric_key'] == 'noise_sigma' else ''
        lines.append(f"- {item['metric']}: {item['value']} (weight {item['weight']:.0%}{suffix})")
    breakdown = '\n'.join(lines) or '- no per-metric data available'

    penalty_keys = list(rule_critique.get('penalties', {}))
    penalties = f"Penalties applied: {', '.join(penalty_keys)}.\n" if penalty_keys else ''

    exif_parts = []
    if photo.get('f_stop'):
        exif_parts.append(f"f/{photo['f_stop']}")
    if photo.get('shutter_speed'):
        exif_parts.append(f"{photo['shutter_speed']}s")
    if photo.get('iso'):
        exif_parts.append(f"ISO {photo['iso']}")
    if photo.get('focal_length'):
        exif_parts.append(f"{photo['focal_length']}mm")
    exif = ', '.join(exif_parts) or 'unknown'

    aggregate = rule_critique.get('aggregate')
    prompt = template.format(
        category=rule_critique.get('category', 'photo'),
        aggregate=f"{aggregate:.1f}" if aggregate is not None else 'unscored',
        breakdown=breakdown,
        penalties=penalties,
        exif=exif,
    )
    language_names = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'zh': 'Simplified Chinese',
    }
    language_name = language_names.get(lang, lang)
    return (
        f"{prompt}\n\nRespond entirely in {language_name} ({lang}), "
        "including the section headings. Do not switch to another language."
    )


def _load_critique_image(path, thumbnail_bytes):
    """Build the PIL image fed to the VLM: stored thumbnail, else decoded original.

    The stored 640px thumbnail is preferred — it exists for every scored photo
    including RAW files, which PIL cannot decode from disk.
    """
    from io import BytesIO

    from PIL import Image

    if thumbnail_bytes:
        return Image.open(BytesIO(thumbnail_bytes)).convert('RGB')

    from api.path_validation import resolve_photo_disk_path
    from utils.image_loading import open_nonraw_image

    disk_path = resolve_photo_disk_path(path)
    img = open_nonraw_image(disk_path)
    img.thumbnail((640, 640))
    return img


def _generate_vlm_critique(tagger, photo, rule_critique, image, lang='en', full_config=None):
    """Generate one VLM critique using an already-resolved tagger."""
    full_config = _FULL_CONFIG if full_config is None else full_config
    prompt = _build_vlm_prompt(rule_critique, photo, lang, full_config)
    max_new_tokens = int(
        full_config.get('critique', {}).get('vlm', {}).get('max_new_tokens', 320)
    )
    with vlm_generate_lock:
        response = tagger.generate(image, prompt, max_new_tokens=max_new_tokens)
    response = (response or '').strip()
    return response or None


def _generate_personalized_suggestions(tagger, photo, rule_critique, image,
                                       lang='en', full_config=None):
    """Generate validated suggestions using an already-resolved tagger."""
    full_config = _FULL_CONFIG if full_config is None else full_config
    prompt = _build_personalized_prompt(photo, rule_critique, lang, full_config)
    vlm_settings = (full_config.get('critique', {}).get('vlm') or {})
    max_new_tokens = int(
        vlm_settings.get('personalized_max_new_tokens', vlm_settings.get('max_new_tokens', 384))
    )
    with vlm_generate_lock:
        response = tagger.generate(image, prompt, max_new_tokens=max_new_tokens)
    return _parse_personalized_suggestions(response)


def _get_vlm_critique(photo, rule_critique, thumbnail_bytes, lang='en'):
    """Generate a VLM critique; None when the VLM is unavailable or fails."""
    try:
        if not VIEWER_CONFIG.get('features', {}).get('show_vlm_critique', False):
            return None

        vlm_config = resolve_vlm_config()
        if not vlm_config:
            return None

        tagger = get_or_load_vlm_tagger(vlm_config)
        img = _load_critique_image(photo['path'], thumbnail_bytes)
        return _generate_vlm_critique(
            tagger, photo, rule_critique, img, lang, _FULL_CONFIG,
        )

    except Exception:
        logger.exception("VLM critique failed")
        return None


def _get_personalized_suggestions(photo, rule_critique, thumbnail_bytes, lang='en'):
    """Generate and validate structured personalized suggestions."""
    try:
        if not VIEWER_CONFIG.get('features', {}).get('show_vlm_critique', False):
            return None

        vlm_config = resolve_personalized_vlm_config()
        if not vlm_config:
            return None

        tagger = get_or_load_vlm_tagger(vlm_config)
        img = _load_critique_image(photo['path'], thumbnail_bytes)
        return _generate_personalized_suggestions(
            tagger, photo, rule_critique, img, lang, _FULL_CONFIG,
        )
    except Exception:
        logger.exception("Personalized suggestions generation failed")
        return None
