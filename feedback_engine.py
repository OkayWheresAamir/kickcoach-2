# feedback_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Computes per-feature deviations between a user's kick snapshot and the
# dynamically selected PRO_PROFILE, builds a structured prompt, calls the LLM,
# and returns a clean feedback payload ready for the frontend.
#
# Changes vs original:
#   - Dynamic profile selection based on player context (age, level, position, kick_type)
#   - Drill library lookup from reference_data instead of relying solely on LLM
#   - Richer prompt that includes coaching cues, player context, and drill suggestions
#   - Trend analysis when multiple sessions are available
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Any

from reference_data import (
    FEATURE_META,
    select_profile,
    get_relevant_drills,
)
from llm_client import call_gemini

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEVIATION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "knee_angle":        {"moderate": 0.05,  "significant": 0.12},
    "trunk":             {"moderate": 0.15,  "significant": 0.35},
    "knee_ang_vel_dps":  {"moderate": 0.10,  "significant": 0.25},
}

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _severity(feature: str, pct_diff: float) -> str:
    thresholds = SEVERITY_THRESHOLDS.get(feature, {"moderate": 0.10, "significant": 0.25})
    abs_diff   = abs(pct_diff)
    if abs_diff <= thresholds["moderate"]:
        return "good"
    if abs_diff <= thresholds["significant"]:
        return "moderate"
    return "significant"


def _compute_deviations(snapshot: dict[str, Any], pro_reference: dict[str, float]) -> list[dict]:
    """
    Compare every feature in snapshot against the chosen pro_reference.
    Returns a list of deviation dicts sorted by severity (worst first).
    """
    deviations = []

    for key, pro_val in pro_reference.items():
        user_val = snapshot.get(key)
        meta     = FEATURE_META.get(key, {})

        if user_val is None:
            deviations.append({
                "feature":    key,
                "label":      meta.get("label", key),
                "unit":       meta.get("unit", ""),
                "user_value": None,
                "pro_value":  pro_val,
                "abs_diff":   None,
                "pct_diff":   None,
                "direction":  "unavailable",
                "severity":   "unavailable",
                "hint":       meta.get("hint", ""),
                "coaching_cue": meta.get("coaching_cue", ""),
            })
            continue

        abs_diff = user_val - pro_val
        pct_diff = (abs_diff / pro_val) if abs(pro_val) > 1e-9 else 0.0

        direction = (
            "above_pro"  if abs_diff > 0
            else "below_pro" if abs_diff < 0
            else "matches_pro"
        )

        deviations.append({
            "feature":      key,
            "label":        meta.get("label", key),
            "unit":         meta.get("unit", ""),
            "user_value":   round(float(user_val), 4),
            "pro_value":    round(float(pro_val),  4),
            "abs_diff":     round(float(abs_diff), 4),
            "pct_diff":     round(float(pct_diff), 4),
            "direction":    direction,
            "severity":     _severity(key, pct_diff),
            "hint":         meta.get("hint", ""),
            "coaching_cue": meta.get("coaching_cue", ""),
        })

    _order = {"significant": 0, "moderate": 1, "good": 2, "unavailable": 3}
    deviations.sort(key=lambda d: _order.get(d["severity"], 9))
    return deviations


def _build_prompt(
    snapshot: dict,
    deviations: list[dict],
    profile: dict,
    player_context: dict,
    suggested_drills: list[dict],
) -> str:
    """
    Build a rich, context-aware plain-text prompt for the LLM.
    """
    ctx = player_context
    position  = ctx.get("position", "unspecified")
    kick_type = ctx.get("kick_type", "unspecified")
    age       = ctx.get("age", "unspecified")
    level     = ctx.get("level", "unspecified")
    goals     = ctx.get("goals", "improve overall kicking technique")

    lines = [
        "You are KickCoach, an expert football/soccer biomechanics coach.",
        "You coach players at every level from grassroots juniors to professional academies.",
        "Your feedback is honest, motivating, specific, and actionable — like a real pitch-side coach.",
        "",
        "═══ PLAYER CONTEXT ═══",
        f"Position  : {position}",
        f"Age       : {age}",
        f"Level     : {level}",
        f"Kick type : {kick_type}",
        f"Goals     : {goals}",
        f"Reference : {profile.get('label', 'Pro')} — {profile.get('description', '')}",
        "",
        "═══ BIOMECHANICAL COMPARISON vs REFERENCE ═══",
    ]

    for d in deviations:
        if d["severity"] == "unavailable":
            lines.append(f"• {d['label']}: data unavailable for this frame.")
        else:
            direction_str = (
                f"{abs(d['pct_diff']*100):.1f}% {'above' if d['direction']=='above_pro' else 'below'} reference"
                if d["direction"] != "matches_pro"
                else "matches reference exactly"
            )
            lines.append(
                f"• {d['label']}: player = {d['user_value']} {d['unit']} | "
                f"reference = {d['pro_value']} {d['unit']} | "
                f"{direction_str} | severity: {d['severity'].upper()}"
            )
            lines.append(f"  Biomechanical context: {d['hint']}")
            if d.get("coaching_cue"):
                lines.append(f"  Coaching cue: {d['coaching_cue']}")

    # Include pre-computed drill suggestions for the LLM to refine
    if suggested_drills:
        lines += ["", "═══ SUGGESTED DRILLS FROM LIBRARY (refine and personalise) ═══"]
        for i, drill in enumerate(suggested_drills, 1):
            lines.append(
                f"{i}. {drill['name']} — addresses: {drill.get('addressing_feature', '')} — "
                f"{drill['sets_reps']}"
            )
            lines.append(f"   Base description: {drill['description']}")

    lines += [
        "",
        "═══ YOUR TASK ═══",
        f"Write feedback as if you are speaking directly to this {age}-year-old {position} on the training pitch.",
        "Be encouraging but direct — this player is passionate and wants real coaching.",
        "Reference their specific position and goals in your feedback where relevant.",
        "",
        "Respond with a JSON object (no markdown, no code fences) with exactly this structure:",
        "{",
        '  "overall_rating": <integer 1-10, where 10 = matches reference exactly>,',
        '  "summary": "<2-3 sentence plain-English summary tailored to their position/level>",',
        '  "strengths": ["<specific strength with a brief why>", ...],',
        '  "areas_to_improve": [',
        '    {',
        '      "area": "<short biomechanical name>",',
        '      "explanation": "<why it matters for their position and how to fix it>",',
        '      "severity": "<good|moderate|significant>",',
        '      "coaching_cue": "<one memorable phrase they can think of mid-kick>"',
        '    }',
        '  ],',
        '  "drills": [',
        '    {',
        '      "name": "<drill name>",',
        '      "targets": ["<feature it improves>"],',
        '      "description": "<clear step-by-step 2-4 sentences, personalised to position>",',
        '      "sets_reps": "<e.g. 3 sets × 10 reps>",',
        '      "progression": "<how to make it harder next week>"',
        '    }',
        '  ],',
        '  "pro_tip": "<one memorable motivational sentence specific to their position/goal>",',
        '  "next_session_focus": "<the single most important thing to work on next session>"',
        "}",
        "",
        "Rules:",
        "- 2-4 drills maximum. Prioritise the most significant deviations.",
        "- Reference the player's position (e.g. 'As a striker...', 'For a winger...').",
        "- Keep explanations jargon-light but technically precise.",
        "- The coaching_cue in each area_to_improve must be 10 words or fewer.",
        "- Do NOT include markdown, code fences, or any text outside the JSON object.",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def generate_feedback(
    snapshot: dict[str, Any],
    player_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Full feedback pipeline:
      1. Select the appropriate pro profile for this player.
      2. Compute deviations vs selected reference.
      3. Look up relevant drills from the library.
      4. Build context-rich LLM prompt.
      5. Call Gemini.
      6. Return structured payload.

    Parameters
    ----------
    snapshot       : Output of ``analyze_video_file()``.
    player_context : Optional dict with keys:
                       age (int), level (str), position (str),
                       kick_type (str), goals (str)

    Returns
    -------
    dict with keys: deviations, feedback, profile_used
    """
    if player_context is None:
        player_context = {}

    # ── Step 1: select profile ────────────────────────────────────────────────
    profile = select_profile(
        age       = player_context.get("age"),
        level     = player_context.get("level"),
        position  = player_context.get("position"),
        kick_type = player_context.get("kick_type"),
    )
    pro_reference = profile["reference"]
    logger.info(
        "Profile selected: %s for context: %s",
        profile.get("label"), player_context
    )

    # ── Step 2: deviations ────────────────────────────────────────────────────
    logger.info("Computing deviations for snapshot at frame %s", snapshot.get("frame_number"))
    deviations = _compute_deviations(snapshot, pro_reference)

    # ── Step 3: drill lookup ──────────────────────────────────────────────────
    suggested_drills = get_relevant_drills(
        deviations,
        position = player_context.get("position"),
        level    = player_context.get("level"),
        max_drills = 4,
    )

    # ── Step 4: build prompt ──────────────────────────────────────────────────
    prompt = _build_prompt(snapshot, deviations, profile, player_context, suggested_drills)
    logger.debug("Prompt built (%d chars), calling Gemini…", len(prompt))

    # ── Step 5: call LLM ──────────────────────────────────────────────────────
    feedback_json = await call_gemini(prompt)

    # ── Step 6: return structured payload ─────────────────────────────────────
    return {
        "frame_number": snapshot.get("frame_number"),
        "deviations":   deviations,
        "feedback":     feedback_json,
        "profile_used": {
            "key":         profile.get("label"),
            "description": profile.get("description"),
            "level":       profile.get("level"),
        },
    }