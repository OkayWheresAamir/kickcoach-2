# reference_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Dynamic reference data for multiple player archetypes.
# Instead of a single hardcoded pro, we now match the user to the closest
# archetype based on their profile (age, level, position, kick type).
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PRO REFERENCE PROFILES
# Each profile represents a distinct player archetype.
# Values sourced from published biomechanics literature and motion-capture data.
# ─────────────────────────────────────────────────────────────────────────────

PRO_PROFILES = {

    # ── Elite senior – instep power shot ─────────────────────────────────────
    "elite_power": {
        "label":       "Elite Power Striker",
        "description": "Optimised for maximum ball velocity — long shots, volleys.",
        "positions":   ["striker", "midfielder", "defender"],
        "kick_types":  ["power", "long_pass", "shot"],
        "age_range":   (18, 40),
        "level":       "elite",
        "reference": {
            "knee_angle":         139.60,   # ° near full extension at contact
            "trunk":               10.37,   # ° lean toward kicking side
            "hip_rotation":        -4.09,   # ° slight counter-rotation
            "ankle_speed_pps":   2227.17,   # px/s very high foot speed
            "knee_ang_vel_dps":  1689.65,   # °/s explosive knee snap
        },
    },

    # ── Elite senior – finesse / placed shot ─────────────────────────────────
    "elite_finesse": {
        "label":       "Elite Finesse Shooter",
        "description": "Curl and accuracy focused — cut-ins, placed shots.",
        "positions":   ["winger", "midfielder", "striker"],
        "kick_types":  ["finesse", "curl", "pass"],
        "age_range":   (18, 40),
        "level":       "elite",
        "reference": {
            "knee_angle":         128.50,
            "trunk":               14.20,
            "hip_rotation":         8.30,
            "ankle_speed_pps":   1850.00,
            "knee_ang_vel_dps":  1320.00,
        },
    },

    # ── Semi-pro / advanced adult ─────────────────────────────────────────────
    "advanced_adult": {
        "label":       "Advanced Adult Player",
        "description": "Amateur to semi-pro level — realistic achievable targets.",
        "positions":   ["any"],
        "kick_types":  ["power", "pass", "shot"],
        "age_range":   (18, 50),
        "level":       "advanced",
        "reference": {
            "knee_angle":         132.00,
            "trunk":                9.00,
            "hip_rotation":        -3.00,
            "ankle_speed_pps":   1750.00,
            "knee_ang_vel_dps":  1300.00,
        },
    },

    # ── Youth U17-U21 ─────────────────────────────────────────────────────────
    "youth_senior": {
        "label":       "Youth Senior (U17–U21)",
        "description": "Academy / youth elite benchmarks for teenage players.",
        "positions":   ["any"],
        "kick_types":  ["power", "pass", "shot"],
        "age_range":   (15, 21),
        "level":       "youth",
        "reference": {
            "knee_angle":         130.00,
            "trunk":                8.50,
            "hip_rotation":        -2.50,
            "ankle_speed_pps":   1580.00,
            "knee_ang_vel_dps":  1150.00,
        },
    },

    # ── Youth U12-U16 ─────────────────────────────────────────────────────────
    "youth_junior": {
        "label":       "Youth Junior (U12–U16)",
        "description": "Junior academy targets — developmentally appropriate.",
        "positions":   ["any"],
        "kick_types":  ["power", "pass", "shot"],
        "age_range":   (10, 15),
        "level":       "youth",
        "reference": {
            "knee_angle":         122.00,
            "trunk":                7.00,
            "hip_rotation":        -1.50,
            "ankle_speed_pps":   1200.00,
            "knee_ang_vel_dps":   900.00,
        },
    },

    # ── Recreational / beginner adult ────────────────────────────────────────
    "recreational": {
        "label":       "Recreational Adult",
        "description": "Good club-level targets for weekend players.",
        "positions":   ["any"],
        "kick_types":  ["power", "pass"],
        "age_range":   (16, 60),
        "level":       "recreational",
        "reference": {
            "knee_angle":         120.00,
            "trunk":                7.50,
            "hip_rotation":        -2.00,
            "ankle_speed_pps":   1100.00,
            "knee_ang_vel_dps":   850.00,
        },
    },
}

# Default fallback
DEFAULT_PROFILE_KEY = "advanced_adult"

# Backward-compat single reference
PRO_REFERENCE = PRO_PROFILES["elite_power"]["reference"]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE META  (extended with coaching cues)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_META = {
    "knee_angle": {
        "label":        "Knee Angle at Contact",
        "unit":         "°",
        "hint": (
            "Angle at the knee joint (hip–knee–ankle) at ball contact. "
            "Elite strikers achieve ~135-145° — almost straight but not locked. "
            "Too bent = power loss; fully locked = injury risk."
        ),
        "coaching_cue": "Drive your knee through the ball like a piston.",
        "priority":     1,
    },
    "trunk": {
        "label":        "Trunk Tilt",
        "unit":         "°",
        "hint": (
            "Lateral lean of the torso at contact. A slight lean over the ball (~8-12°) "
            "keeps shots low and adds power. Over-leaning causes the ball to rise."
        ),
        "coaching_cue": "Stay over the ball — imagine a string pulling your chest forward.",
        "priority":     2,
    },
    "hip_rotation": {
        "label":        "Hip Rotation",
        "unit":         "°",
        "hint": (
            "How much your shoulders rotate relative to your hips at contact. "
            "A small negative value means hips lead shoulders — the kinetic chain "
            "that transfers power from your core to the ball."
        ),
        "coaching_cue": "Open your hips early, then snap them through at contact.",
        "priority":     2,
    },
    "ankle_speed_pps": {
        "label":        "Ankle Speed",
        "unit":         "px/s",
        "hint": (
            "Linear speed of the ankle at ball contact. This is the single biggest "
            "predictor of ball velocity. Elite players exceed 2000 px/s."
        ),
        "coaching_cue": "Whip your foot through — think of cracking a whip, not pushing.",
        "priority":     1,
    },
    "knee_ang_vel_dps": {
        "label":        "Knee Angular Velocity",
        "unit":         "°/s",
        "hint": (
            "How fast the knee extends through the swing phase. High values (>1500°/s) "
            "indicate the explosive snap that separates powerful kickers."
        ),
        "coaching_cue": "Generate the snap from your hip, let it travel down to your knee.",
        "priority":     1,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DRILL LIBRARY  (position + severity + level aware)
# ─────────────────────────────────────────────────────────────────────────────

DRILL_LIBRARY = {
    "knee_angle": [
        {
            "name":        "Pendulum Knee Snap",
            "targets":     ["knee_angle", "knee_ang_vel_dps"],
            "description": (
                "Stand next to a wall for balance. Swing your kicking leg back, then "
                "explosively snap it forward, focusing on extending the knee fully at the "
                "peak. Hold the extended position for 1 second."
            ),
            "sets_reps":   "4 sets × 12 reps",
            "level":       ["all"],
            "position":    ["all"],
            "phase":       "swing",
        },
        {
            "name":        "Resistance Band Knee Drive",
            "targets":     ["knee_angle", "ankle_speed_pps"],
            "description": (
                "Attach a resistance band just above the knee of your kicking leg. "
                "Drive through the kick against the band resistance, focusing on full "
                "extension. The band forces your muscles to work harder through the range."
            ),
            "sets_reps":   "3 sets × 8 reps each side",
            "level":       ["advanced", "elite"],
            "position":    ["all"],
            "phase":       "swing",
        },
    ],
    "ankle_speed_pps": [
        {
            "name":        "Towel Whip Drill",
            "targets":     ["ankle_speed_pps", "knee_ang_vel_dps"],
            "description": (
                "Hold a rolled towel at hip height and flick it rapidly using only your "
                "wrist. Then replicate that 'whipping' sensation with your kicking leg — "
                "think speed not power. Do 5 towel flicks then 5 kicks alternating."
            ),
            "sets_reps":   "5 rounds × 5+5",
            "level":       ["all"],
            "position":    ["all"],
            "phase":       "contact",
        },
        {
            "name":        "Speed Juggling",
            "targets":     ["ankle_speed_pps"],
            "description": (
                "Juggle the ball using rapid light touches — aim for speed of foot "
                "movement not height. Use the instep. Count touches in 30 seconds "
                "and try to beat your record each session."
            ),
            "sets_reps":   "4 × 30 seconds",
            "level":       ["all"],
            "position":    ["all"],
            "phase":       "contact",
        },
    ],
    "trunk": [
        {
            "name":        "Trunk-Over-Ball Pass",
            "targets":     ["trunk"],
            "description": (
                "Place a cone 20m away. Kick the ball keeping your chin over it. "
                "A training partner holds a stick at your chest height as a lean reference. "
                "If the ball rises above knee height you leaned back too much."
            ),
            "sets_reps":   "3 sets × 10 kicks",
            "level":       ["all"],
            "position":    ["all"],
            "phase":       "contact",
        },
        {
            "name":        "Medicine Ball Core Rotation",
            "targets":     ["trunk", "hip_rotation"],
            "description": (
                "Stand sideways to a wall with a 3-4kg medicine ball. Rotate your torso, "
                "throw the ball against the wall, catch and repeat in one fluid motion. "
                "Teaches the trunk-hip separation pattern used in kicking."
            ),
            "sets_reps":   "3 sets × 15 reps each side",
            "level":       ["advanced", "elite"],
            "position":    ["all"],
            "phase":       "preparation",
        },
    ],
    "hip_rotation": [
        {
            "name":        "Hip Gate Openers",
            "targets":     ["hip_rotation"],
            "description": (
                "Walk forward lifting each knee to hip height, then rotate the hip "
                "outward like opening a gate before placing the foot down. "
                "Exaggerate the rotation — this mobilises the hip capsule for kicking."
            ),
            "sets_reps":   "3 × 20m walks",
            "level":       ["all"],
            "position":    ["all"],
            "phase":       "preparation",
        },
        {
            "name":        "Lateral Band Kick",
            "targets":     ["hip_rotation"],
            "description": (
                "Attach a resistance band to your plant leg ankle and a fixed point "
                "to your side. Perform full kicks focusing on driving the hip through "
                "before the leg. Feel the hip lead the knee lead the ankle — the kinetic chain."
            ),
            "sets_reps":   "3 sets × 10 reps",
            "level":       ["advanced", "elite"],
            "position":    ["all"],
            "phase":       "swing",
        },
    ],
    "knee_ang_vel_dps": [
        {
            "name":        "Explosive Knee Snap Volleys",
            "targets":     ["knee_ang_vel_dps", "ankle_speed_pps"],
            "description": (
                "Have a partner toss balls at thigh height. Focus on the explosive "
                "snap of the knee through contact — exaggerate the speed, not the "
                "power. Record sessions and review in slow-mo to see the snap."
            ),
            "sets_reps":   "3 sets × 10 volleys",
            "level":       ["advanced", "elite"],
            "position":    ["striker", "midfielder", "winger"],
            "phase":       "contact",
        },
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# PROFILE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_profile(
    age: int | None = None,
    level: str | None = None,
    position: str | None = None,
    kick_type: str | None = None,
) -> dict:
    """
    Choose the most appropriate PRO_PROFILE for a given player context.

    Parameters
    ----------
    age       : Player age in years.
    level     : "recreational" | "youth" | "advanced" | "elite"
    position  : "striker" | "winger" | "midfielder" | "defender" | "goalkeeper"
    kick_type : "power" | "finesse" | "curl" | "pass" | "shot" | "long_pass"

    Returns
    -------
    dict  – one entry from PRO_PROFILES (includes "reference", "label", etc.)
    """
    # Youth age gate — always apply first
    if age is not None:
        if age <= 15:
            return PRO_PROFILES["youth_junior"]
        if age <= 21:
            return PRO_PROFILES["youth_senior"]

    # Finesse / curl preference for advanced+ players
    if kick_type in ("finesse", "curl") and level in ("elite", "advanced"):
        return PRO_PROFILES["elite_finesse"]

    # Level mapping
    if level == "elite":
        return PRO_PROFILES["elite_power"]
    if level == "advanced":
        return PRO_PROFILES["advanced_adult"]
    if level == "recreational":
        return PRO_PROFILES["recreational"]

    return PRO_PROFILES[DEFAULT_PROFILE_KEY]


def get_relevant_drills(
    deviations: list[dict],
    position: str | None = None,
    level: str | None = None,
    max_drills: int = 4,
) -> list[dict]:
    """
    Pick the most relevant drills from DRILL_LIBRARY based on the worst deviations.
    Returns up to *max_drills* drills, prioritising significant deviations.
    """
    severity_order = {"significant": 0, "moderate": 1, "good": 2, "unavailable": 9}
    sorted_devs = sorted(deviations, key=lambda d: severity_order.get(d["severity"], 9))

    seen_drills: set[str] = set()
    result: list[dict] = []

    for dev in sorted_devs:
        if len(result) >= max_drills:
            break
        feature = dev["feature"]
        drills  = DRILL_LIBRARY.get(feature, [])
        for drill in drills:
            if drill["name"] in seen_drills:
                continue
            # Level filter (["all"] = no restriction)
            if (
                level
                and drill.get("level") != ["all"]
                and level not in drill.get("level", ["all"])
            ):
                continue
            seen_drills.add(drill["name"])
            result.append({**drill, "addressing_feature": feature})
            if len(result) >= max_drills:
                break

    return result