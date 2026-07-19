# CLAUDE.md

This file provides guidance to Claude Code when working with code in this
repository. It is shared and committed — keep it limited to facts true for
anyone working in this repo. Personal working-style preferences belong in
`CLAUDE.local.md` instead (gitignored, not this file).

## What this project is

**KickCoach** — an AI football (soccer) coaching app: upload a video of your
kick, get biomechanical analysis and personalized coaching feedback.

Pipeline: a TFLite pose-estimation model (`3.tflite`, MoveNet-style, 17
keypoints) extracts biomechanical features at the moment of the kick (knee
angle, trunk lean, hip rotation, ankle speed, knee angular velocity) → the
snapshot is compared against a pro reference profile matched to the player's
age/level/position/kick type → Google Gemini generates context-aware coaching
feedback and drills → the session is saved to CSV history for trend tracking.

## Current focus

[What is the active area of work right now? Which surface or feature set?]

**In scope now:**
- [Feature / screen / module A]
- [Feature / screen / module B]

**Parked — do not edit this sprint:**
- [Module or file that is done/frozen/belongs to a later phase]

## How to treat the plan in PROJECT_STATE.md

- Phase sequencing is fixed; don't reorder it.
- Specific technical implementation choices are defaults, not mandates. If
  you see a better approach, say so explicitly and wait for a decision before
  doing it.
- When in doubt: a working, secure implementation beats matching the plan's
  suggested detail to the letter.

## Commands

```bash
# Backend (FastAPI) — requires GEMINI_API_KEY in .env for /feedback
uvicorn app:app --reload --port 8000

# Frontend — index.html is a static page; open it directly or serve it
# (VSCode Live Server is configured on port 5501). It calls the API at
# http://127.0.0.1:8000 by default (editable in the page's API-base field).
```

There is currently **no requirements.txt, no test suite, and no linter**.
Dependencies (installed manually so far): fastapi, uvicorn, httpx,
python-dotenv, opencv-python, numpy, tensorflow.

## Stack

- **Backend:** Python 3.11/3.13 · FastAPI + Starlette middleware · uvicorn
- **ML:** TensorFlow Lite interpreter (`3.tflite` pose model) · OpenCV · NumPy
- **LLM:** Google Gemini (`gemini-2.5-flash`) via raw REST calls in `llm_client.py`
- **Frontend:** single static `index.html`, vanilla JS, no build step
  (`index2.html` is a minimal API test harness)
- **Storage:** flat CSV files (`kick_history.csv` session history,
  `final_kick_features.csv` extracted kick features) — no database

## Layout (flat, single-directory)

- `app.py` — FastAPI app: middleware, routes (`/analyze`, `/feedback`,
  `/profiles`, `/history`, health checks), CSV history helpers
- `pose_engine.py` — model loading, per-frame inference, feature extraction,
  kick detection, annotated-frame export; tunable via env vars
- `feedback_engine.py` — deviation computation vs. pro reference, severity
  thresholds, prompt building, orchestrates the LLM call
- `reference_data.py` — `PRO_PROFILES` archetypes, `select_profile()`,
  feature metadata, drill library
- `llm_client.py` — thin async Gemini wrapper; all LLM config lives here
- `index.html` — the real frontend UI

## Configuration

- `.env` (gitignored): `GEMINI_API_KEY` — required for `/feedback`.
- Pose tuning env vars (all optional, defaults in `pose_engine.py`):
  `MODEL_PATH`, `CONF_THRESH`, `KICKING_LEG`, `KICK_THRESHOLD_PPS`,
  `KNEE_ANG_VEL_THRESHOLD`, `KICK_COOLDOWN_SECS`, `SMOOTH_ALPHA`,
  `FPS_ESTIMATE`, `MAX_KICKS_TO_TRACK`.

## Key docs (read when relevant — don't load all by default)

- `docs/PROJECT_STATE.md` — live status, current phase, open questions. Check first.
- `docs/CONTEXT.md` — decisions made and why, conventions, deferred items.

## Conventions

- Flat module layout — one file per concern (see Layout above); no packages.
- Section banners (`# ───────`) divide each file into CONFIG / HELPERS /
  ROUTES etc. — keep new code under the right banner.
- Comments explain *why*, not *what*.
- Config comes from env vars with defaults, read at module top, never
  hardcoded mid-function.
- The five feature names (`knee_angle`, `trunk`, `hip_rotation`,
  `ankle_speed_pps`, `knee_ang_vel_dps`) are a shared contract across
  `pose_engine`, `reference_data`, `feedback_engine`, the CSVs, and the
  frontend — renaming one is a multi-file operation.

## When stuck

- Status / "what's done, what's next" → check `docs/PROJECT_STATE.md` first.
- Stack question → Python/FastAPI + vanilla-JS static frontend. Don't
  introduce new frameworks (React, databases, build tools) without flagging it.
