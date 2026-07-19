# PROJECT_STATE.md

Live status for the current sprint. **Check this first** before starting any
session. Update it as things land or change — stale state here is worse than
no state.

_Last updated: 2026-07-19_

---

## Current phase

**[Phase name / number]** — [one-line description of what this phase is about.
The core pipeline works end-to-end; this repo was just set up for a structured
upgrade — define the phase here once the plan is agreed.]

---

## What's done

- [x] Pose engine (`pose_engine.py`) — TFLite MoveNet-style inference, dynamic
      kicking-leg detection, best-kick selection by ankle speed, annotated
      frame exported as base64 PNG, env-var config
- [x] FastAPI backend (`app.py`) — `/analyze`, `/feedback`, `/profiles`,
      `/history`, health checks; upload-size middleware; CORS (currently `*`)
- [x] Pro reference profiles (`reference_data.py`) — multiple archetypes
      (elite power/finesse, youth, etc.), profile selection by player context,
      drill library
- [x] Feedback engine (`feedback_engine.py`) — per-feature deviations vs.
      selected profile with severity thresholds, structured Gemini prompt
- [x] Gemini client (`llm_client.py`) — async REST wrapper with fence-stripping
      and truncated-JSON repair
- [x] Session history — appended to `kick_history.csv`, queried via `/history`
- [x] Frontend (`index.html`) — styled single-page UI: upload, player context
      form, profile dropdown, results display, history view

---

## In progress

| Task | Owner | Status | Blocker |
|------|-------|--------|---------|
| [Task] | [Person/Me] | [WIP / review / blocked] | [blocker or —] |

---

## Up next (this phase)

1. [Next task — be specific enough to act on]
2. [Next task]
3. [Next task]

---

## Open questions / decisions needed

- [ ] [Question that needs an answer before work can proceed]
- [ ] [Technical tradeoff that hasn't been decided yet]

---

## Parked / later phases

- **[Feature or module]** — parked until [phase / condition]. Do not touch.
- **[Feature or module]** — [why it's deferred]

---

## Known issues / tech debt

- No `requirements.txt` / lockfile — dependencies are undocumented and
  installed by hand.
- No tests of any kind.
- `.gitignore` only covers `.env` and `CLAUDE.local.md` — `__pycache__/`,
  `.DS_Store`, the 9 MB `3.tflite` model, and runtime CSVs
  (`kick_history.csv`, `final_kick_features.csv`) are all tracked in git.
- Storage is flat CSV with no schema versioning; history rows widen as
  features change (see `kick_history.csv` header drift).
- CORS is wide open (`allow_origins=["*"]` with credentials).
- `app.py`'s header comment says `routes.py` — leftover from a rename.
- `kick_history.csv` contains at least one garbage row (frame 19: impossible
  values like knee_ang_vel 16444°/s) — kick detection can misfire.
- No README.

---

## Demo path (definition of done for this phase)

Current working golden path (pre-upgrade baseline):

1. Start backend: `uvicorn app:app --reload --port 8000` (with
   `GEMINI_API_KEY` in `.env`).
2. Open `index.html` (Live Server, port 5501).
3. Upload a kick video, fill in age/level/position/kick type, submit.
4. See biomechanical metrics, deviations vs. the matched pro profile,
   Gemini coaching feedback + drills, and the session appear in history.
