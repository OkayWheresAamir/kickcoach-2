# PROJECT_STATE.md

Live status for the current sprint. **Check this first** before starting any
session. Update it as things land or change — stale state here is worse than
no state.

_Last updated: 2026-07-19_

---

## Current phase

**Phase 1 — Shippable** — make the existing pipeline installable, tested,
hardened, and deployed as one unit. No new features until this lands.

---

## Upgrade plan (high level)

Sequence is fixed: **Shippable → Analysis quality → Data & accounts.**
Each numbered step is roughly one focused session; each phase gets its own
detailed planning chat before work starts.

### Phase 1 — Shippable

Goal: anyone can clone, install, run, and reach a hosted URL that works.

1. **Reproducible setup** — dependency manifest (requirements.txt or
   pyproject), README with setup/run steps, fix `.gitignore` and untrack
   `__pycache__/`, `.DS_Store`, runtime CSVs; decide how the 9 MB model file
   is distributed.
2. **Smoke tests + CI** — pytest covering health/profiles/history and an
   `/analyze` run against a small fixture video; GitHub Actions running
   tests + a linter (ruff).
3. **Hardening pass** — CORS allowlist instead of `*`, config cleanup,
   consistent error responses, remove stale comments/dead files
   (`index2.html`, `app.py` header).
4. **One deployable unit** — serve `index.html` from FastAPI static files,
   Dockerfile.
5. **Deploy** — pick a host (Railway / Render / Fly), Gemini key via env,
   verify the demo path on the live URL.

### Phase 2 — Analysis quality

Goal: kick detection and feedback we can measure, not just eyeball.
Evaluation is manual/ML-focused.

1. **Ground-truth eval set** — collect and hand-label a small set of kick
   videos (true kick frame, expected leg, plus known-garbage cases).
2. **Eval harness** — script that runs the pose engine over the eval set and
   reports kick-frame accuracy and feature plausibility; establishes the
   baseline before any tuning.
3. **Robustness fixes** — outlier rejection / sanity bounds on features
   (e.g. the 16,444°/s garbage row), read real FPS from video metadata
   instead of `FPS_ESTIMATE`, confidence gating.
4. **Pose model comparison** — try stronger models (MoveNet Thunder,
   MediaPipe BlazePose, RTMPose) and compare on the eval harness; keep the
   winner.
5. **Feedback quality** — validate severity thresholds against the eval set;
   schema-validate and spot-check Gemini output quality.

### Phase 3 — Data & accounts

Goal: real persistence and per-user experience.

1. **Database** — replace CSVs with a real DB (choice made in this phase's
   planning chat: likely SQLite first or Supabase/Postgres directly);
   migrate history schema.
2. **Accounts/auth** — user signup/login, sessions tied to users.
3. **Per-user history + trends** — history UI becomes per-player with
   progress over time.
4. **(Optional) media storage** — persist uploaded videos / annotated frames
   to object storage if needed.

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

1. Phase 1.1 — reproducible setup (deps manifest, README, .gitignore cleanup,
   model-file distribution decision)
2. Phase 1.2 — smoke tests + CI
3. Phase 1.3 — hardening pass

---

## Open questions / decisions needed

- [ ] Dependency tooling: plain `requirements.txt` vs `pyproject.toml` + uv
  (decide at start of Phase 1.1)
- [ ] Model file distribution: Git LFS vs download script vs keep in repo
  (Phase 1.1)
- [ ] Deployment host: Railway vs Render vs Fly (Phase 1.5)
- [ ] Database choice: SQLite-first vs Supabase/Postgres directly (Phase 3
  planning chat)

---

## Parked / later phases

- **Analysis quality work** (pose model swaps, threshold tuning, eval
  harness) — parked until Phase 2. Don't tune `pose_engine.py` detection
  logic during Phase 1.
- **Database & accounts** — parked until Phase 3. CSVs stay as-is through
  Phases 1–2; don't build schema/auth early.
- **Frontend rewrite/framework** — not planned in any phase; `index.html`
  stays vanilla JS unless explicitly reopened.

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
