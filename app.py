# routes.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application: middleware, route handlers.
#
# Changes vs original:
#   - /feedback now accepts optional player_context form fields
#     (age, level, position, kick_type, goals)
#   - /history endpoint returns a player's past sessions from CSV
#   - /profiles endpoint exposes available pro profiles to the frontend
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import json
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from pose_engine import (
    analyze_video_file,
    MAX_UPLOAD_BYTES,
    ALLOWED_MIME_PREFIX,
)
from feedback_engine import generate_feedback
from reference_data import PRO_PROFILES, select_profile

logger = logging.getLogger(__name__)

HISTORY_CSV = "kick_history.csv"

# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class MaxSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body: int):
        super().__init__(app)
        self.max_body = max_body

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > self.max_body:
                    raise HTTPException(status_code=413, detail="Upload too large")
            except ValueError:
                pass
        return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="KickCoach")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MaxSizeMiddleware, max_body=MAX_UPLOAD_BYTES)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _append_history(record: dict) -> None:
    """Append a session record to the history CSV."""
    write_header = not os.path.exists(HISTORY_CSV)
    try:
        with open(HISTORY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)
    except Exception:
        logger.exception("Failed to write history CSV")


def _read_history(player_id: str | None = None, limit: int = 20) -> list[dict]:
    """Read recent sessions from history CSV, optionally filtered by player_id."""
    if not os.path.exists(HISTORY_CSV):
        return []
    rows = []
    try:
        with open(HISTORY_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if player_id is None or row.get("player_id") == player_id:
                    rows.append(row)
    except Exception:
        logger.exception("Failed to read history CSV")
    return rows[-limit:]  # most recent N


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPER
# ─────────────────────────────────────────────────────────────────────────────

async def _upload_to_snapshot(file: UploadFile) -> dict:
    ctype = (file.content_type or "").lower()
    if not ctype.startswith(ALLOWED_MIME_PREFIX):
        raise HTTPException(status_code=415, detail="Unsupported file type; expected video/*")

    suffix = os.path.splitext(file.filename or "upload")[1] or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload too large")

        tmp.write(content)
        tmp.flush()
        tmp.close()

        try:
            snapshot = analyze_video_file(tmp.name)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        if snapshot is None:
            raise HTTPException(
                status_code=400,
                detail="No kick detected in the uploaded video. "
                       "Ensure the video shows a clear kicking motion.",
            )
        return snapshot

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ready",
        "endpoints": {
            "POST /analyze":    "Raw biomechanical snapshot.",
            "POST /feedback":   "Full AI coaching feedback (pass player context fields).",
            "GET  /profiles":   "Available pro reference profiles.",
            "GET  /history":    "Past session records.",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready", "model_loaded": True}


@app.get("/profiles")
def list_profiles():
    """Return available pro reference profiles for the frontend dropdown."""
    return JSONResponse({
        "profiles": [
            {
                "key":         k,
                "label":       v["label"],
                "description": v["description"],
                "level":       v["level"],
                "age_range":   list(v["age_range"]),
            }
            for k, v in PRO_PROFILES.items()
        ]
    })


@app.get("/history")
def get_history(player_id: Optional[str] = None, limit: int = 20):
    """Return the last *limit* session records, optionally filtered by player_id."""
    rows = _read_history(player_id=player_id, limit=limit)
    return JSONResponse({"sessions": rows, "count": len(rows)})


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Raw biomechanical snapshot — no LLM call."""
    snapshot = await _upload_to_snapshot(file)
    logger.info("/analyze – kick detected at frame %s", snapshot.get("frame_number"))
    return JSONResponse({"status": "ok", "final_kick": snapshot}, status_code=200)


@app.post("/feedback")
async def get_feedback(
    file: UploadFile = File(...),
    # ── Player context fields (all optional) ──────────────────────────────────
    age:       Optional[int] = Form(None),
    level:     Optional[str] = Form(None),   # recreational | youth | advanced | elite
    position:  Optional[str] = Form(None),   # striker | winger | midfielder | defender
    kick_type: Optional[str] = Form(None),   # power | finesse | curl | pass | shot
    goals:     Optional[str] = Form(None),   # free text
    player_id: Optional[str] = Form(None),   # for history tracking
):
    """
    Full pipeline:
      1. Pose analysis → biomechanical snapshot.
      2. Select pro reference profile for this player.
      3. Compare snapshot → compute deviations.
      4. Gemini generates context-aware feedback + drills.
      5. Persist session to history.
      6. Return everything to the frontend.
    """
    # ── Step 1: pose analysis ─────────────────────────────────────────────────
    snapshot = await _upload_to_snapshot(file)
    logger.info("/feedback – kick at frame %s | player: %s, level: %s, pos: %s",
                snapshot.get("frame_number"), player_id, level, position)

    # ── Build context dict ────────────────────────────────────────────────────
    player_context = {
        k: v for k, v in {
            "age":       age,
            "level":     level,
            "position":  position,
            "kick_type": kick_type,
            "goals":     goals,
        }.items() if v is not None
    }

    # ── Steps 2-4: profile selection + deviations + LLM ──────────────────────
    try:
        result = await generate_feedback(snapshot, player_context)
    except RuntimeError as e:
        logger.error("LLM config error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        logger.error("LLM response parse error: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM response error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in feedback pipeline: %s", e)
        raise HTTPException(status_code=500, detail="Internal feedback error.")

    # ── Step 5: persist history ───────────────────────────────────────────────
    history_record = {
        "player_id":     player_id or "anonymous",
        "frame_number":  result.get("frame_number"),
        "profile":       result.get("profile_used", {}).get("key", "unknown"),
        "overall_rating": result.get("feedback", {}).get("overall_rating"),
        "age":           age,
        "level":         level,
        "position":      position,
        "kick_type":     kick_type,
        # Store per-feature values for trend tracking
        **{
            d["feature"]: d.get("user_value")
            for d in result.get("deviations", [])
        },
    }
    _append_history(history_record)

    return JSONResponse({"status": "ok", **result}, status_code=200)


# ─────────────────────────────────────────────────────────────────────────────
# ADD NEW ROUTES BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────