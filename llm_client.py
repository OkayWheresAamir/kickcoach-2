# llm_client.py
# ─────────────────────────────────────────────────────────────────────────────
# Thin async wrapper around the Google Gemini API.
# All LLM config lives here — swap model or provider by editing this one file.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import logging
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL     =  "gemini-2.5-flash"
GEMINI_ENDPOINT  = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT    = 90          # seconds
GEMINI_MAX_TOKENS = 8192        # enough for full JSON response

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences the model may add despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$",       "", text)
    return text.strip()


def _is_truncated(text: str) -> bool:
    """
    Heuristic: if the JSON object never closes its outermost brace
    the response was cut off mid-stream.
    """
    opens  = text.count("{")
    closes = text.count("}")
    return opens > closes or (opens > 0 and closes == 0)


def _extract_json(text: str) -> dict:
    """
    Parse the JSON object from the model's raw text response.
    Handles markdown fences and attempts a best-effort repair for
    mildly truncated responses (e.g. last field cut short).
    """
    text = _strip_fences(text)

    # Fast path — well-formed response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ── Repair: truncated response ────────────────────────────────────────
    if _is_truncated(text):
        logger.warning("Gemini response appears truncated (%d chars); attempting repair.", len(text))

        # Drop the last incomplete line (likely a half-written string)
        lines = text.rstrip().splitlines()
        while lines:
            last = lines[-1].rstrip().rstrip(",")
            # If the last line doesn't look like a complete JSON value, drop it
            if last.endswith(('"', '}', ']', 'true', 'false')) or last[-1:].isdigit():
                break
            lines.pop()

        repaired = "\n".join(lines)

        # Close any open arrays/objects
        open_arrays  = repaired.count("[") - repaired.count("]")
        open_objects = repaired.count("{") - repaired.count("}")
        repaired += "]" * max(0, open_arrays)
        repaired += "}" * max(0, open_objects)

        try:
            result = json.loads(repaired)
            logger.info("Truncation repair succeeded.")
            return result
        except json.JSONDecodeError:
            pass

    # ── Last resort: extract first {...} blob via regex ───────────────────
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse Gemini response as JSON.\nRaw text: %s", text[:600])
    raise ValueError(
        "LLM returned a response that could not be parsed as JSON. "
        "This usually means the model output was cut off. "
        f"Raw excerpt: {text[:200]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL REQUEST
# ─────────────────────────────────────────────────────────────────────────────

async def _post_to_gemini(client: httpx.AsyncClient, prompt: str) -> str:
    """Send one request to Gemini and return the raw text content."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_MAX_TOKENS,
            "temperature":     0.4,
            "topP":            0.9,
        },
    }

    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    response = await client.post(url, json=payload)
    response.raise_for_status()

    data = response.json()

    # Log finish reason so truncation is obvious in server logs
    try:
        finish_reason = data["candidates"][0].get("finishReason", "UNKNOWN")
        logger.info("Gemini finishReason: %s", finish_reason)
        if finish_reason == "MAX_TOKENS":
            logger.warning(
                "Gemini hit MAX_TOKENS — response may be truncated. "
                "Consider raising GEMINI_MAX_TOKENS further."
            )
    except (KeyError, IndexError):
        pass

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error("Unexpected Gemini response structure: %s", data)
        raise ValueError(f"Unexpected Gemini response structure: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def call_gemini(prompt: str) -> dict:
    """
    Send *prompt* to Gemini and return the parsed JSON feedback dict.

    Strategy:
      1. First attempt with the full prompt.
      2. If the response is truncated and repair fails, retry once with a
         tighter prompt that asks for shorter drills/explanations.

    Raises
    ------
    RuntimeError        – GEMINI_API_KEY not set.
    httpx.HTTPStatusError – non-2xx from the API.
    ValueError          – response cannot be parsed after retry.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before starting the server."
        )

    logger.info("Calling Gemini model: %s", GEMINI_MODEL)

    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:

        # ── Attempt 1 ─────────────────────────────────────────────────────
        raw_text = await _post_to_gemini(client, prompt)
        logger.info("Gemini response received (%d chars)", len(raw_text))

        try:
            return _extract_json(raw_text)
        except ValueError as first_err:
            logger.warning("First parse attempt failed: %s — retrying with compact prompt.", first_err)

        # ── Attempt 2: compact prompt ──────────────────────────────────────
        compact_suffix = (
            "\n\nIMPORTANT: Your previous response was truncated. "
            "Respond with ONLY valid JSON. "
            "Keep ALL string values under 120 characters. "
            "Limit to 2 drills max. No markdown, no code fences."
        )
        raw_text2 = await _post_to_gemini(client, prompt + compact_suffix)
        logger.info("Retry Gemini response (%d chars)", len(raw_text2))
        return _extract_json(raw_text2)  # raises ValueError if still broken