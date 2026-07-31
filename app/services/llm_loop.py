"""Verified LLM loop: generate → validate schema → run code verifier → retry.

The core of "loop engineering" for this app. Instead of trusting a single
LLM shot, every call goes through:

  1. Generate  — ask Gemini
  2. Validate  — Pydantic schema check (structure: fields, types)
  3. Verify    — pure-Python fact checks against real data (grounding)
  4. On failure, feed the exact errors back to the model and retry
  5. After max_attempts, return None so the caller can use its fallback

Judgment is encoded once in the verifier and applied on every run.
"""

import json

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_ATTEMPTS = 3


def _parse_json(text: str) -> dict | None:
    """Extract the first balanced JSON object from messy LLM output."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def call_with_verification(
    messages: list[dict],
    schema: type[BaseModel],
    verify=None,
    *,
    label: str = "llm",
    max_attempts: int = MAX_ATTEMPTS,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 8000,
    reasoning_effort: str = "none",
) -> BaseModel | None:
    """Run the generate→validate→verify loop. Returns a validated schema
    instance, or None if all attempts fail (caller falls back).

    verify: optional callable taking the validated instance and returning a
    list of error strings (empty list = passed). Must be pure code, no LLM.
    """
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.effective_gemini_key,
        base_url=GEMINI_BASE_URL,
    )
    messages = list(messages)  # don't mutate the caller's list

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            print(f"[loop:{label}] attempt {attempt}: API call failed: {e}")
            continue

        text = response.choices[0].message.content or ""
        raw = _parse_json(text)
        if raw is None:
            print(f"[loop:{label}] attempt {attempt}: unparseable JSON")
            messages += [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": "Your answer was not valid JSON. Respond again with ONLY raw JSON in the exact format requested."},
            ]
            continue

        # Structural check — deterministic
        try:
            result = schema.model_validate(raw)
        except ValidationError as e:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()[:8]
            )
            print(f"[loop:{label}] attempt {attempt}: schema failed: {errors}")
            messages += [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": f"Your JSON had invalid structure: {errors}. Fix these fields and respond again with ONLY the corrected raw JSON."},
            ]
            continue

        # Fact check — deterministic, encoded judgment
        problems = verify(result) if verify else []
        if problems:
            print(f"[loop:{label}] attempt {attempt}: verifier failed: {problems}")
            messages += [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": "Your answer failed these checks:\n- " + "\n- ".join(problems) + "\nFix these issues and respond again with ONLY the corrected raw JSON."},
            ]
            continue

        if attempt > 1:
            print(f"[loop:{label}] succeeded on attempt {attempt}")
        return result

    print(f"[loop:{label}] all {max_attempts} attempts failed — falling back")
    return None
