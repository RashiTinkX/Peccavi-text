"""
eval/quality.py
Text quality scoring: GPT-4o-mini judge (primary) with Flesch fallback.
GPT-4 judge is significantly stronger than Flesch for EMNLP evaluation.
"""

from __future__ import annotations
import os
import logging
import textstat

logger = logging.getLogger(__name__)

_gpt4_client = None


def _get_client():
    global _gpt4_client
    if _gpt4_client is not None:
        return _gpt4_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        _gpt4_client = OpenAI(api_key=api_key)
        return _gpt4_client
    except ImportError:
        logger.warning("openai package not installed — falling back to Flesch score")
        return None


def flesch_quality_score(text: str) -> float:
    """Maps Flesch Reading Ease (0–100) to a 1–5 scale."""
    fre = textstat.flesch_reading_ease(text)
    return round(1 + (fre / 100) * 4, 2)


def gpt4_quality_score(text: str, prompt: str = None) -> float:
    """
    Rate text quality 1–5 using GPT-4o-mini as judge.
    Criteria: coherence, fluency, relevance to prompt, informativeness.
    Falls back to Flesch if OpenAI key unavailable.
    """
    client = _get_client()
    if client is None:
        return flesch_quality_score(text)

    system = (
        "You are a strict text quality evaluator for NLP research. "
        "Rate the text on a scale of 1 to 5 based on coherence, fluency, "
        "and informativeness. Respond with only a single integer (1, 2, 3, 4, or 5)."
    )
    user_parts = []
    if prompt:
        user_parts.append(f"Prompt: {prompt[:200]}")
    user_parts.append(f"Text: {text[:400]}")
    user_msg = "\n".join(user_parts)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=3,
            temperature=0,
        )
        score = float(response.choices[0].message.content.strip())
        return round(min(max(score, 1.0), 5.0), 2)
    except Exception as e:
        logger.warning(f"GPT-4 quality judge failed: {e} — falling back to Flesch")
        return flesch_quality_score(text)


def quality_score(text: str, prompt: str = None) -> float:
    """Primary quality function: GPT-4 if available, else Flesch."""
    return gpt4_quality_score(text, prompt=prompt)
