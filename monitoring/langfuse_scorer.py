"""
langfuse_scorer.py  –  Automated post-hoc scoring of Langfuse traces.

Run manually or from a cron/GitHub Actions job:
    python monitoring/langfuse_scorer.py

What it does:
1. Fetches recent unscored traces from Langfuse
2. Scores them on basic heuristics (answer length, error presence)
3. Pushes scores back to Langfuse so they're visible in dashboards
"""

import os
import re
from datetime import datetime, timedelta
from langfuse import Langfuse

langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ.get("LANGFUSE_BASE_URL"),
)


def score_answer_quality(answer: str) -> float:
    """
    Heuristic quality score 0-1 based on:
    - Answer length (short answers score lower)
    - Presence of error keywords (penalise errors)
    - Presence of useful structure (lists, steps)
    """
    if not answer:
        return 0.0

    score = 0.0

    # Length heuristic  (0-0.4)
    length = len(answer)
    if length > 500:
        score += 0.4
    elif length > 200:
        score += 0.3
    elif length > 50:
        score += 0.2
    else:
        score += 0.1

    # Error penalty  (-0.3)
    error_patterns = [r"i apologize", r"error generating", r"encountered an error"]
    for pattern in error_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            score -= 0.3
            break

    # Structure bonus  (+0.2 each, max 0.6)
    if re.search(r"\d+\.", answer):          # numbered list
        score += 0.2
    if re.search(r"^[-*]", answer, re.M):    # bullet list
        score += 0.2
    if re.search(r"```", answer):            # code block
        score += 0.2

    return max(0.0, min(1.0, score))


def score_recent_traces(hours: int = 1):
    """Fetch and score traces from the last N hours."""
    since = datetime.utcnow() - timedelta(hours=hours)

    print(f"[scorer] Fetching traces since {since.isoformat()} UTC …")

    traces = langfuse.fetch_traces(
        from_timestamp=since,
        limit=50,
        order_by="createdAt",
        order="desc",
    ).data

    print(f"[scorer] Found {len(traces)} traces.")

    scored = 0
    for trace in traces:
        output = trace.output or {}
        answer = output.get("answer", "") if isinstance(output, dict) else str(output)

        if not answer:
            continue

        quality = score_answer_quality(answer)

        langfuse.create_score(
            trace_id=trace.id,
            name="auto-quality-score",
            value=quality,
            data_type="NUMERIC",
            comment=f"Automated heuristic score (answer_length={len(answer)})",
        )

        # Flag errors explicitly
        has_error = re.search(r"i apologize|error generating", answer, re.IGNORECASE)
        langfuse.create_score(
            trace_id=trace.id,
            name="has-error",
            value=0 if has_error else 1,
            data_type="BOOLEAN",
        )

        scored += 1
        print(f"  [+] trace={trace.id[:12]}…  quality={quality:.2f}  error={'yes' if has_error else 'no'}")

    print(f"[scorer] Done. Scored {scored}/{len(traces)} traces.")
    langfuse.flush()


if __name__ == "__main__":
    score_recent_traces(hours=1)
