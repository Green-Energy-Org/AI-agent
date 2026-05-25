"""
langfuse_scorer.py

Two modes:
  A) inline_score(trace_id, question, answer) — called from main.py after each run
  B) score_recent_traces(hours=1)             — called from cron/CI

Groq is used as the judge via its OpenAI-compatible CHAT endpoint:
  POST https://api.groq.com/openai/v1/chat/completions
The GET /openai/v1 "unknown URL" error is normal — that root path doesn't exist,
only the full /chat/completions path does. The openai SDK handles this correctly.
"""

import os, re, json
from datetime import datetime, timedelta
from openai import OpenAI
from langfuse import Langfuse

# ── Clients ─────────────────────────────────────────────────────────────────
langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

# Groq via OpenAI-compatible CHAT endpoint (POST only — GET root returns 404, that's expected)
groq_judge = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",  # SDK appends /chat/completions automatically
)
JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_PROMPT = """\
You are an expert evaluator for a DevOps AI assistant.
Score this answer (each 0.0-1.0). Reply ONLY with JSON, no markdown:
{{"relevance":0.0,"faithfulness":0.0,"completeness":0.0,"hallucination_risk":0.0,"reasoning":"one line"}}

Where hallucination_risk: 1.0=clearly grounded, 0.0=suspicious/fabricated

Question: {question}
Answer: {answer}"""


def _llm_judge(question: str, answer: str) -> dict:
    """Call Groq judge. Returns score dict or {} on failure."""
    try:
        resp = groq_judge.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question[:500], answer=answer[:1500])}],
            max_tokens=256, temperature=0.0,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [scorer] LLM judge failed: {e}")
        return {}


def _heuristic(answer: str) -> float:
    if not answer: return 0.0
    s = 0.4 if len(answer) > 500 else (0.3 if len(answer) > 200 else 0.1)
    if re.search(r"i apologize|error generating", answer, re.I): s -= 0.3
    if re.search(r"\d+\.", answer): s += 0.2
    if re.search(r"^[-*]", answer, re.M): s += 0.2
    if re.search(r"```", answer): s += 0.2
    return max(0.0, min(1.0, s))


def inline_score(trace_id: str, question: str, answer: str):
    """
    Called immediately after agent run with the known trace_id.
    This is the RELIABLE path — no polling needed.
    """
    if not trace_id or not answer:
        return

    judge = _llm_judge(question, answer)

    # Push LLM judge scores
    for metric in ("relevance", "faithfulness", "completeness", "hallucination_risk"):
        value = judge.get(metric)
        if isinstance(value, (int, float)):
            langfuse.create_score(
                trace_id=trace_id,
                name=f"llm-judge/{metric}",
                value=float(value),
                data_type="NUMERIC",
                comment=judge.get("reasoning", ""),
            )

    # Always push heuristic score (works even if LLM judge fails)
    langfuse.create_score(
        trace_id=trace_id,
        name="heuristic/quality",
        value=_heuristic(answer),
        data_type="NUMERIC",
    )
    has_err = bool(re.search(r"i apologize|error generating", answer, re.I))
    langfuse.create_score(
        trace_id=trace_id,
        name="has-error",
        value=0 if has_err else 1,
        data_type="BOOLEAN",
    )

    langfuse.flush()
    print(f"  [scorer] scored trace={trace_id[:12]}… relevance={judge.get('relevance','n/a')}")


def score_recent_traces(hours: int = 1):
    """Batch mode: score unscored traces from the last N hours."""
    since = datetime.utcnow() - timedelta(hours=hours)
    traces = langfuse.fetch_traces(from_timestamp=since, limit=50,
                                    order_by="createdAt", order="desc").data
    print(f"[scorer] Found {len(traces)} traces since {since.isoformat()} UTC")
    scored = 0
    for trace in traces:
        output  = trace.output or {}
        input_  = trace.input  or {}
        answer   = output.get("answer", "") if isinstance(output, dict) else str(output)
        question = input_.get("query",  "") if isinstance(input_,  dict) else str(input_)
        if not answer:
            continue
        inline_score(trace.id, question, answer)
        scored += 1
    print(f"[scorer] Done. Scored {scored}/{len(traces)}.")


if __name__ == "__main__":
    score_recent_traces(hours=1)
