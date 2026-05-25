"""
scoring_test.py - Run ONCE to verify scorer + Langfuse connection work.

Usage:
    python scoring_test.py

If a score appears in Langfuse → SDK connectivity is fine.
If scores tab stays empty → check your env vars / network.
"""
import os
from langfuse import Langfuse

lf = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

# 1. Create a test trace
trace = lf.trace(
    name="score-connectivity-test",
    input={"query": "test question"},
    output={"answer": "test answer - this is a connectivity check"},
)
print(f"[+] Trace created: {trace.id}")

# 2. Attach a score directly — this is what langfuse_scorer.py does
lf.create_score(
    trace_id=trace.id,
    name="test/manual-score",
    value=0.9,
    data_type="NUMERIC",
    comment="connectivity test",
)

lf.flush()
print("[+] Score pushed. Check Langfuse → Scores tab for 'test/manual-score'.")
print(f"[+] Direct trace URL: {os.environ.get('LANGFUSE_BASE_URL','https://cloud.langfuse.com')}/traces/{trace.id}")
