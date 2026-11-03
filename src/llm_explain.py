"""Plain-language explanation of a score.

The explanation is only a readable rewrite of the model's reason codes. The LLM never sees the
applicant's record and never makes the decision. If the LLM is switched off, fails, or writes
something that breaks the rules below, the deterministic template is used instead.

Optional extras (not needed for the core project):
    pip install -r requirements-llm.txt      # google-genai, langfuse
    set ENABLE_LLM_EXPLANATION=1  GEMINI_API_KEY=...   (and the LANGFUSE_* keys for tracing)
"""
import os
import re

# Langfuse tracing is optional. Works with both the v3 and the older v2 package layout.
try:                                                   # pragma: no cover
    from langfuse import observe
except Exception:                                      # pragma: no cover
    try:
        from langfuse.decorators import observe
    except Exception:
        def observe(*args, **kwargs):
            def wrap(fn):
                return fn
            return wrap(args[0]) if args and callable(args[0]) else wrap

DISCLAIMER = ("This explanation is generated text for readability. It does not change the score or the decision.")

FORBIDDEN = ["guarantee", "will be approved", "will be rejected", "race", "gender", "religion",
             "caste", "age ", "nationality", "marital"]


def template_explanation(score, decision, reasons):
    if not reasons:
        body = "No single factor stands out as lowering this score."
    else:
        labels = [r["label"].lower() for r in reasons]
        joined = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
        body = f"The main factors lowering this score were: {joined}."
    return f"Score {score:.0f}, decision: {decision}. {body} {DISCLAIMER}"


def build_prompt(score, decision, reasons):
    items = "\n".join(f"- {r['label']}" for r in reasons) or "- (none)"
    return (
        "Rewrite the following credit-score result in two short, neutral, plain-English sentences "
        "for a customer. Use ONLY the facts below. Do not add reasons, numbers, advice about "
        "approval, or anything about personal characteristics.\n"
        f"Score: {score:.0f}\nDecision: {decision}\nFactors that lowered the score:\n{items}\n"
    )


def validate(text, score, reasons):
    """Cheap safety checks on LLM output. Returns (ok, reason)."""
    low = text.lower()
    for word in FORBIDDEN:
        if word in low:
            return False, f"contains '{word.strip()}'"
    allowed = {str(int(round(score)))}
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        if number not in allowed:
            return False, f"mentions an unexpected number {number}"
    if reasons and not any(r["label"].lower().split()[0] in low for r in reasons):
        return False, "does not mention any of the factors"
    return True, "ok"


@observe(name="llm_explanation")
def _call_llm(prompt):                                  # pragma: no cover
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return client.models.generate_content(model=model, contents=prompt).text.strip()


def explain(score, decision, reasons):
    """Returns (text, source) where source is 'llm' or 'template'."""
    fallback = template_explanation(score, decision, reasons)
    if os.getenv("ENABLE_LLM_EXPLANATION", "0") != "1" or not os.getenv("GEMINI_API_KEY"):
        return fallback, "template"
    try:                                                # pragma: no cover
        text = _call_llm(build_prompt(score, decision, reasons))
        ok, why = validate(text, score, reasons)
        if ok:
            return f"{text} {DISCLAIMER}", "llm"
        return fallback, f"template (llm rejected: {why})"
    except Exception as exc:                            # pragma: no cover
        return fallback, f"template (llm error: {type(exc).__name__})"
