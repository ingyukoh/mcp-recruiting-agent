"""Deterministic guardrails applied at three boundaries of the agent graph.

1. User input   -> direct prompt injection, discriminatory screening requests, PII redaction.
2. Tool output  -> indirect prompt injection hidden in retrieved data.
3. Final answer -> PII redaction and a grounding check on every cited record id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions"
    r"|disregard (the |your )?(system|previous) (prompt|instructions)"
    r"|reveal (your |the )?(system prompt|hidden instructions)"
    r"|you are now (in )?(developer|dan|jailbreak) mode"
    r"|rank (this|me) (candidate )?first)",
    re.IGNORECASE,
)

PROTECTED = re.compile(
    r"\b(age|aged|young(er)?|old(er)?|(under|over) \d{2}(?!\s*\+?\s*(years?|yrs))"
    r"|gender|male|female|men|women|race|ethnic\w*|religio\w*|christian|muslim|jewish|pregnan\w*|disabilit\w*|disabled"
    r"|married|single mothers?|nationality|national origin|native[- ]born|sexual orientation)\b",
    re.IGNORECASE,
)
SELECTION = re.compile(
    r"\b(only|exclude|excluding|filter|prefer|preferred|no|without|rank|screen out|avoid|under"
    r"|over|younger|older|find|show|list|search|hire|want|looking for|shortlist)\b",
    re.IGNORECASE,
)
POLICY_QUESTION = re.compile(
    r"\b(policy|allowed|legal|lawful|can we|can i|may we|is it ok|compliant|eeo)\b", re.IGNORECASE
)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RECORD_ID = re.compile(r"\b[CJ]-\d{3}\b")

INJECTION_REFUSAL = (
    "I can't follow instructions that try to override my operating rules. "
    "I can help search jobs, review candidate skills, score matches, or explain hiring policy."
)
DISCRIMINATION_REFUSAL = (
    "I can't filter or rank candidates by protected characteristics such as age, gender, race, "
    "religion, disability, pregnancy, marital status, or national origin (policy: eeo). "
    "I can shortlist on job-related skills and experience instead."
)


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    refusal: str = ""
    events: list[str] = field(default_factory=list)


def redact_pii(text: str) -> tuple[str, list[str]]:
    events = []
    for label, pattern in (("email", EMAIL), ("ssn", SSN), ("phone", PHONE)):
        if pattern.search(text):
            text = pattern.sub(f"[redacted {label}]", text)
            events.append(f"pii_redacted:{label}")
    return text, events


def check_input(text: str) -> GuardResult:
    if INJECTION.search(text):
        return GuardResult(text, True, INJECTION_REFUSAL, ["blocked:prompt_injection"])
    if PROTECTED.search(text) and SELECTION.search(text) and not POLICY_QUESTION.search(text):
        return GuardResult(text, True, DISCRIMINATION_REFUSAL, ["blocked:protected_attribute"])
    redacted, events = redact_pii(text)
    return GuardResult(redacted, events=events)


def sanitize_tool_output(text: str) -> tuple[str, list[str]]:
    """Neutralize instruction-like text inside retrieved data, keeping the payload parseable."""
    if not INJECTION.search(text):
        return text, []
    span = re.compile(INJECTION.pattern + r'[^.!?"\n]*[.!?]?', re.IGNORECASE)
    return span.sub("[removed instruction-like text]", text), [
        "sanitized:indirect_prompt_injection"
    ]


def check_output(answer: str, grounded_ids: set[str]) -> tuple[str, list[str]]:
    answer, events = redact_pii(answer)
    unknown = sorted(set(RECORD_ID.findall(answer)) - grounded_ids)
    if unknown:
        for rid in unknown:
            answer = answer.replace(rid, "[unverified id]")
        events.append("ungrounded_ids_removed:" + ",".join(unknown))
    return answer, events
