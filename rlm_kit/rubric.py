"""Reward-free rubric primitives — the shared substrate for decomposing "did this run succeed?" into
observable CRITERIA carried as LABELS.

``category`` is an OPAQUE, caller-defined label: rlm-kit never interprets it, hardcodes no taxonomy, and
carries no domain vocabulary. A consumer defines its own category set, criterion descriptions, and the
``lens`` mapping each category to the trace facts it surfaces; this module owns only the generic types, the
run_start-meta (de)serialization, a structural lint, and the pure per-criterion fact-assembly loop.

Scoring — turning facts into a per-criterion or aggregate value — is the downstream TRAINER's job, never
here: this keeps the rubric inside rlm-kit's "trajectories, never reward" invariant (emit the criteria +
per-criterion FACTS as data; scoring stays downstream). Pure pydantic + stdlib; dspy-free, so
``import rlm_kit`` stays cheap and this module is testable in isolation.

A consumer wraps these — supplying its own category set + criterion skeleton, its own ``trace -> facts``
function, and its own ``category -> keys`` lens — and re-exports the types so its own call sites are
unchanged. rlm-kit stays taxonomy-agnostic; the meaning of a category lives entirely in the consumer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .trace import EVENT_RUN_START


class Criterion(BaseModel):
    """One rubric criterion — the STRUCTURE only. Scoring is the trainer's job, never here."""

    name: str = Field(..., description="short unique criterion id")
    description: str = Field(..., description="what the trajectory must satisfy, observable from the trace")
    weight: float = Field(1.0, description="relative weight WITHIN its category (the trainer aggregates)")
    category: str = Field(..., description="caller-defined category label — OPAQUE to rlm-kit")


class RubricCriteria(BaseModel):
    """A rubric — a set of criteria, carried in a run's run_start meta as LABELS (never a reward)."""

    criteria: list[Criterion] = Field(default_factory=list)


class CriterionFact(BaseModel):
    """A DETERMINISTIC observation about one criterion, re-sourced from the trace (a FACT, never a score)."""

    criterion: str
    category: str
    weight: float
    observed: dict = Field(
        default_factory=dict,
        description="deterministic facts (counts, ids); never a score or met/unmet decision",
    )


def rubric_to_meta(rubric: RubricCriteria) -> list[dict]:
    """Serialize a rubric for run_start meta (LABELS carried alongside the run — never a reward)."""
    return [c.model_dump() for c in rubric.criteria]


def rubric_from_meta(events: list[dict], *, categories: tuple[str, ...] | None = None) -> RubricCriteria:
    """Recover the rubric stored under a run's ``run_start`` meta ``['rubric']`` (empty if none recorded).

    Tolerant: a non-dict or malformed entry is skipped rather than crashing the read path (legacy traces
    that stored extra keys still coerce — pydantic ignores unknown keys). If ``categories`` is given (the
    caller's allowed label set), entries whose ``category`` is not in it are dropped; if None, any truthy
    category is accepted — rlm-kit imposes no taxonomy."""
    for e in events:
        if e.get("type") == EVENT_RUN_START:
            raw = ((e.get("payload") or {}).get("meta") or {}).get("rubric")
            if isinstance(raw, list):
                crits: list[Criterion] = []
                for c in raw:
                    if not isinstance(c, dict):
                        continue
                    cat = c.get("category")
                    if categories is not None:
                        if cat not in categories:
                            continue
                    elif not cat:
                        continue
                    try:  # skip a malformed entry (missing name/description) — never crash the read path
                        crits.append(Criterion(**c))
                    except (TypeError, ValueError):
                        continue
                return RubricCriteria(criteria=crits)
    return RubricCriteria(criteria=[])


def validate_rubric(
    rubric: RubricCriteria,
    *,
    categories: tuple[str, ...] | None = None,
    observable_vocab: tuple[str, ...] | None = None,
) -> list[str]:
    """A DETERMINISTIC structural lint of a rubric (NOT a semantic-quality judge). Returns human-readable
    issues (empty list = clean).

    Always checks: non-empty rubric, unique names, non-empty descriptions. When ``categories`` is given,
    also checks every category is represented. When ``observable_vocab`` is given, also runs a weak
    trace-observability heuristic — each description should mention at least one vocab term (deeper
    "is this rubric GOOD" validation needs a real training signal and is out of scope)."""
    criteria = rubric.criteria
    if not criteria:
        return ["rubric has no criteria"]
    issues: list[str] = []
    if categories is not None:
        present = {c.category for c in criteria}
        missing = [cat for cat in categories if cat not in present]
        if missing:
            issues.append(f"categories not represented: {missing}")
    names = [c.name for c in criteria]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        issues.append(f"duplicate criterion names: {dupes}")
    empty = [c.name for c in criteria if not (c.description or "").strip()]
    if empty:
        issues.append(f"criteria with empty descriptions: {empty}")
    if observable_vocab:
        vague = [c.name for c in criteria
                 if not any(w in (c.description or "").lower() for w in observable_vocab)]
        if vague:
            issues.append("criteria whose description may not be trace-observable "
                          f"(no observable_vocab term): {vague}")
    return issues


def criteria_facts(
    criteria: list[Criterion], facts: dict, lens: dict[str, tuple[str, ...]]
) -> list[CriterionFact]:
    """Assemble per-criterion facts by slicing a resolved ``facts`` dict through a ``category -> keys`` lens.

    PURE: takes the already-resolved ``criteria``, a ``facts`` dict (whatever the consumer's trace yields),
    and a ``lens`` mapping each category to the fact keys it surfaces. A category absent from ``lens``
    yields empty facts (``.get`` — never ``KeyError``, which matters for an opaque category the lens does
    not cover). NO trace/event/domain knowledge here — the consumer supplies criteria, facts, and lens.
    This NEVER decides met/unmet or a score."""
    out: list[CriterionFact] = []
    for c in criteria:
        keys = lens.get(c.category, ())
        observed = {k: facts[k] for k in keys if k in facts}
        out.append(CriterionFact(criterion=c.name, category=c.category, weight=c.weight, observed=observed))
    return out
