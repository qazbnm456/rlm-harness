"""Contract tests for the reward-free rubric primitives.

Deliberately uses OPAQUE, methodology-agnostic categories (``"X"`` / ``"Y"``) — rlm-kit imposes no
taxonomy, so the contract must hold for any caller-defined labels. This doubles as a vendor-neutrality
guard: nothing here names a specific methodology or its categories.
"""

from rlm_kit.rubric import (
    Criterion,
    CriterionFact,
    RubricCriteria,
    criteria_facts,
    rubric_from_meta,
    rubric_to_meta,
    validate_rubric,
)


def _rubric():
    return RubricCriteria(criteria=[
        Criterion(name="a", description="does the thing", weight=1.0, category="X"),
        Criterion(name="b", description="does the other thing", weight=2.0, category="Y"),
    ])


def _run_start(rubric_meta):
    return [{"type": "run_start", "payload": {"meta": {"rubric": rubric_meta}}}]


def test_to_meta_shape_and_roundtrip():
    meta = rubric_to_meta(_rubric())
    assert meta == [
        {"name": "a", "description": "does the thing", "weight": 1.0, "category": "X"},
        {"name": "b", "description": "does the other thing", "weight": 2.0, "category": "Y"},
    ]
    back = rubric_from_meta(_run_start(meta))
    assert [(c.name, c.category, c.weight) for c in back.criteria] == [("a", "X", 1.0), ("b", "Y", 2.0)]


def test_from_meta_empty_when_absent_or_malformed():
    assert rubric_from_meta([]).criteria == []                       # no run_start
    assert rubric_from_meta(_run_start(None)).criteria == []          # rubric not a list
    # a non-dict entry and a dict missing required fields are both skipped, never crash
    ok = rubric_from_meta(_run_start(["nope", {"name": "a"}, {"name": "a", "description": "d", "weight": 1.0, "category": "X"}]))
    assert [c.name for c in ok.criteria] == ["a"]


def test_from_meta_categories_filter_is_opt_in():
    meta = rubric_to_meta(_rubric())                                 # categories X, Y
    # None (default) accepts any truthy category
    assert len(rubric_from_meta(_run_start(meta)).criteria) == 2
    # a filter set drops the ones outside it
    assert [c.category for c in rubric_from_meta(_run_start(meta), categories=("X",)).criteria] == ["X"]
    # an entry with a falsy category is dropped when categories is None
    assert rubric_from_meta(_run_start([{"name": "z", "description": "d", "weight": 1.0, "category": ""}])).criteria == []


def test_criteria_facts_slices_by_lens_and_never_scores():
    facts = {"calls": 3, "ok": True, "misses": 0, "unrelated": 99}
    lens = {"X": ("calls", "ok"), "Y": ("misses",)}
    out = criteria_facts(_rubric().criteria, facts, lens)
    assert all(isinstance(f, CriterionFact) for f in out)
    assert out[0].observed == {"calls": 3, "ok": True}               # X lens; drops "unrelated"
    assert out[1].observed == {"misses": 0}                          # Y lens
    # a category absent from the lens yields empty facts, NOT a KeyError
    lonely = criteria_facts([Criterion(name="c", description="d", weight=1.0, category="Z")], facts, lens)
    assert lonely[0].observed == {}
    # reward-free: no score/met/reward key ever appears in observed or the dump
    dumped = [f.model_dump() for f in out]
    assert all(set(d) == {"criterion", "category", "weight", "observed"} for d in dumped)
    assert all(not any(k in d["observed"] for k in ("score", "met", "unmet", "reward")) for d in dumped)


def test_validate_rubric_generic_checks_need_no_params():
    assert validate_rubric(RubricCriteria(criteria=[])) == ["rubric has no criteria"]
    dup = RubricCriteria(criteria=[
        Criterion(name="a", description="d", weight=1.0, category="X"),
        Criterion(name="a", description="", weight=1.0, category="Y"),
    ])
    issues = validate_rubric(dup)                                    # no categories/vocab params
    assert any("duplicate criterion names" in i for i in issues)
    assert any("empty descriptions" in i for i in issues)


def test_validate_rubric_optional_category_and_vocab_checks():
    r = _rubric()
    # categories coverage check only when the set is supplied
    assert any("categories not represented" in i for i in validate_rubric(r, categories=("X", "Y", "Z")))
    assert not any("categories not represented" in i for i in validate_rubric(r, categories=("X", "Y")))
    # observability heuristic only when a vocab is supplied; message keeps "trace-observable", no domain words
    vague = validate_rubric(r, observable_vocab=("frobnicate",))
    assert any("may not be trace-observable" in i for i in vague)
    assert not validate_rubric(r, observable_vocab=("thing",))       # "thing" is in both descriptions
