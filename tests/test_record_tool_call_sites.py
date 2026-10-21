"""Two structural rules about recording, from the two ways a trace has gone silent.

**A site that records must declare its outcome.** `trace.payload_cause` reads a MISSING `ok` as
`CAUSE_INVALID`, so a writer that omits it does not produce an unknown: it produces a confident
wrong answer, and `metrics.compute_run_facts` counts it in `tool_declines`.

**A shipped tool's silence must carry a reason.** `make_schema_validator` recorded nothing, and a
consumer reading its own traces could not tell "the model never validated" from "the validator is
not instrumented": theirs emitted ZERO events across 42 turns of a run that called it. Recording it
turned out to be the wrong fix and was reverted inside 1.24.0 (it composes, so its own verdict is
not the one the model was given), which is the point of the rule rather than a counter-example to
it: the answer is an argued exemption, and the defect was that nothing made anyone argue.

Both were live in 1.22.0, and the reason the suite missed each one is worth keeping, because it is
not carelessness in either case. The first: `ok=` was only ever written next to a REFUSAL BRANCH,
and the two files that omitted it are exactly the two with no `ok=False` site anywhere -- one tool
cannot fail, one returns its miss as a value from the same call site, one recorded only its success.
No sibling to copy. The second: the suite's only package-wide factory table, in
`test_tool_durations.py`, asks whether a tool TIMES ITSELF, and classified the validator as exempt
for a reason about I/O that says nothing about whether an event exists at all. That table even
records "deliberately record nothing" for two base factories, so silence was a legitimate answer to
a question nobody had asked.

No table for the first rule, unlike `_OUTBOUND`/`_NOT_OUTBOUND` there: the walk finds every call
site, so a new one is covered when it is written rather than when someone remembers to list it.
Needs no dspy, because the `tools` package must not either.
"""

import ast
import pathlib

# Exactly the keys `payload_cause` READS, in its own precedence: `circuit_broken` (nothing ran),
# the endpoint string under either spelling (nothing was validated), `ok`. A `cause=` keyword is
# deliberately NOT here: the reader does not look at it, so a site declaring only that one would
# still produce a `CAUSE_UNKNOWN`, and a gate that disagreed with the reader would pass it.
_DECLARES = {"ok", "circuit_broken", "endpoint_error", "error"}

_PKG = pathlib.Path(__file__).resolve().parent.parent / "rlm_harness"


def _sites():
    """Every `record_tool_call(...)` call in the package, as (path, lineno, declares_outcome)."""
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "record_tool_call":
                continue
            named = {kw.arg for kw in node.keywords if kw.arg is not None}
            splatted = any(kw.arg is None for kw in node.keywords)
            yield path, node.lineno, bool(named & _DECLARES) or splatted


def test_every_record_tool_call_declares_its_outcome():
    """A site that names none of `_DECLARES` is recorded as a decline whatever it did."""
    sites = list(_sites())
    silent = [
        f"{path.relative_to(_PKG.parent)}:{lineno}"
        for path, lineno, declares in sites
        if not declares
    ]
    assert not silent, (
        "record_tool_call sites that declare no outcome, so `payload_cause` reads them as "
        f"CAUSE_INVALID: {silent}"
    )


def test_the_walk_actually_finds_the_sites():
    """Guards the instrument: an `_sites()` that found nothing would pass the rule above."""
    assert len(list(_sites())) > 40


# --- rule 2: a shipped tool factory records, or says why it does not -------------------------
#
# The whole public surface that RETURNS something a model can call. Checked structurally rather
# than by building each one, because a fixture per factory is what kept the question unasked: the
# gate has to be cheap enough that a new factory cannot ship past it. It therefore catches "records
# NOTHING", which is the observed failure, and not "records on only some paths", which rule 1 above
# and `test_tool_durations.py` cover between them.
_EXEMPT_FROM_RECORDING = {
    # Side-effect-free BASES, by design: the consumer's own wrapper owns the recording, which is
    # also why `ModelToolResult.cause` exists as a structured handoff instead of an event.
    "make_model_tool": "base factory; the consumer's wrapper records",
    "make_harness_tool": "base factory; the consumer's wrapper records",
    # Takes an already-PARSED object, so a model cannot call it: an event here would describe the
    # consumer's composition step rather than a turn.
    "make_json_schema_validator": "not model-callable; takes a parsed object",
    # Model-callable, unlike the three above, and still exempt: it COMPOSES. The common shape runs
    # this check first and then applies checks the schema cannot express, so an `ok=True` from here
    # can sit on a turn where the model was told "Validation failed". The verdict belongs to the
    # outermost callable, which is the only one that knows it.
    "make_schema_validator": "composes; the outermost callable owns the verdict",
}


def _tool_factories():
    """Every public name that returns a model-facing callable."""
    import rlm_harness
    import rlm_harness.tools as tools_pkg

    names = {n for n in tools_pkg.__all__ if n.startswith("make_")}
    # The three outside the `tools` package, which is where half the silence was.
    for extra in ("load_skills_as_tools", "model_as_tool", "mcp_tools"):
        assert extra in rlm_harness.__all__, f"{extra} left the public surface; update this list"
        names.add(extra)
    return names


def _called_names(node):
    """Every name called anywhere under `node`, attribute calls included."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            func = inner.func
            yield func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _defs_that_record(name):
    """True when `def <name>`, or a helper it calls in the same module, records a `tool_call`.

    Following same-module calls is not thoroughness, it is a correction: without it the gate
    reported `mcp_tools` as recording nothing, which is false. It delegates to `_make_tool`, whose
    nested `call` carries both sites. A check that cannot see one level of delegation produces a
    finding about its own reach.
    """
    found = False
    for path in _PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(node.name, []).append(node)
        if name not in defs:
            continue
        found = True
        pending, seen = [name], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            for node in defs.get(current, ()):
                for called in _called_names(node):
                    if called == "record_tool_call":
                        return True
                    if called in defs:
                        pending.append(called)
    assert found, f"no `def {name}` found in the package; this instrument cannot answer"
    return False


def test_every_tool_factory_records_or_is_exempt():
    """A factory whose tool emits no event makes "the model never called it" unreadable."""
    silent = sorted(
        name
        for name in _tool_factories()
        if name not in _EXEMPT_FROM_RECORDING and not _defs_that_record(name)
    )
    assert not silent, (
        f"tool factories that record no tool_call at all: {silent}. Record one, or add it to "
        "_EXEMPT_FROM_RECORDING with the reason a model cannot call it."
    )


def test_the_exemptions_are_still_exempt():
    """An exemption that starts recording is a stale reason, which is how the duration table's
    "no I/O" entry outlived its own meaning."""
    recording = sorted(n for n in _EXEMPT_FROM_RECORDING if _defs_that_record(n))
    assert not recording, f"exempt but now recording, so the reason is stale: {recording}"
