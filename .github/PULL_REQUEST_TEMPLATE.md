## What & why

<!-- What does this change, and why? Reference any issue it closes, e.g. "Closes #12". -->

## Checklist

- [ ] `uv run --group dev --extra mcp --extra grep --extra gitignore python -m pytest -q` is green (the extras keep their tests from silently skipping)
- [ ] `uvx ruff check .` is clean
- [ ] New behavior has a test
- [ ] No new top-level `dspy` import in a dspy-free module (`config` / `_retry` / `sandbox` / `tools` / `trace` / `skills` / `replay` / `dataset` / `serving` / `harness_serve` / `metrics` / `rubric` / `atomic` / `isolation` / `_dspy_compat` / `_toolname`)
- [ ] No dspy kwarg, attribute, or error class hardcoded at a call site: it belongs in `_dspy_compat.py` with a case in `tests/test_dspy_compat.py`
- [ ] Any tool added or changed still passes `rlm_harness.testing.assert_repl_safe` (sync, explicit params, valid name)
- [ ] The `rlm-harness/trace/v1` trace is unchanged, or the change is additive (a new optional field): `tests/test_contract.py` still green
- [ ] The public surface stays vendor-neutral (no specific downstream project names or values)
