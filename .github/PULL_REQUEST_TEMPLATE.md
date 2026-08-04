## What & why

<!-- What does this change, and why? Reference any issue it closes, e.g. "Closes #12". -->

## Checklist

- [ ] `uv run --group dev python -m pytest` is green
- [ ] `uvx ruff check .` is clean
- [ ] New behavior has a test
- [ ] No new top-level `dspy` import in a dspy-free module (`config` / `_retry` / `sandbox` / `tools` / `trace` / `skills` / `replay` / `dataset`)
- [ ] The `rlm-harness/trace/v1` trace is unchanged, or the change is additive (a new optional field) — `tests/test_contract.py` still green
- [ ] The public surface stays vendor-neutral (no specific downstream project names or values)
