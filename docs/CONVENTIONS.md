# Versioning and the consumer loop

How the public surface is versioned, and how a downstream consumer drives changes here.

## Versioning

- Keep `pyproject.toml` `[project].version` and `rlm_harness/__init__.__version__` in
  sync. On a bump, fold the release's changes into `CHANGELOG.md`.
- **Post-1.0, the public surface follows SemVer, and that RETIRES the lockstep hard rename.**
  Before 1.0.0 this kit renamed a public name with no alias and updated its consumers in the same
  breath (`make_middleware_lm` → `intercept_sub_lm` is the recorded case, and the CHANGELOG entry
  says so in as many words). That is over. The frozen surface is `__init__.__all__` + the
  `rlm-harness/trace/v1` wire format + `RLMTask`'s declaration fields, all pinned by
  `tests/test_contract.py`. **Adding** a public name is a MINOR bump. **Renaming or removing** one
  means: ship the new name, keep the old as an alias that emits a `DeprecationWarning`, note it in
  the CHANGELOG, and do not delete the alias before the next MAJOR. A `_`-prefixed name or module
  internal (`_retry`, `trace._active`) is outside the promise and may still move freely. The trace
  format keeps its own version and its own additive-only rule within v1; a break there is a `v2`
  migration, not a SemVer major on its own. If you find yourself wanting a hard rename because a
  consumer is the only caller, that is exactly the situation the rule exists for. There are nine
  of them now, and you cannot see all of their working trees.

## Consumer-driven hardening

- This kit is driven by a real downstream consumer (a task that builds on the
  scaffold, pinning the kit as a git dep: overlaid editable for local co-dev). That dogfooding is the design loop: when the consumer
  forces a workaround, log the **reusable** gap and fix it in the kit. Do not special-case
  the consumer. Generic mechanics get promoted here via the base/wrap split (a generic base +
  syntactic guard + factory in the kit; the provider + tracing in the consumer); consumer-specific
  values (model names, schemas, paths) stay in the consumer, not here.
