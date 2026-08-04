# Context preservation (read before auto-compacting)

`rlm-harness` already routes durable knowledge into its tracked docs — keep using them, and
when the conversation is about to compact, preserve only what they do NOT already hold:

- **Stable invariants** → the **Invariants** section of `CLAUDE.md`.
- **Resolved decisions / shipped changes** → `CHANGELOG.md` (under the current version).
- **Open / proposed work** → the issue tracker, or the CHANGELOG's unreleased section.

So a handoff summary should carry the *in-flight session state* those files miss. Prioritize,
in order:

1. **Decisions we agreed on this session** that are not yet in CHANGELOG/CLAUDE — design choices
   ("depth stays 1 — depth>1 recursion is out of scope", "skills are knowledge-only, no script
   exec", "dataset split by tool name, not `kind=='tool'`"), API-shape calls, and the *reason*.
   Promote durable ones into CLAUDE.md (invariant) or CHANGELOG.md (change) before they fade.
2. **Files / symbols changed**, as `path:symbol` one-liners on the *final* shape — e.g.
   `sub_lm.py:_InterceptedSubLM.__call__ — records the escalation input on the sub_call event`,
   `dataset.py:export_actions — per-action (planner/tool/sub) records with run reward`. Drop
   diffs and intermediate revisions.
3. **Current status.** What passes `uv run pytest` (and the count), what is broken, last command
   run + result. One paragraph.
4. **Open suggestions / TODOs** not yet tracked — mark each `proposed`,
   `accepted-not-done`, or `rejected`, then move the durable ones into the issue tracker.
5. **In-flight consumer signal.** What the downstream consumer surfaced in the current
   work and what it needs — the dogfooding driver.
6. **In-flight user intent + acceptance criteria** for this session. Without it a resumed
   session drifts.

**Do NOT preserve** (reconstructable / already durable):

- Anything already in `CLAUDE.md`, `CHANGELOG.md`, `README.md`, or `pyproject.toml`.
- Tool-call transcripts, `grep` output, file listings, full file contents readable from disk.
- Step-by-step exploration narration; speculative reasoning that led to no decision.

**Format for a handoff summary** (use when compaction is imminent or the user asks for a recap):

```
## Session state
- Goal: <one sentence>
- Consumer driving it: <the downstream consumer | none>
- Status: <what passes pytest, what doesn't, last command + result>

## Decisions
- <decision> — <why>   (→ promote to CLAUDE.md invariant / CHANGELOG.md)

## Changed
- <path:symbol> — <what & why>

## Open
- [proposed|accepted-not-done|rejected] <item>   (→ issue tracker if durable)
```

Keep it under ~40 lines. If something fits one of the three tracked docs, put it THERE instead
of in the summary.
