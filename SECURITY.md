# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for a security vulnerability. Instead, DM **Boik Su ([@boik_su](https://x.com/boik_su))** on X with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept), and
- the affected version / commit.

You will receive an acknowledgement, and a fix or mitigation will be coordinated
before any public disclosure. Thank you for reporting responsibly.

## Supported versions

Only the latest released version receives security fixes; a fix ships as a patch release on top of it
rather than being backported to an earlier line.

| Version | Supported |
| ------- | --------- |
| latest  | ✅        |
| older   | ❌        |

## Scope notes

`rlm-harness` executes model-written code, so the interpreter choice is the security
boundary:

- The **default** interpreter is sandboxed (`pyodide`/`deno`). Sandbox-escape or
  isolation issues on the default path are **in scope**.
- The `local` interpreter runs code on the host and is **refused** unless explicitly
  opted into (`allow_insecure_sandbox=True` / `RLM_ALLOW_INSECURE_SANDBOX=1`).
  Enabling it is host RCE by design — out of scope.
- The opt-in `container` interpreter runs the REPL inside an isolated Docker container
  (`--network=none`, capabilities dropped, LM credentials kept host-side) so model code can
  spawn subprocesses. It is a *stronger* boundary than the default for that case, not a
  relaxation of `local` — escapes from it are **in scope**.
- The `is_safe_url` SSRF pre-flight guard on the fetch / web-search / git-clone tool
  primitives is **in scope**. It is *syntactic* (scheme + obvious internal-address checks);
  as its docstring notes, it does not stop DNS rebinding — re-checking the *resolved*
  address at connection time is the consuming fetcher's responsibility, and so is out of
  scope for the kit itself.
- **The kit's own containment guards on the filesystem and archive tools are in scope.** A
  way past `resolve_within_root` (a `..` traversal, an absolute path, a symlink inside the
  root pointing outside it) in `read_file` / `write_file` / `edit_file` / `grep_files` /
  `git_clone`, or past `make_extract_archive_tool`'s pre-extraction validation (zip slip, a
  symlink/hardlink/device entry, an entry-count or decompressed-size bound), is a
  vulnerability in the kit. So is a failure of `make_git_clone_tool`'s credential redaction
  to remove the exact secret string it was given — noting that redaction is documented as
  best-effort and does not claim to catch a *derived* or transformed leak.
- **Host-side execution surfaces where the kit ships no executor are out of scope.**
  `run_command`, `make_git_clone_tool`, the fetch/search providers and any MCP server all
  execute on the host, outside the sandbox, through a runner / cloner / fetcher / server the
  *consumer* supplies. The kit owns the syntactic guard; the isolation of what you inject is
  yours, and an unisolated `subprocess.run` runner is model-steered RCE by construction.
  An MCP server is a trusted dependency, and the optional `guard` hook (including
  `refuse_broad_git_history`) is shape-only — never a security claim.
- Third-party skills, fetched pages, MCP tool output and any untrusted content fed to the
  model become LM context — treat them as a prompt-injection surface (documented, not a
  vulnerability in the kit).
