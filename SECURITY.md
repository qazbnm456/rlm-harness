# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for a security vulnerability. Instead, DM **Boik Su ([@boik_su](https://x.com/boik_su))** on X with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept), and
- the affected version / commit.

You will receive an acknowledgement, and a fix or mitigation will be coordinated
before any public disclosure. Thank you for reporting responsibly.

## Supported versions

`rlm-kit` is pre-1.0; only the latest released version receives security fixes.

| Version | Supported |
| ------- | --------- |
| latest  | ✅        |
| older   | ❌        |

## Scope notes

`rlm-kit` executes model-written code, so the interpreter choice is the security
boundary:

- The **default** interpreter is sandboxed (`pyodide`/`deno`). Sandbox-escape or
  isolation issues on the default path are **in scope**.
- The `local` interpreter runs code on the host and is **refused** unless explicitly
  opted into (`allow_insecure_sandbox=True` / `RLM_ALLOW_INSECURE_SANDBOX=1`).
  Enabling it is host RCE by design — out of scope.
- The `is_safe_url` SSRF pre-flight guard on the fetch / web-search tool primitives is
  **in scope**. It is *syntactic* (scheme + obvious internal-address checks); as its
  docstring notes, it does not stop DNS rebinding — re-checking the *resolved* address at
  connection time is the consuming fetcher's responsibility, and so is out of scope for the
  kit itself.
- Third-party skills and any untrusted content fed to the model become LM context —
  treat them as a prompt-injection surface (documented, not a vulnerability in the kit).
