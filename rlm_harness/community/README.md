# Community routes: models on a vendor's consumer sign-in

**Read this before you rely on any route below. Each one runs a role on a vendor's consumer plan (a subscription you sign in to) through that vendor's own CLI or SDK, not through its API with a key. They work as of the dates in the table, and nothing guarantees they keep working. A vendor can change its terms, or how it enforces them, at any time, and a breach can get the account suspended. This kit is not affiliated with any vendor, makes no promise that a route stays usable, and does not interpret anyone's terms: the table quotes what each vendor has said, and whether your workload fits your plan is for you to check.**

The same risk applies to dspy's own sign-in providers (`claude-code:`, `openai-codex:`, `xai:`), which the guide covers under [Subscription logins](../README.md#subscription-logins-this-kits-adapters-and-dspys-own-providers). The table lists only the routes this kit ships.

## Routes

| Prefix | Class | Vendor permission | What the vendor says | Enforcement reports |
|---|---|---|---|---|
| `claude-agent-sdk/` | `ClaudeAgentLM` | explicit | Anthropic's support article [Use the Claude Agent SDK with your Claude plan](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan), update of 2026-10-07: "when you're signed in with your Claude plan, the Claude Agent SDK and `claude -p` still draw from your subscription limits". An update of 2026-06-15 paused a previously announced change to that usage. Read 2026-11-14. | None known for the Agent SDK path. |
| `agy/` | `AntigravityLM` | not explicit | **Google's FAQ says that wrapping `agy` inside third-party coding agents or applications is not supported under consumer plans. To use Gemini from this kit within Google's terms, use a Google AI Studio API key with the `gemini/` prefix instead.** Google's [Headless mode](https://antigravity.google/docs/cli/headless/) page: "Use it whenever you need the agent’s output in a program instead of a terminal UI." Its [FAQ](https://antigravity.google/docs/faq), under "Can I use Antigravity CLI headless mode in third-party coding agents or automate account switching?": "Antigravity CLI headless mode (`agy -p`) is intended for direct scripting and CI workflows. Wrapping `agy` inside third-party coding agents or applications, or automating account rotation, is not supported under consumer plans." Item 6 of the [Google Antigravity Additional Terms of Service](https://antigravity.google/terms): "You must not abuse, harm, interfere with, or disrupt the Service. This includes, but is not limited to, using the Service in connection with products not provided by us. Using third party software, tools, or services to access the Service (e.g. using OpenClaw with Antigravity OAuth) is a breach of this Agreement. Such actions may be grounds for suspension or termination of your Antigravity and/or Gemini CLI accounts." None of the three pages shows a date. Read 2026-11-14. | None known with a stated cause for the `agy` print-mode path. Item 6's own example, OpenClaw reaching Antigravity through its OAuth, led to suspensions in February 2026 ([a forum thread](https://discuss.ai.google.dev/t/account-restricted-without-warning-google-ai-ultra-oauth-via-openclaw/122778)). |

"Explicit" means the vendor has said in writing that this use of a plan is allowed. "Not explicit" means it has not, or that its terms read against it. A route marked not explicit logs one warning per process the first time it is built.

## How routing works

`configure()` reads `rlm_harness.community.ROUTES`. When `RLM_MAIN_MODEL` or `RLM_SUB_MODEL` starts with a route's prefix, that role is built by the route's class from the rest of the string, so `RLM_MAIN_MODEL=claude-agent-sdk/claude-sonnet-5` builds `ClaudeAgentLM("claude-sonnet-5")`. A role you pass yourself (`configure(main_lm=...)`) is never routed. A route's module is imported only when a model string names its prefix.

Every route's class takes `timeout_s=`, an end-to-end deadline for one call that includes time spent waiting for one of its concurrency slots. `request_timeout_s` does not reach it, and `configure()` says so when you set both.

## Status and versioning

`ClaudeAgentLM` and `SUBSCRIPTION_PREFIX` are also exported from `rlm_harness` and are covered by the kit's SemVer promise. Every other route is exported from `rlm_harness.community` only and is outside that promise: if its vendor objects or starts enforcing, the route can be withdrawn in a minor release, with a CHANGELOG entry, and its prefix then raises an error that says why.

## Adding a route

For kit contributors. A consumer who wants a sign-in this kit does not ship builds the LM and passes it to `configure(main_lm=...)`, which takes any `dspy.LM`.

1. Put the adapter in `rlm_harness/community/<name>.py`. Its class subclasses `dspy.LM`, takes `timeout_s=` as above, calls `rlm_harness.community._note_constructed("<ClassName>")` in its constructor, and reads its own prefix with `_route("<ClassName>").prefix` rather than repeating the literal.
2. Add one `Route` to `ROUTES`: the prefix (ending in `/`, and not the name of a provider litellm routes by API key), the module and class, an example model id for the bare-prefix error, and `vendor_permission`.
3. Add a row to the table above. Quote what the vendor says, with a link and the date you read it, and set the permission from that quote: explicit only when the vendor allows this use in writing. List any enforcement reports you found. Do not paraphrase a vendor's terms into a verdict.
4. Export the class from `rlm_harness.community` only (its lazy `__getattr__` already does this for every route). Do not add it to `rlm_harness.__all__`.
5. `tests/test_community.py` must pass. It checks the class, its `timeout_s`, the prefix against other routes and litellm's providers, and that this table has one dated, linked row for the prefix.

When a vendor's position changes, update the row and its date, and the permission with it.
