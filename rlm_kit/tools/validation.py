"""Schema-validation tool factories.

Two shapes, both consumer-facing base primitives:

- ``make_schema_validator(model)`` — a plain callable the RLM invokes inside the REPL to
  check its draft JSON against a pydantic schema before emitting a final answer (returns a
  human message). The generalised form of the original app's ``validate_vulnerability_report``.
- ``make_json_schema_validator(schema)`` — validate a PARSED object against a JSON Schema
  (draft 2020-12) and return the violation messages, the generic base for the "validate
  against a vendored, version-pinned upstream JSON schema" pattern (a consumer vendors the
  schema + a refresh script; the kit owns the validator wiring).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Type, Union

from pydantic import BaseModel


def make_json_schema_validator(
    schema: Union[dict, str, os.PathLike],
    *,
    max_errors: int = 20,
) -> Callable[[object], list[str]]:
    """Return a function that validates a PARSED object against a JSON Schema and returns a
    list of human-readable violation messages (empty list = valid).

    ``schema`` is the JSON Schema as a dict, or a path to a ``.json`` schema file (read once,
    at factory time). The returned validator takes an already-parsed object (a dict from
    ``yaml.safe_load`` / ``json.loads``) — parsing is the consumer's job, so this composes with
    whatever extract/parse step precedes it. Each message is ``"<json/pointer/path>: <reason>"``
    (root violations use ``"(root)"``); the list is truncated to ``max_errors`` (a huge invalid
    doc must not flood the trace). Deterministic ordering (by error path) so traces are stable.

    This is the GENERIC base for the "validate against an official, vendored, version-pinned
    upstream JSON schema" pattern: a consumer vendors the schema file + a refresh script (the
    provider-specific half), and layers its own bespoke checks on top; the kit owns only this
    wiring. ``jsonschema`` is an OPTIONAL dependency (``rlm-kit[jsonschema]``) — imported lazily
    so ``import rlm_kit`` and the dspy-free ``tools`` package stay lean.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "make_json_schema_validator needs the optional 'jsonschema' dependency. "
            "Install it with:  pip install 'rlm-kit[jsonschema]'"
        ) from exc

    if isinstance(schema, (str, os.PathLike)):
        with open(schema, encoding="utf-8") as fh:
            schema = json.load(fh)
    validator = Draft202012Validator(schema)

    def validate(obj: object) -> list[str]:
        """Validate a parsed object against the JSON schema; return violation messages ([] = ok)."""
        errors: list[str] = []
        for err in sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path)):
            loc = "/".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(f"{loc}: {err.message}")
            if len(errors) >= max_errors:
                errors.append(f"… (schema errors truncated at {max_errors})")
                break
        return errors

    return validate


def make_schema_validator(model: Type[BaseModel]) -> Callable[[str], str]:
    """Return a tool that validates a JSON string against ``model``.

    The returned function's name and docstring are set so DSPy surfaces it to the
    model with useful tool metadata.
    """

    def validate(data_json_str: str) -> str:
        """Validate a JSON string against the expected output schema. Pass your
        generated JSON here before emitting it as the final answer."""
        try:
            model.model_validate_json(data_json_str)
            return "Validation successful. You may now output this JSON string."
        except Exception as exc:  # noqa: BLE001 — surfaced back to the model as text
            return f"Validation failed: {exc}"

    validate.__name__ = f"validate_{model.__name__.lower()}"
    validate.__qualname__ = validate.__name__
    return validate
