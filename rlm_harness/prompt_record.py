"""Record what each planner call was shown, so the request can be rendered again byte for byte.

``dspy.RLM`` drives two predictors: ``generate_action``, once per iteration, returning ``{reasoning,
code}``, and ``extract``, once when the iteration budget runs out, returning the task's output fields.
The run's adapter formats each call as one single-turn request from the predictor's signature, its
demos and the call's input values. The kit records those inputs rather than the rendered messages:
the signature once per trajectory in ``rlm_setup`` (:func:`describe_signature`), and per call only
what varies plus the sha256 of the messages actually sent (:func:`messages_sha256`). The REPL history
is the trajectory's own earlier ``main_step`` entries, so it is rebuilt at render time instead of
stored, which keeps a trace linear in its turn count.

:func:`render_steps` rebuilds each request with dspy's own ``Adapter.format`` and compares it with the
recorded hash, so every exported record says whether it is the request that was sent. A field type is
rebuilt from what an adapter reads off it, its name and, for a structured type, its JSON schema.
pydantic cannot rebuild a schema with ``$defs`` through a stand-in, so such a type has to be passed
in ``custom_types`` under its recorded annotation name, or the export raises :class:`RenderError`. A
:class:`dspy.Type` that adds its own description text is rebuilt without it and comes out
``verified: False``.

The module top is dspy-free like ``trace.py`` and ``dataset.py``. The capture side runs inside
``task.py``, which already imports dspy, and the render side imports dspy when called.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

#: The two predictors of a ``dspy.RLM``, by the role the kit records them under. The attribute
#: each lives on is dspy's and is resolved in ``_dspy_compat.rlm_predictor_roles``.
ROLE_ACTION = "action"
ROLE_EXTRACT = "extract"

#: Values of ``assistant_source`` on an exported record.
SOURCE_COMPLETION = "completion"   # the LM's raw content string, as returned
SOURCE_RENDERED = "rendered"       # rebuilt from parsed outputs by the adapter's own formatter
SOURCE_COUNTERFACTUAL = "counterfactual_submit"   # a SUBMIT's values, as the extract call's answer

# Where a counterfactual extract record's target came from.
TARGET_FINAL_OUTPUTS = "final.outputs"   # recorded by the kit (1.26.0 and later)
TARGET_FINAL_REPR = "final_repr"         # parsed back from the SUBMIT turn's `FINAL: {...}` output
TARGET_RESULT = "result"                 # the `result` event: output_field after output_model coercion


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    # `surrogatepass`: a lone surrogate reaches a prompt from a REPL output (see `trace.py`'s note on
    # `backslashreplace`), and the strict codec would raise here and lose the step's hash.
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def messages_sha256(messages: Any) -> str:
    """The digest a recorded step and a re-render are compared on.

    Canonical JSON of the whole list (sorted keys, no whitespace), so the comparison covers every
    message's role and content and nothing about how a dict happened to be ordered.
    """
    return _sha256(_canonical(messages))


def spec_sha256(spec: Mapping[str, Any]) -> str:
    """Digest of a :func:`describe_signature` result, so a step can name the signature it used."""
    return _sha256(_canonical(spec))


def input_sha256(value: Any) -> str:
    """The digest ``rlm_setup.inputs_sha256`` records for one task input, over its full value.

    Taken from the value as the caller bound it, before dspy builds the preview the model sees, so two
    runs over the same input carry the same digest however long it is. A corpus uses it for dedupe and
    for checking a run's inputs against an evaluation set. To compare, a reader reproduces this
    serialisation:

    * a ``str``: its UTF-8 bytes, with ``errors="surrogatepass"`` so a lone surrogate is encoded
      rather than raising, nothing added;
    * a staged ``LargeText`` (or a subclass): the same, over the text it was built from, so it
      matches the digest the same text gets when passed inline;
    * a pydantic model: ``model_dump(mode="json")``, then as below;
    * anything else: the UTF-8 bytes of
      ``json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)``, with json's DEFAULT
      separators (``", "`` and ``": "``), unlike the compact form :func:`messages_sha256` uses.

    A value with no stable JSON form, such as a plain object (its ``str`` holds a memory address) or
    a ``set``, hashes through ``str`` and may differ between processes, so it will not dedupe. A
    ``str`` and the JSON text of a non-string can collide (``"2"`` and ``2``).
    """
    from .large_text import LargeText

    if isinstance(value, LargeText):
        return value._sha256
    if isinstance(value, str):
        return _sha256(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return _sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# ---- capture side (called from task.py, where dspy is already imported) -------------------------


def describe_type(annotation: Any) -> dict[str, Any]:
    """What an adapter reads off a field's annotation, as JSON.

    dspy's adapters read a type in five ways: its name (``get_annotation_name``), an identity test
    against ``str``/``bool``/``int``/``float``, a ``Literal``'s arguments, an ``Enum``'s values, and
    otherwise ``pydantic.TypeAdapter(t).json_schema()``. Those are recorded, and :func:`_rebuild_type`
    builds a type that answers each of them the same way.
    """
    import pydantic
    from dspy.adapters.utils import get_annotation_name
    from dspy.primitives.repl_types import REPLHistory

    out: dict[str, Any] = {"type": get_annotation_name(annotation)}
    if annotation in (str, int, float, bool):
        out["kind"] = "builtin"
    elif annotation is REPLHistory:
        out["kind"] = "repl_history"
    elif getattr(annotation, "__origin__", None) is _literal():
        out["kind"] = "literal"
        out["args"] = _json_safe(list(annotation.__args__))
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        out["kind"] = "enum"
        out["members"] = _json_safe({m.name: m.value for m in annotation})
    else:
        out["kind"] = "schema"
        try:
            out["schema"] = _json_safe(pydantic.TypeAdapter(annotation).json_schema())
        except Exception as exc:  # an arbitrary type pydantic has no schema for
            out["kind"] = "opaque"
            out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _literal() -> Any:
    from typing import Literal

    return Literal


def describe_signature(signature: Any) -> dict[str, Any]:
    """A dspy signature as JSON: instructions, then every field in signature order.

    Each field carries ``name``, ``io`` (``input``/``output``), its type facts from
    :func:`describe_type`, and ``extra``, the field's ``json_schema_extra`` without dspy's own
    input/output marker (``desc``, ``prefix`` and, when set, ``constraints``).
    """
    fields = []
    for name, info in signature.fields.items():
        extra = dict(info.json_schema_extra or {})
        io = extra.pop("__dspy_field_type", None)
        fields.append({"name": name, "io": io, **describe_type(info.annotation),
                       "extra": _json_safe(extra)})
    return {"instructions": signature.instructions, "fields": fields}


def adapter_identity(adapter: Any) -> dict[str, str]:
    """``{"adapter": short name, "adapter_class": dotted class}`` for the adapter that formatted.

    The short name is the family whose ``format`` produced the messages. The kit's lenient JSON adapter
    changes only how a call is sent and parsed, so it is ``json`` here and re-renders through
    ``dspy.JSONAdapter``. ``None`` means dspy's default, a ``ChatAdapter``.
    """
    import dspy

    if adapter is None:
        return {"adapter": "chat", "adapter_class": "dspy.adapters.chat_adapter.ChatAdapter"}
    cls = type(adapter)
    dotted = f"{cls.__module__}.{cls.__qualname__}"
    for short, base in (("json", "JSONAdapter"), ("xml", "XMLAdapter"), ("chat", "ChatAdapter")):
        base_cls = getattr(dspy, base, None)
        if base_cls is not None and isinstance(adapter, base_cls):
            return {"adapter": short, "adapter_class": dotted}
    return {"adapter": cls.__name__, "adapter_class": dotted}


def outputs_as_json(signature: Any, prediction: Any) -> dict[str, Any]:
    """A prediction's output fields, in signature order, as the JSON an adapter would serialise."""
    from dspy.adapters.utils import serialize_for_json

    out = {}
    for name in signature.output_fields:
        value = prediction.get(name) if isinstance(prediction, Mapping) else getattr(
            prediction, name, None)
        out[name] = _json_safe(serialize_for_json(value))
    return out


# ---- render side (called by dataset.export_sft_turns(render="adapter")) -------------------------


class RenderError(ValueError):
    """A recorded step cannot be turned into a request at all (as opposed to rendering unverified)."""


def _schema_standin(name: str, schema: dict) -> type:
    """A type whose name and JSON schema are the recorded ones, which is all an adapter reads.

    pydantic regenerates the schema around what ``__get_pydantic_json_schema__`` returns and rejects
    ``$ref`` targets it did not generate, so a schema with ``$defs`` (a model nesting another, or
    ``list[Flat]``) raises here. Such a field needs the real type in ``custom_types``, keyed by the
    annotation name the trace recorded.
    """
    from pydantic_core import core_schema

    if "$defs" in schema:
        raise RenderError(
            f"the recorded type {name!r} has a schema with $defs (it is or contains a pydantic "
            f"model that references another), which cannot be replayed from the schema. Pass the "
            f"real type under that exact name: custom_types={{{name!r}: <the type>}}."
        )
    frozen = json.dumps(schema)

    class _Recorded:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            return core_schema.any_schema()

        @classmethod
        def __get_pydantic_json_schema__(cls, core: Any, handler: Any) -> Any:
            return json.loads(frozen)

    _Recorded.__name__ = _Recorded.__qualname__ = name
    return _Recorded


def _rebuild_type(field: Mapping[str, Any], custom_types: Mapping[str, Any]) -> Any:
    from typing import Literal

    from dspy.primitives.repl_types import REPLHistory

    name = field.get("type")
    if name in custom_types:
        return custom_types[name]
    kind = field.get("kind")
    if kind == "builtin":
        return {"str": str, "int": int, "float": float, "bool": bool}[name]
    if kind == "repl_history":
        return REPLHistory
    if kind == "literal":
        return Literal[tuple(field.get("args") or ())]
    if kind == "enum":
        return enum.Enum(name, dict(field.get("members") or {}))
    if kind == "schema":
        return _schema_standin(name, dict(field.get("schema") or {}))
    raise RenderError(
        f"field {field.get('name')!r} has type {name!r}, which was recorded without a schema "
        f"({field.get('error') or 'no type facts'}). Pass the real type: custom_types={{{name!r}: ...}}."
    )


def rebuild_signature(spec: Mapping[str, Any], custom_types: Mapping[str, Any] | None = None) -> Any:
    """A dspy signature equivalent, for formatting purposes, to the one :func:`describe_signature` read."""
    import dspy

    custom_types = custom_types or {}
    fields: dict[str, Any] = {}
    for f in spec.get("fields") or []:
        annotation = _rebuild_type(f, custom_types)
        maker = dspy.InputField if f.get("io") == "input" else dspy.OutputField
        # `prefix` stays in the record but is not passed back: dspy 3.4 deprecates it as having no
        # effect and warns on every field. If it ever affects formatting again, `verified` shows it.
        extra = {k: v for k, v in dict(f.get("extra") or {}).items() if k != "prefix"}
        fields[f["name"]] = (annotation, maker(**extra))
    return dspy.Signature(fields, spec.get("instructions") or "")


def build_adapter(identity: Mapping[str, Any]) -> Any:
    """The stock dspy adapter that formats the way the recorded one did.

    Only the three stock families are rebuilt. An unrecognised class is never imported from the
    trace (that would execute whatever a trace names); pass ``adapter=`` instead.
    """
    import dspy

    short = identity.get("adapter")
    if short == "json":
        return dspy.JSONAdapter()
    if short == "chat":
        return dspy.ChatAdapter()
    if short == "xml":
        return dspy.XMLAdapter()
    raise RenderError(
        f"the run was formatted by {identity.get('adapter_class')!r}, which is not a stock dspy "
        "adapter family. Pass an instance that formats the same way: adapter=<instance>."
    )


def _segments(events: Iterable[dict]) -> list[dict]:
    """Split one run's events into recorded trajectories: an ``rlm_setup`` and the steps after it.

    A recorder shared by two ``arun`` calls carries two trajectories under one ``run_id``, each
    with its own ``rlm_setup``; the history of a step is only ever its own trajectory's earlier turns.
    Steps before any ``rlm_setup`` (a trace older than the event) form no segment.
    """
    segments: list[dict] = []
    current: dict | None = None
    closed: dict | None = None   # the segment whose `final` was the previous event
    for e in events:
        t = e.get("type")
        # The kit writes a successful run's `result` straight after its `final`, with no await in
        # between, so only the event right after `final` can be that segment's result. A `result`
        # further on belongs to a later run, possibly one recorded without an `rlm_setup`.
        just_closed, closed = closed, None
        if t == "result" and just_closed is not None:
            just_closed["result"] = e.get("payload") or {}
        elif t == "rlm_setup":
            current = {"setup": e["payload"], "steps": [], "extract": None, "final": None,
                       "result": None}
            segments.append(current)
        elif current is None:
            continue
        elif t == "main_step":
            current["steps"].append(e["payload"])
        elif t == "extract":
            current["extract"] = e["payload"]
        elif t == "final":
            current["final"] = e.get("payload") or {}
            closed, current = current, None
    return segments


def _history(steps: list[dict], length: int, max_output_chars: int) -> Any:
    from dspy.primitives.repl_types import REPLEntry, REPLHistory

    ordered = sorted(steps, key=lambda p: p.get("turn", 0))
    entries = [
        REPLEntry(reasoning=p.get("reasoning") or "", code=p.get("code") or "",
                  output=p.get("output") or "")
        for p in ordered[:length]
    ]
    if len(entries) != length:
        raise RenderError(
            f"a step saw {length} REPL entries but only {len(entries)} earlier turns were recorded"
        )
    return REPLHistory(entries=entries, max_output_chars=max_output_chars)


def render_steps(
    events: list[dict],
    *,
    custom_types: Mapping[str, Any] | None = None,
    adapter: Any = None,
) -> tuple[list[dict], int]:
    """Re-render every recorded action and extract call in one run's events.

    Returns ``(records, skipped)``: one record per call that has its per-call fields, and the number
    of calls without them, which are ``main_step`` events with no ``rlm_setup`` before them (a trace
    written before 1.25.0) and turns the capture could not match to their call. A skipped call does
    not stop the rest of the run or the corpus from exporting. See ``dataset.export_sft_turns``.
    """
    records: list[dict] = []
    covered = 0
    unrenderable = 0
    for seg in _segments(events):
        setup = seg["setup"]
        covered += len(seg["steps"])
        predictors = setup.get("predictors") or {}
        for role, payload in (
            [(ROLE_ACTION, p) for p in seg["steps"]]
            + ([(ROLE_EXTRACT, seg["extract"])] if seg["extract"] else [])
        ):
            if not isinstance(payload.get("history_len"), int):
                unrenderable += 1   # the capture left this call without its fields: skip, count
                continue
            records.append(_render_one(role, payload, seg, predictors, setup,
                                       custom_types or {}, adapter))
    total = sum(1 for e in events if e.get("type") == "main_step")
    return records, total - covered + unrenderable


def _request(role: str, p: dict, seg: dict, predictors: dict, setup: dict,
             custom_types: Mapping[str, Any], adapter: Any) -> tuple[Any, Any, dict, list]:
    """``(signature, adapter, identity, messages)`` for one call, from the recorded fields.

    ``p`` holds the per-call fields; whatever it lacks comes from ``setup``, as it was recorded.
    """
    spec = p.get("signature") or predictors.get(role)
    if spec is None:
        raise RenderError(f"no {role} signature was recorded for this run")
    signature = rebuild_signature(spec, custom_types)
    identity = {k: p.get(k, setup.get(k)) for k in ("adapter", "adapter_class")}
    fmt = adapter if adapter is not None else build_adapter(identity)

    inputs: dict[str, Any] = {
        "variables_info": p.get("variables_info", setup.get("variables_info")),
        "repl_history": _history(seg["steps"], p["history_len"],
                                 p.get("max_output_chars", setup.get("max_output_chars"))),
    }
    if role == ROLE_ACTION:
        inputs["iteration"] = p.get("iteration")
    demos = list(p.get("demos", spec.get("demos")) or [])
    return signature, fmt, identity, fmt.format(signature, demos, inputs)


def _render_one(role: str, p: dict, seg: dict, predictors: dict, setup: dict,
                custom_types: Mapping[str, Any], adapter: Any) -> dict:
    signature, fmt, identity, messages = _request(role, p, seg, predictors, setup,
                                                  custom_types, adapter)
    length = p["history_len"]

    recorded = p.get("messages_sha256")
    record: dict[str, Any] = {
        "kind": role,
        "turn": p.get("turn") if role == ROLE_ACTION else length,
        "adapter": identity.get("adapter"),
        "messages": messages,
        "verified": None if recorded is None else messages_sha256(messages) == recorded,
    }
    # An empty completion is never the target. A reasoning server can return the whole turn in
    # `reasoning_content` with content `""` or null, which `_LenientJSONAdapter` parses as the
    # answer, and a target of `""` would teach the model to reply with nothing.
    if isinstance(p.get("completion"), str) and p["completion"] != "":
        content, source = p["completion"], SOURCE_COMPLETION
    else:
        outputs = (
            {"reasoning": p.get("reasoning"), "code": p.get("code_raw", p.get("code"))}
            if role == ROLE_ACTION else dict(p.get("outputs") or {})
        )
        content = fmt.format_assistant_message_content(signature, outputs)
        source = SOURCE_RENDERED
    record["assistant"] = {"role": "assistant", "content": content}
    record["assistant_source"] = source
    if p.get("reasoning_content") is not None:
        record["reasoning_content"] = p["reasoning_content"]
    for key in ("usage", "run_tag", "attempt", "parse_wrapped"):
        if key in p:
            record[key] = p[key]
    return record


def _submitted_outputs(seg: dict) -> tuple[dict, str] | None:
    """The values a segment's SUBMIT returned, and where they were read from, or ``None``.

    ``final.outputs`` is what the kit recorded. A trace without it still has the SUBMIT turn's
    output, which dspy writes as ``f"FINAL: {parsed_outputs}"``, a Python repr. That parses back
    only when every value is a literal (str, number, bool, None, list, dict). A pydantic value's repr
    does not, and the value is read from the run's ``result`` event instead (:func:`_from_result`).
    When neither works the segment is skipped rather than given a guessed target.
    """
    outputs = (seg["final"] or {}).get("outputs")
    if isinstance(outputs, dict):
        return outputs, TARGET_FINAL_OUTPUTS
    text = (seg["steps"][-1].get("output") if seg["steps"] else None) or ""
    if not text.startswith("FINAL: "):
        return None
    try:
        parsed = ast.literal_eval(text[len("FINAL: "):])
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return _from_result(seg)
    fields = list(seg["setup"].get("output_fields") or [])
    if not isinstance(parsed, dict) or list(parsed) != fields:
        return None
    return parsed, TARGET_FINAL_REPR


def _repr_as_submitted(value: Any, schema: Mapping[str, Any], defs: Mapping[str, Any]) -> str:
    """The Python repr ``value`` would have if it were the parsed SUBMIT value ``schema`` describes.

    A JSON object whose schema is a model with properties is written the way pydantic's ``__repr__``
    writes it, ``Title(field=repr, ...)``, recursing through ``$ref``, ``items`` and
    ``additionalProperties``; everything else is ``repr(value)``. A value the JSON dump changed (a
    ``datetime``, an ``Enum``, a ``set``, a nested model stringified inside a list) comes out
    different from the real repr, which is the point.
    """
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        schema = defs.get(ref[len("#/$defs/"):]) or {}
    if "anyOf" in schema and value is not None:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        schema = options[0] if len(options) == 1 else {}
    props = schema.get("properties")
    if isinstance(value, dict) and isinstance(props, dict) and isinstance(schema.get("title"), str):
        inner = ", ".join(f"{k}={_repr_as_submitted(v, props.get(k) or {}, defs)}"
                          for k, v in value.items())
        return f"{schema['title']}({inner})"
    if isinstance(value, list):
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return "[" + ", ".join(_repr_as_submitted(v, items, defs) for v in value) + "]"
    if isinstance(value, dict):
        extra = schema.get("additionalProperties")
        extra = extra if isinstance(extra, dict) else {}
        return "{" + ", ".join(f"{k!r}: {_repr_as_submitted(v, extra, defs)}"
                               for k, v in value.items()) + "}"
    return repr(value)


def _from_result(seg: dict) -> tuple[dict, str] | None:
    """The SUBMIT value read from the run's ``result`` event, or ``None`` when it cannot be proven.

    ``result`` holds only the task's ``output_field``, after ``output_model`` coercion and a
    Python-mode ``model_dump`` written through ``json.dumps(default=str)``, so on its own it can
    differ from the SUBMIT: defaults and validators ``output_model`` adds, a ``datetime`` or ``Enum``
    turned into its ``str``, a nested model stringified inside a list. It stands for the SUBMIT only
    when the signature has that one output field and re-writing it as the parsed SUBMIT's repr
    reproduces the SUBMIT turn's ``FINAL: {...}`` output exactly. dspy parses every SUBMIT value
    against the field's annotation before writing that line, so equal text means an equal value.
    """
    if "output" not in (seg["result"] or {}):
        return None
    spec = (seg["setup"].get("predictors") or {}).get(ROLE_EXTRACT) or {}
    outs = [f for f in spec.get("fields") or [] if f.get("io") == "output"]
    if len(outs) != 1 or [f.get("name") for f in outs] != list(seg["setup"].get("output_fields") or []):
        return None
    field, value = outs[0], seg["result"]["output"]
    schema = field.get("schema") if field.get("kind") == "schema" else {}
    schema = schema if isinstance(schema, dict) else {}
    text = (seg["steps"][-1].get("output") if seg["steps"] else None) or ""
    wrote = _repr_as_submitted(value, schema, schema.get("$defs") or {})
    if text != f"FINAL: {{{field['name']!r}: {wrote}}}":
        return None
    return {field["name"]: value}, TARGET_RESULT


def render_counterfactual_extracts(
    events: list[dict],
    *,
    cut_back: Iterable[int] = (0,),
    custom_types: Mapping[str, Any] | None = None,
    adapter: Any = None,
) -> tuple[list[dict], int]:
    """Extract calls a successful SUBMIT run never made, with that SUBMIT's values as the answer.

    For each recorded trajectory that ended by SUBMIT in a run that succeeded, and for each ``n`` in
    ``cut_back``, renders the request dspy's extract fallback would send had the run been cut off
    with the SUBMIT turn and the ``n`` turns before it missing. Returns ``(records, skipped)``; see
    ``dataset.export_counterfactual_extracts`` for the record and for what is skipped.
    """
    cuts = sorted({int(n) for n in cut_back})
    if not cuts or cuts[0] < 0:
        raise ValueError(f"cut_back must hold integers >= 0, not {list(cut_back)!r}")
    records: list[dict] = []
    skipped = 0
    for seg in _segments(events):
        setup, steps, final = seg["setup"], seg["steps"], seg["final"] or {}
        if seg["result"] is None or seg["extract"] is not None or final.get("extract") is not False:
            continue   # the trajectory did not end by SUBMIT, or the run failed
        submitted = _submitted_outputs(seg) if steps else None
        if submitted is None:
            skipped += len(cuts)
            continue
        outputs, target_source = submitted
        submit_turn = len(steps) - 1
        predictors = setup.get("predictors") or {}
        for n in cuts:
            length = submit_turn - n
            if length < 1:
                # dspy never sends extract over an empty history: max_iterations is at least 1.
                skipped += 1
                continue
            p = {"history_len": length}
            signature, fmt, identity, messages = _request(
                ROLE_EXTRACT, p, seg, predictors, setup, custom_types or {}, adapter)
            record: dict[str, Any] = {
                "kind": ROLE_EXTRACT,
                "turn": length,
                "adapter": identity.get("adapter"),
                "messages": messages,
                "assistant": {"role": "assistant",
                              "content": fmt.format_assistant_message_content(signature, outputs)},
                "assistant_source": SOURCE_COUNTERFACTUAL,
                # No request was ever sent, so there is no hash to verify against.
                "verified": None,
                "counterfactual": {"submit_turn": submit_turn, "cut_back": n,
                                   "target_source": target_source},
            }
            for key in ("run_tag", "attempt"):
                if key in setup:
                    record[key] = setup[key]
            records.append(record)
    return records, skipped
