"""What a planner turn was SHOWN, recorded so it can be rendered again byte for byte.

``dspy.RLM`` drives two predictors. ``generate_action`` runs once per iteration and returns
``{reasoning, code}``. ``extract`` runs once, only when the iteration budget runs out, and returns
the task's own output fields. Each call is a single-turn request the run's ADAPTER formats from
three things: the predictor's signature (instructions plus typed fields), its demos, and that
call's input values. A trace that keeps only ``{reasoning, code, output}`` per turn cannot rebuild
that request, so a converter that rendered its own chat from it was training on a format the
model is never served. That is the failure this module exists for.

So the kit records the three inputs to ``Adapter.format`` instead of the rendered messages:

* the signature once per run, as :func:`describe_signature` output, in ``rlm_setup``;
* per call, only what varies: ``iteration``, how many REPL entries the history held, and any
  value that differed from the run's (``variables_info``, the signature after a runtime change);
* per call, the sha256 of the messages the adapter actually produced (:func:`messages_sha256`).

The REPL history is not stored per step, because it is the run's own earlier ``main_step``
entries truncated by ``max_output_chars`` at render time; storing it would grow the trace with
the square of the turn count. :func:`render_steps` rebuilds it from those entries, calls dspy's
own ``Adapter.format`` and then checks the result against the recorded hash, so every exported
record says whether it is the request that was sent rather than asserting it.

**Where re-rendering cannot be exact without help, it says so instead of guessing.** A field type
is rebuilt from what the adapters read off it: its name, and for a structured type its JSON
schema. pydantic will not replay a schema that carries ``$defs`` (a model nesting another model)
through a stand-in type, so such a field needs the real type passed as ``custom_types`` under
its recorded annotation name, and without it the export raises :class:`RenderError` naming it.
A :class:`dspy.Type` subclass that contributes its own description text is rebuilt without that
text, and there the hash is what reports it: the record comes out ``verified: False``.

Module top is dspy-free like ``trace.py`` and ``dataset.py``: the capture side runs inside
``task.py``, which already imports dspy, and the render side imports it when it is called.
"""

from __future__ import annotations

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
    """The digest ``rlm_setup.inputs_sha256`` records for one task input, over its FULL value.

    Taken from the value as the caller bound it, before dspy builds the preview the model sees, so
    two runs over the same input carry the same digest however long it is. That is what a corpus
    needs for dedupe, and for checking a run's inputs against an evaluation set it must not train on.

    The serialisation, which a reader must reproduce to compare:

    * a ``str``: its UTF-8 bytes (lone surrogates passed through), nothing added;
    * a staged ``LargeText``: the same, over the text it was built from, so it matches the
      digest the same text gets when passed inline;
    * a pydantic model: ``model_dump(mode="json")``, then as below;
    * anything else: the UTF-8 bytes of
      ``json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)``.
    """
    staged = getattr(value, "_sha256", None)
    if isinstance(staged, str) and type(value).__name__ == "LargeText":
        return staged
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

    dspy's adapters consult a type in exactly these ways: its NAME (``get_annotation_name``), an
    identity test against ``str``/``bool``/``int``/``float``, a ``Literal``'s arguments, an
    ``Enum``'s values, and otherwise ``pydantic.TypeAdapter(t).json_schema()``. So those are what is
    recorded, and :func:`_rebuild_type` builds something that answers each of them the same way.
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

    The short name is the family whose ``format`` produced the messages: the kit's lenient JSON
    adapter only changes how a call is sent and parsed, so it reads ``json`` here and re-renders
    through ``dspy.JSONAdapter``. ``None`` means dspy's own default, a ``ChatAdapter``.
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

    pydantic regenerates the schema around what ``__get_pydantic_json_schema__`` returns, and its
    ``$ref`` bookkeeping refuses references to definitions it did not generate itself, so a schema
    with ``$defs`` raises here. That is any model nesting another, and also a container of a model
    (``list[Flat]``). Such a field needs the real type through ``custom_types``, keyed by the
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
        # `prefix` stays in the record (it is part of what dspy built) but is not passed back:
        # dspy 3.4 deprecates the argument, says it "has no effect", and warns on every field.
        # Whether that is still true is exactly what the sha256 check would catch.
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
    for e in events:
        t = e.get("type")
        if t == "rlm_setup":
            current = {"setup": e["payload"], "steps": [], "extract": None}
            segments.append(current)
        elif current is None:
            continue
        elif t == "main_step":
            current["steps"].append(e["payload"])
        elif t == "extract":
            current["extract"] = e["payload"]
        elif t == "final":
            current = None
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

    Returns ``(records, skipped)``: one record per call that carries its per-call fields, and the
    number of calls that could not be rendered for lack of them. That count covers ``main_step``
    events with no ``rlm_setup`` before them (a trace written before 1.25.0) and steps inside a
    segment that the capture left without fields (a turn whose ``reasoning`` did not match its call,
    or an adapter whose ``format`` arguments the capture could not read). One such step does not
    stop the rest of the run, or the corpus, from exporting. See ``dataset.export_sft_turns``.
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


def _render_one(role: str, p: dict, seg: dict, predictors: dict, setup: dict,
                custom_types: Mapping[str, Any], adapter: Any) -> dict:
    spec = p.get("signature") or predictors.get(role)
    if spec is None:
        raise RenderError(f"no {role} signature was recorded for this run")
    signature = rebuild_signature(spec, custom_types)
    identity = {k: p.get(k, setup.get(k)) for k in ("adapter", "adapter_class")}
    fmt = adapter if adapter is not None else build_adapter(identity)

    length = p["history_len"]
    inputs: dict[str, Any] = {
        "variables_info": p.get("variables_info", setup.get("variables_info")),
        "repl_history": _history(seg["steps"], length,
                                 p.get("max_output_chars", setup.get("max_output_chars"))),
    }
    if role == ROLE_ACTION:
        inputs["iteration"] = p.get("iteration")
    demos = list(p.get("demos", spec.get("demos")) or [])
    messages = fmt.format(signature, demos, inputs)

    recorded = p.get("messages_sha256")
    record: dict[str, Any] = {
        "kind": role,
        "turn": p.get("turn") if role == ROLE_ACTION else length,
        "adapter": identity.get("adapter"),
        "messages": messages,
        "verified": None if recorded is None else messages_sha256(messages) == recorded,
    }
    # An EMPTY completion is not used as the target. A reasoning server can return its whole turn
    # in `reasoning_content` with content `""` or null; `_LenientJSONAdapter` then parsed the
    # reasoning channel as the answer, and an assistant target of `""` would teach an empty reply.
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
    for key in ("usage", "run_tag", "attempt"):
        if key in p:
            record[key] = p[key]
    return record
