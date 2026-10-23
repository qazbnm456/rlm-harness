"""The two premises behind the staged-input instruction, against a REAL Deno/Pyodide sandbox.

`staged_instructions` tells the model to keep the text under a new name and to seek by byte in
binary mode. Both answer failures that are SILENT or ASCII-invisible, so a fake `execute` proves
neither. These need Deno and skip without it (CI has none; see ``docs/VERIFY.md``), so run them
locally on any change to `large_text.py` or to the dspy pin.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest

from rlm_harness.large_text import LargeText

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("dspy") is None, reason="dspy not installed"),
    pytest.mark.skipif(shutil.which("deno") is None, reason="needs a real Deno on PATH"),
]

#: Citation markers around Chinese text: ASCII structure, three bytes per character between.
TEXT = "".join(f"[[SRC:s{i}|p{i}]]\n這是第{i}段引用文字，內容為中文。\n" for i in range(200))


@pytest.fixture
def staged():
    from rlm_harness.sandbox import build_interpreter

    with LargeText(TEXT) as text:
        interp = build_interpreter("pyodide", read_paths=(text.host_path,))
        try:
            yield interp, {"sources": text.sandbox_path}
        finally:
            interp.shutdown()


def test_reading_into_the_input_name_lasts_one_step(staged):
    """dspy says "State persists between iterations", true of every name except an input's."""
    interp, variables = staged
    interp.execute("sources = open(sources, encoding='utf-8').read()", variables=variables)
    out = interp.execute("print(len(sources), sources.find('[[SRC:s5'))", variables=variables)
    assert out.split() == [str(len(variables["sources"])), "-1"], (
        "the input name now survives a step; the 'never reuse the name' sentence can go"
    )
    interp.execute("corpus = open(sources, encoding='utf-8').read()", variables=variables)
    assert interp.execute("print(len(corpus))", variables=variables).strip() == str(len(TEXT))


def test_text_mode_seek_breaks_inside_a_character_and_the_byte_read_does_not(staged):
    interp, variables = staged
    code = """
broken = 0
for off in range(120):
    f = open(sources, encoding='utf-8')
    f.seek(off)
    try:
        f.read(10)
    except UnicodeDecodeError:
        broken += 1
ok = 0
for off in range(120):
    f = open(sources, 'rb')
    f.seek(off)
    f.read(30).decode('utf-8', errors='replace')
    ok += 1
print(broken, ok)
"""
    broken, ok = interp.execute(code, variables=variables).split()
    assert int(broken) > 0, "text-mode seek no longer breaks mid-character; the 'rb' advice is moot"
    assert ok == "120"
