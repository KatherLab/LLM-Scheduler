"""Static checks on the vanilla-JS frontend.

`node --check` only validates syntax, so a function that is *called* but never
*defined* parses cleanly and then throws at render time, in the browser, on a
code path a test never exercises. That has happened twice: once when an edit
that rewrote one function silently removed the helper sitting above it.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

APP_JS = pathlib.Path(__file__).resolve().parent.parent / "app" / "ui" / "app.js"
INDEX = APP_JS.parent / "index.html"

KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "typeof",
    "new", "await", "async", "do", "else", "try", "finally", "throw", "delete",
    "void", "in", "of", "case", "yield",
}
BROWSER_GLOBALS = {
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "requestAnimationFrame", "parseInt", "parseFloat", "isNaN", "isFinite",
    "fetch", "alert", "confirm", "prompt", "encodeURIComponent",
    "decodeURIComponent", "structuredClone", "queueMicrotask", "btoa", "atob",
}


def _strip_comments_and_strings(src: str) -> str:
    """Remove comments and literals so prose and CSS cannot look like code.

    Without this, `rgba(...)` inside a style string and "a pinch (of salt)" in
    a comment both register as calls to undefined functions.
    """
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'`":
            quote, i = c, i + 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
            i += 1
            out.append('""')
        else:
            out.append(c)
            i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def code() -> str:
    return _strip_comments_and_strings(APP_JS.read_text(encoding="utf-8"))


def test_no_undefined_function_references(code):
    defined = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", code))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", code))
    defined |= set(re.findall(r"\bwindow\.([A-Za-z_$][\w$]*)\s*=", code))
    defined |= set(re.findall(r"\b([A-Za-z_$][\w$]*)\s*:\s*(?:async\s*)?(?:function|\()", code))
    called = set(re.findall(r"(?<![.\w$])([a-z_$][\w$]*)\s*\(", code))

    missing = sorted(called - defined - KEYWORDS - BROWSER_GLOBALS)
    assert not missing, f"called but never defined in app.js: {missing}"


def test_every_element_id_the_script_needs_exists_in_the_html(code):
    """A `$('#foo')` with no matching element yields null, and the next
    property access kills the whole script — blanking the page rather than
    degrading one feature."""
    html = INDEX.read_text(encoding="utf-8")
    ids_in_html = set(re.findall(r'id="([^"]+)"', html))
    # Elements the script creates at runtime rather than expecting in the page.
    created = set(re.findall(r"\.id\s*=\s*['\"]([^'\"]+)['\"]", code))

    referenced = set(re.findall(r"""\$\(\s*['"]#([\w-]+)['"]\s*\)""", code))
    missing = sorted(referenced - ids_in_html - created)
    assert not missing, f"app.js looks up #ids absent from index.html: {missing}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    proc = subprocess.run(
        ["node", "--check", str(APP_JS)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
