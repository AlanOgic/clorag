"""One-shot migration helper: rewrite inline onclick handlers in admin
templates to data-action attributes compatible with AdminActions in
admin.js. Leaves non-click events (onchange/onsubmit/oninput) and
complex expressions untouched — those cases must be hand-migrated to
page-level addEventListener blocks.

Usage:
    python scripts/migrate_inline_handlers.py [--apply]

Without --apply the script prints a per-file summary of what would
change; with --apply it writes the files in place and prints remaining
(skipped) handlers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "src/clorag/web/templates"

# onclick="<expr>" with balanced double-quoted value
INLINE_CLICK_RE = re.compile(r'onclick="([^"]*)"')

# Simple call forms we auto-rewrite:
#   funcName()
CALL_NO_ARGS_RE = re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*\(\s*\)\s*;?\s*$")
# Single argument: number or single/double-quoted string with no embedded quotes
CALL_SINGLE_ARG_RE = re.compile(
    r"""^\s*([A-Za-z_$][\w$]*)\s*\(\s*
        (?:
            (?P<num>-?\d+(?:\.\d+)?)
            |
            '(?P<sq>[^'\\]*)'
            |
            "(?P<dq>[^"\\]*)"
        )
        \s*\)\s*;?\s*$""",
    re.VERBOSE,
)


def attr_escape(value: str) -> str:
    """Escape a string for safe insertion as an HTML attribute value."""
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def rewrite_handler(expr: str) -> str | None:
    """Return the replacement attribute string for a handler expression,
    or None if the expression is too complex to auto-rewrite.
    """
    # closeModal('id') / closeModal("id")
    m = re.match(r"""^\s*closeModal\(\s*['"]([^'"]+)['"]\s*\)\s*;?\s*$""", expr)
    if m:
        return f'data-action="close-modal" data-modal="{attr_escape(m.group(1))}"'
    m = re.match(r"""^\s*openModal\(\s*['"]([^'"]+)['"]\s*\)\s*;?\s*$""", expr)
    if m:
        return f'data-action="open-modal" data-modal="{attr_escape(m.group(1))}"'

    m = CALL_NO_ARGS_RE.match(expr)
    if m:
        return f'data-action="call" data-fn="{m.group(1)}"'

    m = CALL_SINGLE_ARG_RE.match(expr)
    if m:
        fn = m.group(1)
        if m.group("num") is not None:
            arg = m.group("num")
        elif m.group("sq") is not None:
            arg = m.group("sq")
        else:
            arg = m.group("dq") or ""
        return f'data-action="call" data-fn="{fn}" data-args="{attr_escape(arg)}"'

    return None


def migrate_file(path: Path, apply: bool) -> tuple[int, int, list[str]]:
    original = path.read_text()
    rewrites = 0
    skipped_exprs: list[str] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal rewrites
        expr = match.group(1).strip()
        rewritten = rewrite_handler(expr)
        if rewritten is None:
            skipped_exprs.append(expr)
            return match.group(0)
        rewrites += 1
        return rewritten

    new_text = INLINE_CLICK_RE.sub(replace, original)
    total_handlers = rewrites + len(skipped_exprs)
    if apply and rewrites > 0:
        path.write_text(new_text)
    return rewrites, total_handlers, skipped_exprs


def main() -> int:
    apply = "--apply" in sys.argv
    total_rewrites = 0
    total_skipped = 0
    for template in sorted(TEMPLATES_DIR.glob("admin*.html")):
        rewrites, total, skipped = migrate_file(template, apply=apply)
        if total == 0:
            continue
        total_rewrites += rewrites
        total_skipped += len(skipped)
        status = "APPLIED" if apply else "DRY-RUN"
        print(f"[{status}] {template.name}: {rewrites}/{total} rewritten")
        for expr in skipped:
            print(f"    SKIP: onclick=\"{expr}\"")
    print()
    print(f"Total rewritten: {total_rewrites}")
    print(f"Total skipped  : {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
