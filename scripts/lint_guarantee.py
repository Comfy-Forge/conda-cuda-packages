#!/usr/bin/env python3
"""Assert the from-source guarantee cannot have silently regressed.

The guarantee is "the build script cannot reach the network", and it has
exactly two ways to break without any test failing:

  * someone passes --allow-network, or
  * someone drops --sandbox

Both are invisible in a green CI run: the build still succeeds, the package
still installs, the kernels still work. It is just not our binary any more.

A grep is too blunt for this — it flags the prose explaining the rule, and
`grep` is not the same program on every machine (ugrep locally, GNU grep on
CI). So: parse the workflow files, strip comments, join line continuations,
and inspect the actual rattler-build invocations.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEARCH = [REPO / ".github", REPO / "scripts", REPO / "templates"]
SUFFIXES = {".yml", ".yaml", ".sh", ".py", ".j2"}


def invocations(text: str):
    """Yield logical rattler-build command lines with comments removed."""
    # strip whole-line comments (# for sh/yaml, :: for the win branch)
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#") or stripped.startswith("::"):
            continue
        lines.append(raw)
    joined = []
    buf = ""
    for line in lines:
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        buf += line
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    for line in joined:
        if re.search(r"\brattler-build\s+build\b", line):
            yield line.strip()


def main() -> int:
    problems: list[str] = []
    checked = 0
    for root in SEARCH:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(errors="replace")
            rel = path.relative_to(REPO)
            for cmd in invocations(text):
                checked += 1
                if "--allow-network" in cmd:
                    problems.append(
                        f"{rel}: passes --allow-network, which defeats the "
                        f"from-source guarantee. Vendor whatever the build needs "
                        f"in scripts/fetch_patched_sources.py instead.\n    {cmd}")
                if "--sandbox" not in cmd:
                    problems.append(
                        f"{rel}: rattler-build invoked WITHOUT --sandbox. The "
                        f"sandbox is not the default (measured): without it the "
                        f"build script has full network access.\n    {cmd}")

    if checked == 0:
        print("lint: no rattler-build invocations found — did the workflows move?",
              file=sys.stderr)
        return 1
    if problems:
        print("::error::from-source guarantee lint failed", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print(f"lint: {checked} rattler-build invocation(s), all sandboxed, "
          f"none allowing network")
    return 0


if __name__ == "__main__":
    sys.exit(main())
