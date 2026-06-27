"""Verify that backtick function references in agents/INDEX.md exist in the named modules.

The ``agents`` package uses a clean "Key functions: ``name(args)``" format in its
INDEX.md, making it straightforward to check that every function name listed there
actually exists as a module-level attribute.  The original drift (``format_bonus_card``
→ ``format_bonus``, ``format_game_state`` → ``format_board``) was found here, and
this test prevents it from regressing.

Wider coverage of other INDEX.md files is deferred: many other packages interleave
prose references to class methods with module-level functions, requiring a more
sophisticated check than simple ``hasattr`` lookup.  Extend this test with a
``_CHECKED_PACKAGES`` entry when a package's INDEX.md format is clean enough to
verify.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_SRC_ROOT = Path(__file__).parent.parent / "src" / "wingspan"

# Packages whose INDEX.md uses an explicit module-attribute format clean enough
# to verify automatically.  Add a package name here when its INDEX.md consistently
# lists module-level exports (not class methods or prose code examples).
_CHECKED_PACKAGES = ("agents",)

# Matches the module header line: **`module.py`**
_MODULE_HEADER = re.compile(r"\*\*`([a-z_]+\.py)`\*\*")
# Matches an inline function reference: `name(`
_FUNC_REF = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)\(")


def _parse_index(index_path: Path) -> dict[str, list[str]]:
    """Return ``{module_stem: [referenced_name, ...]}``, one entry per module section."""
    result: dict[str, list[str]] = {}
    current = ""  # empty string: no module section yet

    for line in index_path.read_text(encoding="utf-8").splitlines():
        header = _MODULE_HEADER.search(line)
        if header:
            current = header.group(1).removesuffix(".py")
            result.setdefault(current, [])
        if current:
            for func_match in _FUNC_REF.finditer(line):
                name = func_match.group(1)
                if name not in result[current]:
                    result[current].append(name)

    return result


def test_agents_index_md_function_references() -> None:
    """Every `func(` reference in agents/INDEX.md must exist in the named module."""
    failures: list[str] = []

    for package_name in _CHECKED_PACKAGES:
        index_path = _SRC_ROOT / package_name / "INDEX.md"
        if not index_path.exists():
            continue
        refs = _parse_index(index_path)

        for module_stem, func_names in refs.items():
            full_module = f"wingspan.{package_name}.{module_stem}"
            try:
                mod = importlib.import_module(full_module)
            except ImportError:
                continue

            for name in func_names:
                if not hasattr(mod, name):
                    failures.append(
                        f"{full_module}: `{name}(` referenced in INDEX.md but not found"
                    )

    assert not failures, "INDEX.md references not found in modules:\n" + "\n".join(
        failures
    )
