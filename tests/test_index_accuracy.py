"""Guard every INDEX.md in the wingspan source tree against symbol drift.

Three test functions — two universal (run against all INDEX.md files) and one
curated (symbol-checked packages only):

* ``test_index_module_headers_exist`` — every ``**`module.py`**`` header in
  every INDEX.md resolves to a real file relative to the INDEX directory.
* ``test_index_documents_all_modules`` — every non-dunder ``*.py`` file in
  each INDEX directory has a ``**`...`**`` header.
* ``test_index_symbol_references_exist`` — for packages in
  ``_CHECKED_PACKAGES``, every backtick ``name(`` call-form reference must
  resolve as a module-level attribute, a class defined in the module, a method
  on such a class, or a builtin.

Known gaps (documented, not checked here):

* ``reporting/`` and ``training/`` — heavily prose-mixed, reference
  cross-package call forms; their module headers are still guarded by the two
  universal tests.
* ``instrumentation/handlers/`` — no dedicated INDEX.md; handlers are
  documented inline in ``instrumentation/INDEX.md`` with slash headers.
* Bare class-name references (no trailing ``(``) and Pydantic field names are
  not validated — the ``name(`` filter keeps false-positive rates low.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import re
from pathlib import Path

_SRC_ROOT = Path(__file__).parent.parent / "src" / "wingspan"

_MODULE_HEADER = re.compile(r"\*\*`([a-z_][a-z0-9_/]*\.py)`\*\*")
_SYMBOL_REF = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)\(")

_BUILTIN_NAMES = frozenset(dir(builtins))

# Packages whose INDEX.md symbol references are validated.
# Maps the directory path relative to src/wingspan/ to the Python import prefix.
_CHECKED_PACKAGES: dict[str, str] = {
    "agents": "wingspan.agents",
    "cards/parse": "wingspan.cards.parse",
    "cloud": "wingspan.cloud",
    "compat": "wingspan.compat",
    "encode": "wingspan.encode",
    "encode/stripes": "wingspan.encode.stripes",
    "engine": "wingspan.engine",
    "engine/powers": "wingspan.engine.powers",
    "instrumentation": "wingspan.instrumentation",
    "model": "wingspan.model",
    "players": "wingspan.players",
    "setup_model": "wingspan.setup_model",
    "tournament": "wingspan.tournament",
    "training/charts": "wingspan.training.charts",
    "training/configure": "wingspan.training.configure",
}

# Symbol tokens that appear in prose inside INDEX files and look like function
# calls but are not importable names. Prefer rephrasing the INDEX over growing
# this set.
_PROSE_SKIP: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Parsing helpers


def _all_index_dirs() -> list[tuple[Path, Path]]:
    """(index_file, package_dir) for every INDEX.md under _SRC_ROOT."""
    return [
        (index_file, index_file.parent)
        for index_file in sorted(_SRC_ROOT.rglob("INDEX.md"))
    ]


def _parse_headers(index_path: Path) -> list[str]:
    """Every module.py path from ``**`...`**`` headers in *index_path*."""
    return _MODULE_HEADER.findall(index_path.read_text(encoding="utf-8"))


def _parse_sections(index_path: Path) -> dict[str, list[str]]:
    """Return ``{module_stem: [symbol, ...]}`` for each module-header section.

    Slash headers (e.g. ``handlers/card_visits.py``) are normalised to dotted
    stems (``handlers.card_visits``) so they map to importable sub-modules.
    """
    result: dict[str, list[str]] = {}
    current = ""

    for line in index_path.read_text(encoding="utf-8").splitlines():
        header = _MODULE_HEADER.search(line)
        if header:
            current = header.group(1).removesuffix(".py").replace("/", ".")
            result.setdefault(current, [])
        if current:
            for sym in _SYMBOL_REF.finditer(line):
                name = sym.group(1)
                if name not in _PROSE_SKIP and name not in result[current]:
                    result[current].append(name)

    return result


# ---------------------------------------------------------------------------
# Symbol resolver


def _symbol_in_module(mod: object, symbol: str) -> bool:
    """True if *symbol* can be found in *mod*."""
    if symbol in _BUILTIN_NAMES:
        return True

    # Dotted form ``ClassName.method`` — look up the class then its attribute
    if "." in symbol:
        class_name, attr_name = symbol.split(".", 1)
        cls = getattr(mod, class_name, None)
        return cls is not None and hasattr(cls, attr_name)

    # Module-level attribute (function, constant, class, etc.)
    if hasattr(mod, symbol):
        return True

    # Method on any class *defined* in this module
    mod_name: str = getattr(mod, "__name__", "")
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if getattr(cls, "__module__", "") == mod_name and hasattr(cls, symbol):
            return True

    return False


# ---------------------------------------------------------------------------
# Tests


def test_index_module_headers_exist() -> None:
    """Every ``**`module.py`**`` header in every INDEX.md resolves to a real file."""
    failures: list[str] = []

    for index_file, pkg_dir in _all_index_dirs():
        for ref in _parse_headers(index_file):
            target = pkg_dir / ref
            if not target.exists():
                failures.append(
                    f"{index_file.relative_to(_SRC_ROOT.parent.parent)}: "
                    f"**`{ref}`** not found (expected {target})"
                )

    assert not failures, "INDEX.md headers reference missing files:\n" + "\n".join(
        failures
    )


def test_index_documents_all_modules() -> None:
    """Every non-dunder ``*.py`` in each INDEX directory must have a header."""
    failures: list[str] = []

    for index_file, pkg_dir in _all_index_dirs():
        documented_files: set[Path] = {
            (pkg_dir / ref).resolve() for ref in _parse_headers(index_file)
        }
        for py_file in sorted(pkg_dir.glob("*.py")):
            if py_file.stem.startswith("__"):
                continue
            if py_file.resolve() not in documented_files:
                failures.append(
                    f"{index_file.relative_to(_SRC_ROOT.parent.parent)}: "
                    f"`{py_file.name}` not documented"
                )

    assert not failures, "Undocumented modules in INDEX.md files:\n" + "\n".join(
        failures
    )


def test_index_symbol_references_exist() -> None:
    """Backtick ``name(`` references in checked packages must exist in their module."""
    failures: list[str] = []

    for pkg_rel_path, import_prefix in sorted(_CHECKED_PACKAGES.items()):
        index_file = _SRC_ROOT / pkg_rel_path / "INDEX.md"
        if not index_file.exists():
            failures.append(f"Missing INDEX.md: src/wingspan/{pkg_rel_path}/INDEX.md")
            continue

        for mod_stem, symbols in _parse_sections(index_file).items():
            # Skip __init__ sections; they document the package's re-export API
            if mod_stem == "__init__" or mod_stem.endswith(".__init__"):
                continue

            full_mod = f"{import_prefix}.{mod_stem}"
            try:
                mod = importlib.import_module(full_mod)
            except ImportError:
                continue

            for symbol in symbols:
                if not _symbol_in_module(mod, symbol):
                    failures.append(
                        f"src/wingspan/{pkg_rel_path}/INDEX.md "
                        f"[{mod_stem}]: `{symbol}(` not found"
                    )

    assert (
        not failures
    ), "INDEX.md symbol references not found in modules:\n" + "\n".join(failures)
