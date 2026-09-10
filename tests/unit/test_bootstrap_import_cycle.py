"""Regression test: a cold ``import src.bootstrap`` must not raise.

Before this fix, ``src.bootstrap`` imported ``src.api.telephony_exotel`` at
module load time, which triggers ``src/api/__init__.py``, which imports
``src.api.livekit_routes``, which did ``from src.bootstrap import
LiveKitModeNotSupported`` -- a class defined ~700 lines further down in
src/bootstrap.py than its own top-level imports. Importing src.bootstrap as
the very first touch of either module hit that half-initialized module and
raised ImportError. The class now lives in src.exceptions (no src.*
dependencies of its own), imported by both sides, so the cycle no longer
exists.

This must run in a fresh subprocess with a clean sys.modules: by the time any
other test in this suite runs, src.api (or src.bootstrap) is almost certainly
already imported by an earlier-collected test, which would make an in-process
check pass regardless of whether the cycle is actually fixed. Same shape as
tests/unit/test_rag_span_scoring.py::test_import_rag_benchmark_does_not_pull_in_agents_module.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cold_import_of_bootstrap_succeeds() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import src.bootstrap"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr


def test_cold_import_of_bootstrap_reexports_livekit_mode_not_supported() -> None:
    # Existing `from src.bootstrap import LiveKitModeNotSupported` call sites
    # (src/api/dev_console.py-style imports, if any appear later) must keep
    # working even though the class's real home moved to src.exceptions.
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "from src.bootstrap import LiveKitModeNotSupported as A; "
            "from src.exceptions import LiveKitModeNotSupported as B; "
            "assert A is B; print('ok')",
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# --- General invariant: no module reachable from src/api/__init__.py may ---
# --- top-level-import from src.bootstrap. -----------------------------------
#
# The two tests above pin down one already-fixed instance (LiveKitModeNotSupported)
# and one still-latent one (DEFAULT_DEMO_SCRIPT, moved to src.defaults). Neither
# guards the general rule, so a *third* module could reintroduce the same shape
# of cycle -- e.g. by adding src.api.dev_console to src/api/__init__.py's import
# list while it still top-level-imports something from src.bootstrap -- and
# nothing above would catch it. The test below walks the whole import graph
# reachable from src/api/__init__.py and asserts none of those modules
# top-level-import from src.bootstrap, which is the actual invariant:
# src.bootstrap imports src.api.telephony_* at module load time, which triggers
# src/api/__init__.py, which imports its sub-routers -- so ANY of those (or
# anything they in turn import) doing `from src.bootstrap import ...` at module
# level makes `import src.bootstrap` hit a half-initialized src.bootstrap
# module and raise ImportError, exactly like the LiveKitModeNotSupported bug.
# Function-local imports (`def f(): from src.bootstrap import ...`) are fine --
# they run long after both modules have finished loading -- so this only
# inspects each module's top-level statements via ast, never a regex (which
# can't distinguish an indented, function-local import from a module-level one,
# nor an import from a comment/string that merely mentions one).


class _ModuleLevelImportCollector(ast.NodeVisitor):
    """Collects a module's top-level Import/ImportFrom nodes.

    Does not descend into function/async-function bodies -- imports there are
    function-local and never run at module-import time, so they cannot cause
    the half-initialized-module failure this test guards against. Does not
    descend into an `if TYPE_CHECKING:` branch either, since that code never
    executes at runtime (it exists purely for type checkers). Does descend
    into other If/Try/With/ClassDef bodies -- those DO run at import time.
    """

    def __init__(self) -> None:
        self.import_from_nodes: list[ast.ImportFrom] = []
        self.import_nodes: list[ast.Import] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        test = node.test
        is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if is_type_checking:
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.import_from_nodes.append(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.import_nodes.append(node)


def _collect_top_level_imports(module_path: Path) -> _ModuleLevelImportCollector:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    collector = _ModuleLevelImportCollector()
    collector.visit(tree)
    return collector


def _resolve_src_module(dotted: str) -> Path | None:
    """Resolve a dotted ``src....`` module name to its file, if it exists.

    Filesystem-only (no importlib): actually importing candidate modules to
    check would re-trigger the very import machinery under test, and would
    give order-dependent results depending on what earlier tests already
    populated into sys.modules.
    """
    rel = Path(*dotted.split("."))
    file_candidate = REPO_ROOT / rel.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    pkg_candidate = REPO_ROOT / rel / "__init__.py"
    if pkg_candidate.is_file():
        return pkg_candidate
    return None


def _referenced_src_modules(collector: _ModuleLevelImportCollector) -> set[str]:
    """Dotted src.* module names a module's top-level imports might reference.

    For `from src.api import calls, chat_tools`, each of `src.api.calls` and
    `src.api.chat_tools` might be a submodule (if it resolves to a real file)
    or just a name defined inside src/api/__init__.py itself -- callers can't
    tell without checking the filesystem, so this returns both the parent
    module and every such candidate; _resolve_src_module filters out the ones
    that aren't real files.
    """
    modules: set[str] = set()
    for node in collector.import_nodes:
        for alias in node.names:
            if alias.name == "src" or alias.name.startswith("src."):
                modules.add(alias.name)
    for node in collector.import_from_nodes:
        if node.level or not node.module:
            continue  # relative import; none expected in this codebase's src/ layout
        if node.module == "src" or node.module.startswith("src."):
            modules.add(node.module)
            for alias in node.names:
                modules.add(f"{node.module}.{alias.name}")
    return modules


def _modules_reachable_from_api_init() -> dict[str, _ModuleLevelImportCollector]:
    start = REPO_ROOT / "src" / "api" / "__init__.py"
    reachable: dict[str, _ModuleLevelImportCollector] = {}
    queue: list[tuple[str, Path]] = [("src.api", start)]
    seen_paths: set[Path] = set()
    while queue:
        dotted, path = queue.pop()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        collector = _collect_top_level_imports(path)
        reachable[dotted] = collector
        for candidate_dotted in _referenced_src_modules(collector):
            candidate_path = _resolve_src_module(candidate_dotted)
            if candidate_path is not None and candidate_path not in seen_paths:
                queue.append((candidate_dotted, candidate_path))
    return reachable


def test_no_module_reachable_from_api_init_top_level_imports_bootstrap() -> None:
    """Guards the general invariant, not just the two fixed instances above.

    src.bootstrap top-level-imports src.api.telephony_* modules, which trigger
    src/api/__init__.py's own imports. If ANY module that import chain reaches
    top-level-imports something from src.bootstrap, then the first cold
    `import src.bootstrap` in a process recurses back into a half-initialized
    src.bootstrap module and raises ImportError -- the exact bug already fixed
    once for LiveKitModeNotSupported (src/exceptions.py) and once more for
    DEFAULT_DEMO_SCRIPT (src/defaults.py). Fix a future violation the same
    way: move the imported name to a module with no src.bootstrap/src.api
    dependency of its own (src.exceptions / src.defaults, or a new one), or
    make the import function-local at its use site -- do NOT special-case this
    test or delete the assertion; that puts the cycle right back.
    """
    reachable = _modules_reachable_from_api_init()
    violations: list[str] = []
    for dotted, collector in reachable.items():
        for node in collector.import_from_nodes:
            if node.module == "src.bootstrap" or (node.module or "").startswith(
                "src.bootstrap."
            ):
                names = ", ".join(alias.name for alias in node.names)
                violations.append(f"{dotted} (line {node.lineno}): from {node.module} import {names}")
        for node in collector.import_nodes:
            for alias in node.names:
                if alias.name == "src.bootstrap" or alias.name.startswith("src.bootstrap."):
                    violations.append(f"{dotted} (line {node.lineno}): import {alias.name}")

    assert not violations, (
        "Found module-level `from src.bootstrap import ...` (or `import "
        "src.bootstrap`) in module(s) reachable from src/api/__init__.py:\n"
        + "\n".join(violations)
        + "\n\nThis reintroduces the src.bootstrap <-> src.api import cycle: "
        "src.bootstrap top-level-imports src.api.telephony_* at module load "
        "time, which triggers src/api/__init__.py, which imports the modules "
        "walked above -- so any of them importing back from src.bootstrap at "
        "module level makes a cold `import src.bootstrap` hit a "
        "half-initialized module and raise ImportError. Fix by moving the "
        "imported name to a dependency-free module (see src/exceptions.py, "
        "src/defaults.py) or making the import function-local at its use "
        "site -- see this test's docstring."
    )
