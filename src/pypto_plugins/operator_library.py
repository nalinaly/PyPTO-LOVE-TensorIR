"""Structural compatibility guard for the canonical PyPTO operator package."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
from types import ModuleType
from types import SimpleNamespace

from .errors import FrameworkCompatibilityError


EXPECTED_OPERATOR_LIBRARY_VERSION = "0.1.0"
EXPECTED_OPERATOR_MODULES = (
    "attention",
    "causal_conv1d",
    "embedding",
    "fused_add_rmsnorm",
    "gdn",
    "gated_rmsnorm",
    "linear",
    "rmsnorm",
    "rope",
    "sigmoid_mul",
    "silu_and_mul",
)

_REQUIRED_TILE_FORMS = (
    "@pl.jit",
    "with pl.at(",
    "pl.range(",
    "pl.load(",
    "pl.store(",
)
_VERSION_LABEL = re.compile(r"(?<![A-Za-z0-9_])v[12](?![A-Za-z0-9_])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class OperatorLibrarySnapshot:
    """The canonical operator identities observed by one compatibility check."""

    version: str
    package_root: str
    modules: tuple[str, ...]
    graph_counts: tuple[tuple[str, int], ...]


def _operator_source(name: str, package_root: Path) -> Path:
    candidate = package_root / f"{name}.py"
    try:
        source = candidate.resolve(strict=True)
        source.relative_to(package_root)
    except (OSError, ValueError) as error:
        raise FrameworkCompatibilityError(
            f"operator {name!r} is outside the canonical pypto-kernels package"
        ) from error
    if source.suffix != ".py" or not source.is_file():
        raise FrameworkCompatibilityError(
            f"operator {name!r} is not a Python operator source"
        )
    return source


def _integer_declaration(tree: ast.Module, name: str, default: int | None) -> int | None:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = statement.value
        if isinstance(value, ast.Constant) and type(value.value) is int:
            return value.value
        return None
    return default


def _graph_count(module: object, source: Path) -> int:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as error:
        raise FrameworkCompatibilityError(
            f"unable to parse operator source {source}"
        ) from error
    primary = _integer_declaration(tree, "GRAPHS", None)
    if type(primary) is not int or primary != 1:
        raise FrameworkCompatibilityError(
            f"{module.__name__} must expose exactly one primary graph"
        )
    update = _integer_declaration(tree, "UPDATE_GRAPHS", 0)
    if type(update) is not int or update not in (0, 1):
        raise FrameworkCompatibilityError(
            f"{module.__name__} has an invalid update graph count"
        )
    if module.__name__.endswith(".gdn"):
        if update != 1:
            raise FrameworkCompatibilityError(
                "pypto_kernels.gdn must expose one read graph and one update graph"
            )
    elif update:
        raise FrameworkCompatibilityError(
            f"{module.__name__} unexpectedly exposes an update graph"
        )
    return primary + update


def _validate_native_tile_source(module: object, source: Path, graphs: int) -> None:
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise FrameworkCompatibilityError(
            f"unable to read operator source {source}"
        ) from error
    missing = tuple(form for form in _REQUIRED_TILE_FORMS if form not in text)
    if missing:
        raise FrameworkCompatibilityError(
            f"{module.__name__} is not an explicit native tile graph; missing {missing!r}"
        )
    if text.count("@pl.jit") != graphs:
        raise FrameworkCompatibilityError(
            f"{module.__name__} JIT graph count disagrees with its graph declaration"
        )
    if text.count("launch_graph(") != graphs:
        raise FrameworkCompatibilityError(
            f"{module.__name__} must launch once for each model operator graph"
        )
    prohibited_whole_tensor_forms = (
        "from ._graph",
        "from pypto import tensor",
        "import pypto.tensor",
    )
    if "ones-matmul" in text.lower() or any(
        form in text for form in prohibited_whole_tensor_forms
    ):
        raise FrameworkCompatibilityError(
            f"{module.__name__} contains a prohibited whole-tensor lowering"
        )
    if _VERSION_LABEL.search(text) or _VERSION_LABEL.search(source.name):
        raise FrameworkCompatibilityError(
            f"{module.__name__} contains a retired version label"
        )


def inspect_operator_library(package: ModuleType) -> OperatorLibrarySnapshot:
    """Validate the active package and every exported Qwen operator graph."""

    version = getattr(package, "__version__", None)
    if version != EXPECTED_OPERATOR_LIBRARY_VERSION:
        raise FrameworkCompatibilityError(
            "pypto-kernels version mismatch: "
            f"expected {EXPECTED_OPERATOR_LIBRARY_VERSION!r}, got {version!r}"
        )
    exports = getattr(package, "__all__", None)
    observed_exports = tuple(exports) if isinstance(exports, (tuple, list)) else ()
    if observed_exports != EXPECTED_OPERATOR_MODULES:
        raise FrameworkCompatibilityError(
            "pypto-kernels exported operator set does not match the Qwen inventory"
        )
    package_file = getattr(package, "__file__", None)
    if not isinstance(package_file, str):
        raise FrameworkCompatibilityError("pypto_kernels has no concrete package source")
    try:
        package_source = Path(package_file).resolve(strict=True)
    except OSError as error:
        raise FrameworkCompatibilityError(
            "pypto_kernels package source cannot be resolved"
        ) from error
    package_root = package_source.parent
    if package_source.name != "__init__.py" or _VERSION_LABEL.search(str(package_root)):
        raise FrameworkCompatibilityError(
            "pypto_kernels is not loaded from the canonical package directory"
        )

    expected_sources = {
        "__init__.py",
        "_boot.py",
        *(f"{name}.py" for name in EXPECTED_OPERATOR_MODULES),
    }
    observed_sources = {path.name for path in package_root.glob("*.py")}
    if observed_sources != expected_sources:
        raise FrameworkCompatibilityError(
            "pypto-kernels package contains an unexpected operator source set"
        )

    counts: list[tuple[str, int]] = []
    for name in EXPECTED_OPERATOR_MODULES:
        module = SimpleNamespace(__name__=f"{package.__name__}.{name}")
        source = _operator_source(name, package_root)
        graphs = _graph_count(module, source)
        _validate_native_tile_source(module, source, graphs)
        counts.append((name, graphs))
    return OperatorLibrarySnapshot(
        version=version,
        package_root=str(package_root),
        modules=EXPECTED_OPERATOR_MODULES,
        graph_counts=tuple(counts),
    )


def assert_operator_library_compatible() -> OperatorLibrarySnapshot:
    """Import and validate the one canonical PyPTO operator package."""

    try:
        package = importlib.import_module("pypto_kernels")
    except Exception as error:
        raise FrameworkCompatibilityError(
            "the canonical pypto-kernels package is unavailable"
        ) from error
    return inspect_operator_library(package)


__all__ = (
    "EXPECTED_OPERATOR_LIBRARY_VERSION",
    "EXPECTED_OPERATOR_MODULES",
    "OperatorLibrarySnapshot",
    "assert_operator_library_compatible",
    "inspect_operator_library",
)
