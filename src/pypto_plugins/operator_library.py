"""Fail-closed identity and public-ABI guard for ``pypto-kernels``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import unquote, urlparse

from .errors import FrameworkCompatibilityError


EXPECTED_OPERATOR_LIBRARY_VERSION = "0.1.0.dev0"
EXPECTED_OPERATOR_ABI_SCHEMA_VERSION = 1
EXPECTED_OPERATOR_ABI_DIGEST = (
    "5a72b08f914c352885111b1a929bcb92c561b98980dede3c344227b7719a2ac4"
)
EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST = (
    "0d7057c62e5df2df9adc8b565e8672f6c95dec0d06db7cf4a1ab2a23f6a48400"
)
_EXPECTED_DISTRIBUTION_NAME = "pypto-kernels"
_MANIFEST_KEYS = frozenset({"abi", "abi_digest", "package_version", "schema_version"})
_NATIVE_MODULE_SUFFIXES = tuple(
    sorted(
        set(importlib.machinery.EXTENSION_SUFFIXES) | {".so", ".pyd", ".dylib"},
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True, slots=True)
class OperatorLibrarySnapshot:
    """The identities actually observed by one compatibility check."""

    version: str
    abi_schema_version: int
    abi_digest: str
    package_tree_digest: str | None
    distribution_name: str | None
    resolved_origin: str | None


def _canonical_digest(value: object) -> str:
    """Recompute the producer's documented canonical JSON digest locally."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FrameworkCompatibilityError(
            "pypto-kernels public ABI manifest is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _symbol(value: object) -> list[str]:
    return [
        str(getattr(value, "__module__", type(value).__module__)),
        str(getattr(value, "__qualname__", type(value).__qualname__)),
    ]


def _operator_spec_contract(public_name: str, spec: object) -> dict[str, Any]:
    return {
        "public_name": public_name,
        "object_type": _symbol(type(spec)),
        "name": spec.name,
        "abi_version": spec.abi_version,
        "parameters": [
            [parameter.name, list(parameter.allowed_dtypes)]
            for parameter in spec.parameters
        ],
        "result_names": list(spec.result_names),
        "mutations": [
            [mutation.argument, mutation.access.value, list(mutation.may_alias)]
            for mutation in spec.mutations
        ],
    }


def _binding_contract(
    *,
    public_name: str,
    value: object,
    declared: object,
    operator_contracts: object,
) -> None:
    if type(declared) is not dict or type(declared.get("kind")) is not str:
        raise FrameworkCompatibilityError(
            f"pypto-kernels manifest binding {public_name!r} is malformed"
        )
    kind = declared["kind"]
    try:
        if kind == "symbol":
            observed: object = _symbol(value)
            expected = declared.get("symbol")
        elif kind == "constant":
            observed = json.loads(
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            expected = declared.get("value")
        elif kind == "catalog":
            observed = {
                "object_type": _symbol(type(value)),
                "families": [
                    [
                        family.name,
                        family.abi_version,
                        _symbol(family.config_type),
                    ]
                    for family in value.families
                ],
            }
            expected = declared.get("contract")
        elif kind == "operator-spec":
            if type(operator_contracts) is not dict:
                raise FrameworkCompatibilityError(
                    "pypto-kernels manifest operator contracts are malformed"
                )
            operator_name = declared.get("name")
            operator = operator_contracts.get(operator_name)
            if type(operator) is not dict or type(operator.get("spec")) is not dict:
                raise FrameworkCompatibilityError(
                    f"pypto-kernels manifest lacks operator spec {operator_name!r}"
                )
            observed = _operator_spec_contract(public_name, value)
            expected = operator["spec"]
        else:
            raise FrameworkCompatibilityError(
                f"pypto-kernels manifest binding {public_name!r} has unknown kind {kind!r}"
            )
    except FrameworkCompatibilityError:
        raise
    except Exception as error:
        raise FrameworkCompatibilityError(
            f"unable to inspect pypto-kernels binding {public_name!r}"
        ) from error
    if observed != expected:
        raise FrameworkCompatibilityError(
            f"pypto-kernels runtime binding {public_name!r} disagrees with its manifest"
        )


def inspect_operator_library(module: object) -> OperatorLibrarySnapshot:
    """Validate producer-owned ABI metadata without importing a framework."""

    version = getattr(module, "__version__", None)
    if version != EXPECTED_OPERATOR_LIBRARY_VERSION:
        raise FrameworkCompatibilityError(
            "pypto-kernels version mismatch: "
            f"expected {EXPECTED_OPERATOR_LIBRARY_VERSION!r}, got {version!r}"
        )
    schema_version = getattr(module, "PYPTO_KERNELS_ABI_SCHEMA_VERSION", None)
    if (
        type(schema_version) is not int
        or schema_version != EXPECTED_OPERATOR_ABI_SCHEMA_VERSION
    ):
        raise FrameworkCompatibilityError(
            "pypto-kernels ABI schema mismatch: "
            f"expected {EXPECTED_OPERATOR_ABI_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    exported_digest = getattr(module, "PYPTO_KERNELS_ABI_DIGEST", None)
    if exported_digest != EXPECTED_OPERATOR_ABI_DIGEST:
        raise FrameworkCompatibilityError(
            "pypto-kernels exported ABI digest mismatch: "
            f"expected {EXPECTED_OPERATOR_ABI_DIGEST!r}, got {exported_digest!r}"
        )
    manifest_factory = getattr(module, "public_abi_manifest", None)
    if not callable(manifest_factory):
        raise FrameworkCompatibilityError(
            "pypto-kernels lacks callable public_abi_manifest"
        )
    try:
        manifest = manifest_factory()
    except Exception as error:
        raise FrameworkCompatibilityError(
            "pypto-kernels public_abi_manifest failed"
        ) from error
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        observed: Any
        observed = sorted(manifest) if type(manifest) is dict else type(manifest).__name__
        raise FrameworkCompatibilityError(
            "pypto-kernels public ABI manifest envelope mismatch: "
            f"expected {sorted(_MANIFEST_KEYS)!r}, got {observed!r}"
        )
    if manifest["schema_version"] != schema_version:
        raise FrameworkCompatibilityError(
            "pypto-kernels manifest and exported ABI schema disagree"
        )
    if manifest["package_version"] != version:
        raise FrameworkCompatibilityError(
            "pypto-kernels manifest and exported package version disagree"
        )
    if manifest["abi_digest"] != exported_digest:
        raise FrameworkCompatibilityError(
            "pypto-kernels manifest and exported ABI digest disagree"
        )
    abi = manifest["abi"]
    if type(abi) is not dict or abi.get("scope") != "framework-adapter":
        raise FrameworkCompatibilityError(
            "pypto-kernels manifest lacks the framework-adapter ABI scope"
        )
    recomputed_digest = _canonical_digest(
        {"abi": abi, "schema_version": schema_version}
    )
    if recomputed_digest != exported_digest:
        raise FrameworkCompatibilityError(
            "pypto-kernels public ABI payload does not match its digest: "
            f"recomputed {recomputed_digest!r}, exported {exported_digest!r}"
        )
    bindings = abi.get("top_level_bindings")
    if type(bindings) is not dict or not bindings:
        raise FrameworkCompatibilityError(
            "pypto-kernels manifest lacks required top-level bindings"
        )
    missing_bindings = []
    for name in bindings:
        if type(name) is not str or not name or not hasattr(module, name):
            missing_bindings.append(repr(name))
    if missing_bindings:
        raise FrameworkCompatibilityError(
            "pypto-kernels lacks manifest-declared top-level bindings: "
            f"{tuple(sorted(missing_bindings))!r}"
        )
    for name, declared in bindings.items():
        _binding_contract(
            public_name=name,
            value=getattr(module, name),
            declared=declared,
            operator_contracts=abi.get("operators"),
        )
    return OperatorLibrarySnapshot(
        version=version,
        abi_schema_version=schema_version,
        abi_digest=exported_digest,
        package_tree_digest=None,
        distribution_name=None,
        resolved_origin=None,
    )


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise FrameworkCompatibilityError(
                f"pypto_kernels source path contains symbolic link {current}"
            )


def _is_unexpected_import_payload(path: PurePosixPath) -> bool:
    if path.suffix in (".pyc", ".pyo"):
        return "__pycache__" not in path.parts
    return any(path.name.endswith(suffix) for suffix in _NATIVE_MODULE_SUFFIXES)


def _package_source_records(package_directory: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Return an unambiguous record for every production Python source."""

    try:
        _reject_symlink_components(package_directory.absolute())
        root = package_directory.resolve(strict=True)
        if not root.is_dir():
            raise FrameworkCompatibilityError(
                "pypto_kernels package path is not a directory"
            )
        records: list[dict[str, Any]] = []
        relative_paths: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise FrameworkCompatibilityError(
                    f"pypto_kernels package tree contains symbolic link {path}"
                )
            relative_entry = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_file() and _is_unexpected_import_payload(relative_entry):
                raise FrameworkCompatibilityError(
                    "pypto_kernels pure-Python package contains untracked "
                    f"import payload {relative_entry.as_posix()!r}"
                )
            if path.suffix != ".py":
                continue
            if not path.is_file():
                raise FrameworkCompatibilityError(
                    f"pypto_kernels Python source is not a regular file: {path}"
                )
            resolved = path.resolve(strict=True)
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as error:
                raise FrameworkCompatibilityError(
                    f"pypto_kernels Python source escapes its package: {path}"
                ) from error
            pure_relative = PurePosixPath(relative)
            if (
                pure_relative.is_absolute()
                or ".." in pure_relative.parts
                or relative in relative_paths
            ):
                raise FrameworkCompatibilityError(
                    f"invalid pypto_kernels source-tree path {relative!r}"
                )
            relative_paths.add(relative)
            content = resolved.read_bytes()
            records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            )
        if not records:
            raise FrameworkCompatibilityError(
                "pypto_kernels package contains no Python sources"
            )
        return root, sorted(records, key=lambda record: record["path"])
    except FrameworkCompatibilityError:
        raise
    except OSError as error:
        raise FrameworkCompatibilityError(
            "unable to read the pypto_kernels package source tree"
        ) from error


def _package_tree_digest(package_directory: Path) -> str:
    """Hash sorted source records while rejecting links and path escapes."""

    _root, records = _package_source_records(package_directory)
    return _canonical_digest(records)


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _editable_direct_url(
    distribution: importlib.metadata.Distribution,
) -> dict[str, Any] | None:
    reader = getattr(distribution, "read_text", None)
    if not callable(reader):
        raise FrameworkCompatibilityError(
            "installed pypto-kernels distribution metadata is unreadable"
        )
    direct_url_text = reader("direct_url.json")
    if direct_url_text is None:
        return None
    direct_url = _strict_json_object(
        direct_url_text,
        label="pypto-kernels direct_url.json",
    )
    dir_info = direct_url.get("dir_info")
    if dir_info is not None and type(dir_info) is not dict:
        raise FrameworkCompatibilityError(
            "pypto-kernels direct_url.json dir_info is malformed"
        )
    if type(dir_info) is not dict or "editable" not in dir_info:
        return None
    if type(dir_info["editable"]) is not bool:
        raise FrameworkCompatibilityError(
            "pypto-kernels direct_url.json editable flag must be boolean"
        )
    return direct_url if dir_info["editable"] is True else None


def _unique_distribution() -> tuple[importlib.metadata.Distribution, int]:
    try:
        matches = [
            distribution
            for distribution in importlib.metadata.distributions()
            if _normalized_distribution_name(distribution.metadata.get("Name", ""))
            == _EXPECTED_DISTRIBUTION_NAME
        ]
    except Exception as error:
        raise FrameworkCompatibilityError(
            "unable to enumerate installed pypto-kernels distributions"
        ) from error
    if len(matches) == 1:
        return matches[0], 1
    if len(matches) == 2:
        editable_candidates = [
            (distribution, direct_url)
            for distribution in matches
            if (direct_url := _editable_direct_url(distribution)) is not None
        ]
        if len(editable_candidates) == 1:
            primary, direct_url = editable_candidates[0]
            auxiliary = next(
                distribution for distribution in matches if distribution is not primary
            )
            editable_root = _local_editable_root(direct_url)
            auxiliary_path = getattr(auxiliary, "_path", None)
            try:
                resolved_auxiliary_path = Path(auxiliary_path).resolve(strict=True)
                resolved_auxiliary_base = Path(auxiliary.locate_file("")).resolve(
                    strict=True
                )
            except (AttributeError, OSError, TypeError) as error:
                raise FrameworkCompatibilityError(
                    "pypto-kernels editable source metadata cannot be resolved"
                ) from error
            expected_metadata = editable_root / "src" / "pypto_kernels.egg-info"
            if (
                getattr(auxiliary, "version", None) == primary.version
                and callable(getattr(auxiliary, "read_text", None))
                and auxiliary.read_text("direct_url.json") is None
                and resolved_auxiliary_path == expected_metadata
                and resolved_auxiliary_base == editable_root / "src"
            ):
                return primary, 2
    if len(matches) != 1:
        raise FrameworkCompatibilityError(
            "exactly one installed pypto-kernels distribution is required; "
            f"found {len(matches)}"
        )
    raise AssertionError("unreachable")


def _strict_json_object(text: str, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FrameworkCompatibilityError(f"{label} is malformed") from error
    if type(value) is not dict:
        raise FrameworkCompatibilityError(f"{label} must contain a JSON object")
    return value


def _local_editable_root(direct_url: dict[str, Any]) -> Path:
    try:
        parsed = urlparse(direct_url["url"])
    except (KeyError, TypeError, ValueError) as error:
        raise FrameworkCompatibilityError(
            "pypto-kernels editable direct_url.json is malformed"
        ) from error
    if (
        parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
        or parsed.query
        or parsed.fragment
    ):
        raise FrameworkCompatibilityError(
            "pypto-kernels editable install must use a local file URL"
        )
    editable_path = Path(unquote(parsed.path))
    if not editable_path.is_absolute():
        raise FrameworkCompatibilityError(
            "pypto-kernels editable source root must be absolute"
        )
    _reject_symlink_components(editable_path)
    try:
        editable_root = editable_path.resolve(strict=True)
    except OSError as error:
        raise FrameworkCompatibilityError(
            "pypto-kernels editable source root cannot be resolved"
        ) from error
    return editable_root


def _bind_editable_source(
    direct_url: dict[str, Any],
    *,
    package_root: Path,
    module_path: Path,
) -> str:
    editable_root = _local_editable_root(direct_url)
    expected_module = editable_root / "src" / "pypto_kernels" / "__init__.py"
    if module_path != expected_module or package_root != expected_module.parent:
        raise FrameworkCompatibilityError(
            "pypto-kernels import origin is outside its editable source root"
        )
    return "editable"


def _distribution_source_mode(
    distribution: importlib.metadata.Distribution,
    *,
    package_root: Path,
    module_path: Path,
) -> str:
    """Bind an imported package to either wheel RECORD or PEP-660 metadata."""

    _root, records = _package_source_records(package_root)
    actual_relative_paths = {record["path"] for record in records}
    direct_url = _editable_direct_url(distribution)
    if direct_url is not None:
        return _bind_editable_source(
            direct_url,
            package_root=package_root,
            module_path=module_path,
        )

    wheel_paths: dict[str, Path] = {}
    for entry in distribution.files or ():
        pure = PurePosixPath(str(entry))
        if not pure.parts or pure.parts[0] != "pypto_kernels":
            continue
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) < 2:
            raise FrameworkCompatibilityError(
                f"invalid pypto-kernels distribution source path {str(entry)!r}"
            )
        relative = PurePosixPath(*pure.parts[1:]).as_posix()
        if _is_unexpected_import_payload(PurePosixPath(relative)):
            raise FrameworkCompatibilityError(
                "pypto-kernels wheel RECORD contains untracked import payload "
                f"{relative!r}"
            )
        if pure.suffix != ".py":
            continue
        if relative in wheel_paths:
            raise FrameworkCompatibilityError(
                f"duplicate pypto-kernels distribution source path {relative!r}"
            )
        located = Path(distribution.locate_file(entry))
        _reject_symlink_components(located.absolute())
        try:
            resolved = located.resolve(strict=True)
            resolved.relative_to(package_root)
        except (OSError, ValueError) as error:
            raise FrameworkCompatibilityError(
                f"pypto-kernels distribution source escapes package root: {located}"
            ) from error
        wheel_paths[relative] = resolved
    if wheel_paths:
        if set(wheel_paths) != actual_relative_paths:
            raise FrameworkCompatibilityError(
                "pypto-kernels wheel RECORD source set does not match imported package"
            )
        if wheel_paths.get("__init__.py") != module_path:
            raise FrameworkCompatibilityError(
                "pypto-kernels wheel RECORD does not own the imported module"
            )
        return "wheel"

    raise FrameworkCompatibilityError(
        "pypto-kernels distribution owns neither wheel sources nor an approved "
        "editable source"
    )


def assert_operator_library_compatible() -> OperatorLibrarySnapshot:
    """Validate installed ownership, import identity, source tree and public ABI."""

    try:
        import pypto_kernels
    except Exception as error:
        raise FrameworkCompatibilityError(
            "the standalone pypto-kernels package is unavailable"
        ) from error
    distribution, allowed_provider_count = _unique_distribution()
    if distribution.version != EXPECTED_OPERATOR_LIBRARY_VERSION:
        raise FrameworkCompatibilityError(
            "installed pypto-kernels distribution version mismatch: "
            f"expected {EXPECTED_OPERATOR_LIBRARY_VERSION!r}, "
            f"got {distribution.version!r}"
        )
    providers = importlib.metadata.packages_distributions().get("pypto_kernels", [])
    normalized_providers = [
        _normalized_distribution_name(name) for name in providers if type(name) is str
    ]
    if normalized_providers != [
        _EXPECTED_DISTRIBUTION_NAME
    ] * allowed_provider_count:
        raise FrameworkCompatibilityError(
            "pypto_kernels must have exactly one pypto-kernels distribution owner"
        )
    module_file = getattr(pypto_kernels, "__file__", None)
    try:
        spec = importlib.util.find_spec("pypto_kernels")
    except (ImportError, ValueError) as error:
        raise FrameworkCompatibilityError(
            "unable to resolve the pypto_kernels import specification"
        ) from error
    if type(module_file) is not str or spec is None or spec.origin is None:
        raise FrameworkCompatibilityError(
            "pypto_kernels has no concrete source import origin"
        )
    module_path = Path(module_file)
    spec_path = Path(spec.origin)
    _reject_symlink_components(module_path.absolute())
    _reject_symlink_components(spec_path.absolute())
    try:
        resolved_module_path = module_path.resolve(strict=True)
        resolved_spec_path = spec_path.resolve(strict=True)
    except OSError as error:
        raise FrameworkCompatibilityError(
            "pypto_kernels import origin cannot be resolved"
        ) from error
    if (
        resolved_module_path != resolved_spec_path
        or resolved_module_path.name != "__init__.py"
        or not resolved_module_path.is_file()
    ):
        raise FrameworkCompatibilityError(
            "pypto_kernels import origin does not match its source package specification"
        )
    package_root = resolved_module_path.parent
    _distribution_source_mode(
        distribution,
        package_root=package_root,
        module_path=resolved_module_path,
    )
    package_tree_digest = _package_tree_digest(package_root)
    if package_tree_digest != EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST:
        raise FrameworkCompatibilityError(
            "pypto-kernels source tree digest mismatch: "
            f"expected {EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST!r}, "
            f"got {package_tree_digest!r}"
        )
    abi_snapshot = inspect_operator_library(pypto_kernels)
    distribution_name = distribution.metadata.get("Name")
    return OperatorLibrarySnapshot(
        version=abi_snapshot.version,
        abi_schema_version=abi_snapshot.abi_schema_version,
        abi_digest=abi_snapshot.abi_digest,
        package_tree_digest=package_tree_digest,
        distribution_name=str(distribution_name),
        resolved_origin=str(resolved_module_path),
    )


__all__ = (
    "EXPECTED_OPERATOR_ABI_DIGEST",
    "EXPECTED_OPERATOR_ABI_SCHEMA_VERSION",
    "EXPECTED_OPERATOR_LIBRARY_VERSION",
    "EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST",
    "OperatorLibrarySnapshot",
    "assert_operator_library_compatible",
    "inspect_operator_library",
)
