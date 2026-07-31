#!/usr/bin/env python3
"""Static internal-gold architecture boundary gate; imports no product code."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = "governance/audit/policy/operational-surface.json"


def _finding(kind: str, path: str, **details: object) -> dict[str, object]:
    return {"type": kind, "path": path, **details}


def _python_files(root: Path, roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in roots:
        base = root / item
        if base.is_dir():
            files.extend(path for path in base.rglob("*.py") if path.is_file())
    return sorted(set(files))


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _module_name(root: Path, package_root: str, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or package_root


def _imports(tree: ast.AST, module: str, *, is_package: bool) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = module if is_package else module.rpartition(".")[0]
                prefix = package.split(".") if package else []
                ascend = node.level - 1
                if ascend:
                    prefix = prefix[:-ascend] if ascend <= len(prefix) else []
                if node.module:
                    prefix.extend(node.module.split("."))
                imported = ".".join(prefix)
            else:
                imported = node.module or ""
            if imported:
                result.add(imported)
                result.update(
                    f"{imported}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            elif node.names:
                result.update(alias.name for alias in node.names)
    return result


def _dynamic_imports(tree: ast.AST) -> set[str]:
    """Resolve literal dynamic imports so package boundaries cannot be bypassed."""
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_import = isinstance(function, ast.Name) and function.id == "__import__"
        is_importlib = (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id == "importlib"
        )
        module = node.args[0]
        if (is_import or is_importlib) and isinstance(module, ast.Constant) and isinstance(module.value, str):
            result.add(module.value)
    return result


def _all_imports(tree: ast.AST, module: str, *, is_package: bool) -> set[str]:
    return _imports(tree, module, is_package=is_package) | _dynamic_imports(tree)


def _cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for target in sorted(graph.get(node, set())):
            if target not in graph:
                continue
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in graph.get(node, set()):
                components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components)


def _is_sys_path(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Subscript):
        return _is_sys_path(node.value)
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _has_sys_path_mutation(node: ast.AST) -> bool:
    target = node.func if isinstance(node, ast.Call) else getattr(node, "target", None)
    if isinstance(target, ast.Attribute) and target.attr in {"append", "extend", "insert"}:
        value = target.value
    else:
        value = target
    return _is_sys_path(value)


def _assignment_mutates_sys_path(node: ast.AST) -> bool:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = list(node.targets)
    return any(_is_sys_path(target) for target in targets)


def _yaml_scalar(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return None


def _file_size_findings(
    root: Path,
    files: list[Path],
    budgets: dict[str, object],
    supported: set[str],
    exceptions: set[str],
) -> list[dict[str, object]]:
    """Enforce file budgets across every recursively discovered active module."""
    findings: list[dict[str, object]] = []
    general_limit = int(budgets.get("supported_file_lines", 1200))
    adapter_limit = int(budgets.get("entrypoint_adapter_lines", 250))
    for path in files:
        relative = path.relative_to(root).as_posix()
        limit = min(general_limit, adapter_limit) if relative in supported else general_limit
        lines = len(_source(path).splitlines())
        if lines > limit and relative not in exceptions:
            findings.append(_finding("file_size_budget", relative, lines=lines, limit=limit))
    return findings


def _checkpoint_findings(
    root: Path,
    boundary: dict[str, object],
    parsed: dict[Path, ast.AST],
) -> list[dict[str, object]]:
    checkpoint = boundary.get("checkpoint_authority", {})
    if not isinstance(checkpoint, dict):
        return []
    findings: list[dict[str, object]] = []
    authority = str(checkpoint.get("path", ""))
    delegated_strict_loaders = {
        str(item) for item in checkpoint.get("delegated_strict_loaders", [])
    }
    if not (root / authority).is_file():
        findings.append(_finding("checkpoint_authority_missing", authority))
    strict_roots = [
        str(item) for item in checkpoint.get("strict_false_forbidden_roots", [])
    ]
    for path in _python_files(root, strict_roots):
        tree = parsed.get(path)
        if tree is None:
            try:
                tree = ast.parse(_source(path))
            except SyntaxError:
                continue
        has_load = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "load_state_dict":
                continue
            has_load = True
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value is False:
                findings.append(_finding("permissive_checkpoint_load", path.relative_to(root).as_posix(), line=node.lineno))
            for keyword in node.keywords:
                if keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    findings.append(_finding("permissive_checkpoint_load", path.relative_to(root).as_posix(), line=node.lineno))
        relative = path.relative_to(root).as_posix()
        if has_load and "checkpoint" in path.stem and relative != authority and relative not in delegated_strict_loaders:
            findings.append(_finding("checkpoint_authority_duplicate", relative))
    return findings


def evaluate(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    policy_file = root / POLICY_PATH
    if not policy_file.is_file():
        return {"status": "fail", "findings": [_finding("architecture_policy_missing", POLICY_PATH)]}
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    boundary = policy.get("gold_boundaries", {})
    if not isinstance(boundary, dict):
        return {"status": "fail", "findings": [_finding("architecture_policy_invalid", POLICY_PATH)]}
    findings: list[dict[str, object]] = []
    roots = [str(item) for item in boundary.get("active_python_roots", [])]
    package_root = str(boundary.get("package_root", "f51_darwin"))
    files = _python_files(root, roots)
    parsed: dict[Path, ast.AST] = {}
    imports: dict[Path, set[str]] = {}

    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(_source(path), filename=relative)
        except (SyntaxError, UnicodeDecodeError) as exc:
            findings.append(_finding("python_parse_error", relative, detail=str(exc)))
            continue
        parsed[path] = tree
        module = _module_name(root, package_root, path)
        imports[path] = _all_imports(tree, module, is_package=path.name == "__init__.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call) and _has_sys_path_mutation(node)
            ) or _assignment_mutates_sys_path(node):
                findings.append(_finding("sys_path_mutation", relative, line=getattr(node, "lineno", 0)))

    forbidden_roots = set(str(item) for item in boundary.get("forbidden_package_import_roots", []))
    package_base = root / package_root
    for path, names in imports.items():
        if path.is_relative_to(package_base):
            for name in sorted(names):
                top = name.split(".", 1)[0]
                if top in forbidden_roots:
                    findings.append(_finding("forbidden_dependency", path.relative_to(root).as_posix(), import_name=name))

    modules = {
        _module_name(root, package_root, path): path
        for path in parsed
        if path.is_relative_to(package_base)
    }
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        for name in imports.get(path, set()):
            matches = [candidate for candidate in modules if name == candidate or name.startswith(candidate + ".")]
            if matches:
                graph[module].add(max(matches, key=len))
    for cycle in _cycles(graph):
        findings.append(_finding("package_cycle", package_root, modules=list(cycle)))

    scripts = root / "src" / "scripts"
    actual_script_files = {
        path.relative_to(root).as_posix()
        for path in scripts.iterdir()
        if path.is_file()
    } if scripts.is_dir() else set()
    allowed_scripts = set(str(item) for item in boundary.get("script_allowed_files", []))
    if actual_script_files != allowed_scripts:
        findings.append(
            _finding(
                "script_surface_mismatch",
                "scripts",
                missing=sorted(allowed_scripts - actual_script_files),
                extra=sorted(actual_script_files - allowed_scripts),
            )
        )

    budgets = boundary.get("size_budgets", {})
    exceptions = set(str(item) for item in boundary.get("exceptions", []))
    supported = set(str(item) for item in policy.get("supported_entrypoints", []))
    findings.extend(_file_size_findings(root, files, budgets, supported, exceptions))

    for path in files:
        relative = path.relative_to(root).as_posix()
        tree = parsed.get(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = int(getattr(node, "end_lineno", node.lineno)) - node.lineno + 1
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            limit = int(budgets.get(f"{kind}_lines", 500 if kind == "class" else 180))
            identity = f"{relative}:{node.name}"
            if size > limit and identity not in exceptions:
                findings.append(_finding(f"{kind}_size_budget", relative, symbol=node.name, lines=size, limit=limit))

    config = boundary.get("canonical_config", {})
    if isinstance(config, dict):
        config_path = str(config.get("path", ""))
        expected_model = str(config.get("model_name", ""))
        if _yaml_scalar(root / config_path, "model_name") != expected_model:
            findings.append(_finding("canonical_config_mismatch", config_path))
        forbidden_labels = set(str(item) for item in config.get("forbidden_default_labels", []))
        for path, tree in parsed.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or node.name != "DarwinXConfig":
                    continue
                for statement in node.body:
                    target = getattr(statement, "target", None)
                    value = getattr(statement, "value", None)
                    if isinstance(target, ast.Name) and target.id == "model_name" and isinstance(value, ast.Constant):
                        if value.value in forbidden_labels:
                            findings.append(_finding("legacy_default_model", path.relative_to(root).as_posix(), value=value.value))
        for relative, expected in sorted(config.get("entrypoint_config_defaults", {}).items()):
            path = root / str(relative)
            if not path.is_file():
                continue
            try:
                tree = ast.parse(_source(path))
            except SyntaxError:
                continue
            expected_name = Path(str(expected)).name
            matched = False
            observed: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "add_argument":
                    continue
                option_names = {
                    argument.value
                    for argument in node.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                }
                if "--config" not in option_names:
                    continue
                default = next((item.value for item in node.keywords if item.arg == "default"), None)
                literals = {
                    child.value
                    for child in ast.walk(default) if isinstance(child, ast.Constant) and isinstance(child.value, str)
                } if default is not None else set()
                observed.update(literals)
                matched = expected_name in literals or str(expected) in literals
            if not matched:
                findings.append(
                    _finding(
                        "entrypoint_config_default_mismatch",
                        str(relative),
                        expected=str(expected),
                        observed=sorted(observed),
                    )
                )

    findings.extend(_checkpoint_findings(root, boundary, parsed))

    findings.sort(key=lambda item: (str(item.get("type")), str(item.get("path")), int(item.get("line", 0))))
    return {"status": "pass" if not findings else "fail", "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = evaluate(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
