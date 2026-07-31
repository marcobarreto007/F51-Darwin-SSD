from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DATASET_ROOT_ENV = "F51_DATASET_ROOT"
WORKSPACE_DIR_NAME = "workspace"

RAW_RELATIVE = Path("00_BRUTOS")
RAW_CLASSICAL_RELATIVE = RAW_RELATIVE / "classical_corpus"
TOKENIZED_RELATIVE = Path("01_TOKENIZADOS")
CORPUS_WORKSPACE_RELATIVE = Path("02_CORPUS")
FEAST_TOKEN_RELATIVE = TOKENIZED_RELATIVE / "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"
FEAST_MANIFEST_RELATIVE = Path(f"{FEAST_TOKEN_RELATIVE}.manifest.json")
CLASSICAL_TOKEN_RELATIVE = TOKENIZED_RELATIVE / "CLASSICOS" / "tokens_classical.bin"
LIVE_TOKEN_RELATIVE = TOKENIZED_RELATIVE / "CLASSICOS" / "tokens_live.bin"
APPROVED_TOKEN_CACHE_RELATIVE = TOKENIZED_RELATIVE / "tokens_approved_cache.bin"
CLASSICAL_TOKEN_MANIFEST_RELATIVE = Path("04_MANIFESTOS") / "tokens_classical.json"
CHECKPOINTS_RELATIVE = Path("03_CHECKPOINTS")
ORGANISM_CHECKPOINTS_RELATIVE = CHECKPOINTS_RELATIVE
DARWIN_X_CHECKPOINTS_RELATIVE = CHECKPOINTS_RELATIVE / "darwin_x"
LATEST_POINTER_RELATIVE = CHECKPOINTS_RELATIVE / "organism_latest.json"
MANIFESTS_RELATIVE = Path("04_MANIFESTOS")
READINESS_MANIFEST_RELATIVE = MANIFESTS_RELATIVE / "overnight_16b_readiness.json"
TOKENIZER_RELATIVE = Path("tokenizer")
RUNTIME_RELATIVE = Path("runtime")


def resolve_dataset_root(
    project_root: str | Path,
    *,
    explicit: str | Path | None = None,
    use_environment: bool = False,
    require: bool = False,
) -> Path:
    """Resolve the approved runtime workspace.

    Explicit overrides take precedence. The process environment is consulted
    only when the caller deliberately opts in, keeping temporary project tests
    isolated from an operator shell's environment.
    """

    project = Path(project_root).expanduser().resolve()
    configured = explicit
    if configured is None and use_environment:
        configured = os.environ.get(DATASET_ROOT_ENV)
    root = (
        Path(configured).expanduser().resolve()
        if configured is not None and str(configured).strip()
        else (project / WORKSPACE_DIR_NAME).resolve()
    )
    if require and not root.is_dir():
        raise FileNotFoundError(
            f"Workspace root not found: {root}. Create {project / WORKSPACE_DIR_NAME} "
            f"or opt into {DATASET_ROOT_ENV} for an explicit override."
        )
    return root


def resolve_dataset_path(
    project_root: str | Path,
    relative_path: str | Path,
    *,
    explicit: str | Path | None = None,
    use_environment: bool = False,
    require: bool = False,
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"workspace_boundary: unsafe relative path {relative_path!s}")

    root = resolve_dataset_root(
        project_root,
        explicit=explicit,
        use_environment=use_environment,
        require=require,
    )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"workspace_boundary: resolved path {path} is outside {root}"
        ) from exc
    if require and not path.exists():
        raise FileNotFoundError(f"Workspace artifact not found: {path}")
    return path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    raw: Path
    tokenized: Path
    corpus: Path
    checkpoints: Path
    manifests: Path
    tokenizer: Path
    runtime: Path
    runs: Path
    logs: Path
    evaluations: Path
    feast_token_bin: Path
    feast_manifest: Path
    latest_pointer: Path
    readiness_manifest: Path

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        *,
        explicit: str | Path | None = None,
        use_environment: bool = False,
        require: bool = False,
    ) -> "WorkspacePaths":
        root = resolve_dataset_root(
            project_root,
            explicit=explicit,
            use_environment=use_environment,
            require=require,
        )
        runtime = root / RUNTIME_RELATIVE
        return cls(
            root=root,
            raw=root / RAW_RELATIVE,
            tokenized=root / TOKENIZED_RELATIVE,
            corpus=root / CORPUS_WORKSPACE_RELATIVE,
            checkpoints=root / CHECKPOINTS_RELATIVE,
            manifests=root / MANIFESTS_RELATIVE,
            tokenizer=root / TOKENIZER_RELATIVE,
            runtime=runtime,
            runs=runtime / "runs",
            logs=runtime / "logs",
            evaluations=runtime / "evaluations",
            feast_token_bin=root / FEAST_TOKEN_RELATIVE,
            feast_manifest=root / FEAST_MANIFEST_RELATIVE,
            latest_pointer=root / LATEST_POINTER_RELATIVE,
            readiness_manifest=root / READINESS_MANIFEST_RELATIVE,
        )


def resolve_feast_token_bin(project_root: str | Path, *, require: bool = False) -> Path:
    return resolve_dataset_path(project_root, FEAST_TOKEN_RELATIVE, require=require)


def resolve_corpus_workspace_root(
    project_root: str | Path, *, require: bool = False
) -> Path:
    return resolve_dataset_path(project_root, CORPUS_WORKSPACE_RELATIVE, require=require)


def resolve_checkpoints_root(project_root: str | Path, *, require: bool = False) -> Path:
    return resolve_dataset_path(project_root, CHECKPOINTS_RELATIVE, require=require)


def resolve_organism_checkpoints_root(
    project_root: str | Path, *, require: bool = False
) -> Path:
    return resolve_dataset_path(project_root, ORGANISM_CHECKPOINTS_RELATIVE, require=require)


def resolve_darwin_x_checkpoints_root(
    project_root: str | Path, *, require: bool = False
) -> Path:
    return resolve_dataset_path(project_root, DARWIN_X_CHECKPOINTS_RELATIVE, require=require)
