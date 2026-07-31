# Repository governance

The tracked repository contains source code, reviewed operational entrypoints,
tests, documentation, and redacted audit evidence. Heavy or generated state is
kept under the ignored `workspace/` tree.

Background automation must not stage or commit repository content. Operational
scripts must never use broad staging, bypass Git hooks, recursively remove
artifact trees, or silently prune checkpoints. Every delivery commit uses a
Conventional Commit message, explicitly staged paths, and focused verification.

Corpus binaries, checkpoints, raw datasets, secrets, virtual environments,
runtime logs, generated audit reports, and local audit tools are excluded from
Git. Artifact cleanup is a separate evidence-gated operation; it is not part of
the single-root migration.

The scripts in `governance/archive/legacy-scratch/` are retained as historical evidence.
They are non-operational and must not be executed.
