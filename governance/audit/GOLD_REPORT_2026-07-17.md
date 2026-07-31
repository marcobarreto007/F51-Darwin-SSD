# External audit simulation - 2026-07-17

> Historical evidence snapshot. Commands below were reconciled to the current
> `src/tools/` layout, but commit-specific counts and verdict refer to the audited
> source named below, not to the current HEAD. Use the latest report under
> `workspace/runtime/evaluations/` for a current verdict.

## Verdict

**NOT GOLD - external security blocker.**

The local single-root system and current tracked source are internally
consistent, but reachable Git history contains nine redacted secret findings.
Credential revocation is not proven and history has not been rewritten. This
is one P1 security blocker and forbids a GOLD claim, readiness launch and GPU
canary. No P0 was found.

Audited source: `f54a579802e8e56b819eba37ce81b69c444fd781`

Source tree: `9ece38a0b833c65f5f05255017400778bea8c1f1`

Machine-readable manifest: `governance/audit/provenance/source-manifest.json`

Schema-valid tracked source-audit report:
`governance/audit/test-reports/gold-source-f54a579.json`. The latest evidence commit is
resolved with `git log -1 --format=%H -- governance/audit/provenance/source-manifest.json`;
the source and evidence commits are deliberately separate.

Evidence hashes use the `git_blob_bytes_v1` profile. Verify them from any
checkout with `git cat-file blob <evidence-commit>:<path>`; do not hash archive
working-tree bytes without first accounting for Git line-ending conversion.

## Requirement ledger

| Requirement | Result | Direct evidence |
|---|---:|---|
| One physical root | PASS | `workspace/` is a physical directory; sibling, `data/` and root `checkpoints/` paths are absent; `governance/audit/provenance/single-root-migration-summary.json` |
| Reversible migration | PASS | Same-volume rename manifest, hashes and fixture rollback; command in `governance/docs/operacao/OPERACAO_SEGURA.md` |
| No unjustified duplicates | PASS | `src/tools/check_duplicates.py`; one hash-bound AGENTS/CLAUDE exception; checker exit 0 in worktree and source archive |
| Canonical documents current | PASS | links, canonical and duplicate checkers exit 0; independent Task 10 review found zero P0/P1 |
| Six supported entrypoints | PASS | `governance/audit/policy/operational-surface.json`; surface checker exit 0 |
| Corpus identity | PASS | feast_v2: 74,195,890,756 bytes, 18,548,972,689 tokens, SHA-256 `9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff` |
| Checkpoint lineage | PASS | cycle 71, step 40751, strict resume and identity valid; SHA-256 `239fdcf175ac35d9402d664e2a2b40252ec9c73aea8ee458250d426adde9241b`; cycle 77 rejected |
| Pointer bound to source | PASS | `organism_latest.json` and `gold_local_lineage.json` both bind to audited source |
| CPU suite and compilation | PASS | source audit pytest exit 0; 384 tests collected; 250/250 active tracked Python files compiled |
| Clean-checkout portability | PASS | independent `git archive` verification; LF/CRLF duplicate mirror test; exact Git-blob supply-chain index |
| CPU dependency supply chain | PASS | hash-locked requirements, CycloneDX SBOM, vulnerability/license reports and dependency checker exit 0 |
| Nitro runtime supply chain | PASS | pinned CPython/Torch/CUDA provenance and separate Nitro SBOM; non-hardware checker exit 0 |
| Current-tree secret scan | PASS | Gitleaks 8.30.1 scanned 467 tracked files, zero findings |
| Reachable-history secret scan | **FAIL** | nine redacted findings over all local refs; no revocation proof; no rewrite |
| Source audit | **HISTORICAL** | `governance/audit/test-reports/gold-source-f54a579.json` preserva o relatorio antigo de 12 gates; o harness atual possui 15 gates e deve ser reexecutado contra o HEAD limpo |
| Readiness/canary | GATED | source gate is red; official canary gate and launch were not executed; no process created |
| Legal/security controls | PASS | proprietary `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `SECURITY.md` |

## Reproduction

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m tools.verify_gold_lineage --project-root .
.\.venv_nitro\Scripts\python.exe -m tools.check_docs_links --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_canonical_docs --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_duplicates --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_dependency_policy --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_operational_surface --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_physical_hygiene --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_architecture_boundaries --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_distribution --root .
powershell -ExecutionPolicy Bypass -File src\\tools\\run_gold_source_audit.ps1
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

The source audit is expected to return 2 until the external security blocker is
resolved. A zero result obtained by allowlisting real credentials would be an
invalid audit.

## Exact remediation required for GOLD

1. Identify the owners of every credential represented by the redacted history
   findings and revoke/rotate them outside this repository.
2. Preserve auditable proof of revocation without recording secret values.
3. Separately authorize and execute a coordinated rewrite of all affected Git
   refs; update every clone and remote reference.
4. Rescan the complete reachable refset with pinned Gitleaks and require zero
   findings.
5. Re-run all 15 source gates, regenerate readiness for the same clean source
   commit, then run the bounded isolated canary.

Those actions require external credential authority and coordinated history
replacement. They were deliberately not simulated, bypassed or claimed here.
