# F51 Darwin-X — Dicionário Semântico de Arquivos

Gerado em 2026-07-31 a partir do HEAD observado `590d08d` e do manifesto de trabalho após a separação entre `src/scripts/`, `src/tools/` e `research/`. Inclui uma linha local por caminho, classificação de domínio, tamanho, linhas quando textual e o primeiro sinal textual observável. Não importa módulos nem executa código.

O dicionário não transforma um arquivo em autoridade operacional. `HISTORICO`, `PLANO`, `SPEC`, `PESQUISA` e `FRONTEIRA` são classificações explícitas. Para classes, funções e métodos Python, consultar também [`F51_DARWIN_SYSTEM_SYMBOL_DICTIONARY.md`](F51_DARWIN_SYSTEM_SYMBOL_DICTIONARY.md). Para conteúdo ignorado pelo Git, consultar [`F51_DARWIN_PHYSICAL_INVENTORY.md`](F51_DARWIN_PHYSICAL_INVENTORY.md).

Cobertura: 860 caminhos no manifesto de trabalho. O teste de certificado de proveniência permanece separado quando não pertence ao índice Git.

| Caminho | Classificação | Bytes | Linhas | Primeiro sinal |
|---|---|---:|---:|---|
| `.agent_bus/state.json` | ARQUIVO / RAIZ | 1531 | 1 | { |
| `.gitattributes` | GOVERNANCA / RAIZ | 63 | 1 | checkpoints/organism/*.pt filter=lfs diff=lfs merge=lfs -text |
| `.github/workflows/ci.yml` | CI / AUTOMACAO | 2204 | 1 | name: source-audit |
| `.gitignore` | GOVERNANCA / RAIZ | 1353 | 1 | # === Python === |
| `.gitleaks.toml` | GOVERNANCA / RAIZ | 728 | 1 | title = "F51 Darwin source secret policy" |
| `AGENTS.md` | GOVERNANCA / RAIZ | 4584 | 1 | # F51 Darwin-X - contrato de agentes |
| `governance/archive/agent-bus/CODEX_16B_CANARY_20260715.md` | HISTORICO | 536 | 1 | # CODEX 16B CANARY — 2026-07-15 |
| `governance/archive/agent-bus/CODEX_BF16_GRADIENT_GEOMETRY_FIX_2026-07-14.md` | HISTORICO | 1332 | 1 | # Codex — correção BF16 da geometria mutacional (2026-07-14) |
| `governance/archive/agent-bus/CODEX_CLAUDE_OVERNIGHT_PROMPT_2026-07-14.md` | HISTORICO | 454 | 1 | # Codex — prompt noturno para Claude (2026-07-14) |
| `governance/archive/agent-bus/CODEX_DATA_BOUNDARY_2026-07-13.md` | HISTORICO | 2509 | 1 | # F51 data boundary cutover — 2026-07-13 |
| `governance/archive/agent-bus/CODEX_DOC_TRUTH_2026-07-13.md` | HISTORICO | 1260 | 1 | # CODEX - reconciliacao documental - 2026-07-13 |
| `governance/archive/agent-bus/CODEX_EXPERIMENT_RECOVERY_2026-07-13.md` | HISTORICO | 3448 | 1 | # Recuperacao do experimento C0/C1/F51 — 2026-07-13 |
| `governance/archive/agent-bus/CODEX_GOLD_EXECUTOR_1_2026-07-17.md` | HISTORICO | 1533 | 1 | # Codex GOLD Executor 1 — 2026-07-17 |
| `governance/archive/agent-bus/CODEX_GPU_HEALTH_AUDIT_2026-07-15.md` | HISTORICO | 3754 | 1 | # Auditoria de saude das GPUs — 2026-07-15 EDT |
| `governance/archive/agent-bus/CODEX_GPU_RECOVERY_2026-07-15.md` | HISTORICO | 3520 | 1 | # Recuperacao do crash dual-GPU — 2026-07-15 EDT |
| `governance/archive/agent-bus/CODEX_GRADIENT_MUTATIONAL_V2_2026-07-14.md` | HISTORICO | 1533 | 1 | # Gradient Mutational State v2 — 2026-07-14 |
| `governance/archive/agent-bus/CODEX_OVERNIGHT_16B_2026-07-13.md` | HISTORICO | 2124 | 1 | # Codex - Overnight 1.6B canonico - 2026-07-13 |
| `governance/archive/agent-bus/CODEX_OVERNIGHT_25B_2026-07-13.md` | HISTORICO | 2333 | 1 | # CODEX - ponte de treino noturno 2.5B - 2026-07-13 |
| `governance/archive/agent-bus/CODEX_OVERNIGHT_CRASH_AUDIT_2026-07-15.md` | HISTORICO | 3251 | 1 | # Auditoria do crash noturno — 2026-07-15 EDT |
| `governance/archive/agent-bus/CODEX_REBOOT_BLOCK_2026-07-15.md` | HISTORICO | 1822 | 1 | # Bloqueio de reinicio automatico — 2026-07-15 EDT |
| `governance/archive/cloud-local/configs/darwin_x_1.6b_dae_cloud.yaml` | HISTORICO | 1089 | 1 | model_name: F51-Darwin-X-1.6B-DAE-Cloud |
| `governance/archive/cloud-local/configs/darwin_x_5b.yaml` | HISTORICO | 4125 | 1 | # ============================================================================ |
| `governance/archive/cloud-local/README.md` | HISTORICO | 322 | 1 | # Cloud-era local archive |
| `governance/archive/cloud-local/root/deploy_montreal.sh` | HISTORICO | 2945 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/root/l40s_search.json` | HISTORICO | 4 | 1 | [] |
| `governance/archive/cloud-local/root/requirements_cloud.txt` | HISTORICO | 180 | 1 | # F51 Darwin-SSD — Cloud requirements |
| `governance/archive/cloud-local/root/setup_cloud.sh` | HISTORICO | 2850 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/auto_deploy_gpu.ps1` | HISTORICO | 10044 | 1 | #!/usr/bin/env pwsh |
| `governance/archive/cloud-local/scripts/auto_deploy_gpu.sh` | HISTORICO | 10111 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/auto_deploy.sh` | HISTORICO | 3453 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/auto_train_1h.sh` | HISTORICO | 2388 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/build_remote_token_index.py` | HISTORICO | 1421 | 1 | from __future__ import annotations |
| `governance/archive/cloud-local/scripts/cloud_train_automated.py` | HISTORICO | 3943 | 3 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/cloud_train_darwin_x_fixed.sh` | HISTORICO | 2775 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/deploy_dae_cloud.ps1` | HISTORICO | 9287 | 1 | [CmdletBinding()] |
| `governance/archive/cloud-local/scripts/deploy_math_cloud.py` | HISTORICO | 8275 | 2 | #!/usr/bin/env python |
| `governance/archive/cloud-local/scripts/evaluate_dae_cloud_canary.py` | HISTORICO | 6935 | 2 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/final_push.py` | HISTORICO | 15548 | 10 | #!/usr/bin/env python |
| `governance/archive/cloud-local/scripts/ingest_remote_corpus_pack.py` | HISTORICO | 1503 | 1 | from __future__ import annotations |
| `governance/archive/cloud-local/scripts/launch_1.2b_cloud.ps1` | HISTORICO | 2389 | 1 | # F51 Darwin-X 1.2B — Cloud Launch Script |
| `governance/archive/cloud-local/scripts/launch_600m_cloud.ps1` | HISTORICO | 2242 | 1 | # F51 Darwin-X 600M — Cloud Launch Script |
| `governance/archive/cloud-local/scripts/nucleo_inventory.sh` | HISTORICO | 1542 | 2 | #!/usr/bin/env bash |
| `governance/archive/cloud-local/scripts/prepare_training_pack.ps1` | HISTORICO | 901 | 1 | # F51 Darwin-SSD — PREPARADOR DE TRAINING PACK |
| `governance/archive/cloud-local/scripts/prepare_training_pack.py` | HISTORICO | 16656 | 7 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/pull_cloud_checkpoint.bat` | HISTORICO | 1795 | 1 | @echo off |
| `governance/archive/cloud-local/scripts/pull_cloud_checkpoint.sh` | HISTORICO | 2147 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/pull_from_hub.sh` | HISTORICO | 4616 | 2 | #!/usr/bin/env bash |
| `governance/archive/cloud-local/scripts/rent_gpu_math.py` | HISTORICO | 14874 | 16 | #!/usr/bin/env python |
| `governance/archive/cloud-local/scripts/rent_safe.sh` | HISTORICO | 1215 | 1 | #!/bin/bash |
| `governance/archive/cloud-local/scripts/run_organism_a100.py` | HISTORICO | 2593 | 1 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/run_vast_test.py` | HISTORICO | 3092 | 2 | import time |
| `governance/archive/cloud-local/scripts/start_overnight_25b.ps1` | HISTORICO | 7006 | 2 | param( |
| `governance/archive/cloud-local/scripts/sync_cloud_daemon.py` | HISTORICO | 5575 | 3 | #!/usr/bin/env python |
| `governance/archive/cloud-local/scripts/sync_code_to_hub.sh` | HISTORICO | 2983 | 1 | #!/usr/bin/env bash |
| `governance/archive/cloud-local/scripts/tokenize_local_batch_to_hub.py` | HISTORICO | 7551 | 2 | from __future__ import annotations |
| `governance/archive/cloud-local/scripts/tokenize_remote_corpus_batch.py` | HISTORICO | 1557 | 1 | from __future__ import annotations |
| `governance/archive/cloud-local/scripts/train_cloud.py` | HISTORICO | 3403 | 1 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/upload_corpus_pack.py` | HISTORICO | 1467 | 1 | from __future__ import annotations |
| `governance/archive/cloud-local/scripts/verify_5b_instantiation.py` | HISTORICO | 1807 | 1 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/verify_5b_params.py` | HISTORICO | 1403 | 1 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/verify_checkpoint_25b.py` | HISTORICO | 21104 | 2 | #!/usr/bin/env python3 |
| `governance/archive/cloud-local/scripts/verify_remote_token_batch.py` | HISTORICO | 1432 | 1 | from __future__ import annotations |
| `governance/archive/cloud-local/scripts/worker_c4.py` | HISTORICO | 687 | 5 | from datasets import load_dataset |
| `governance/archive/cloud-local/scripts/worker_fineweb.py` | HISTORICO | 721 | 5 | from datasets import load_dataset |
| `governance/archive/cloud-local/scripts/worker_gutenberg.py` | HISTORICO | 1337 | 1 | import urllib.request, json, time, re |
| `governance/archive/cloud-local/scripts/worker_pubmed.py` | HISTORICO | 1314 | 3 | import urllib.request, json, time |
| `governance/archive/cloud-local/tests/test_dae_cloud_canary.py` | HISTORICO | 3055 | 3 | import hashlib |
| `governance/archive/cloud-local/tests/test_train_cloud_tokens.py` | HISTORICO | 1376 | 1 | from __future__ import annotations |
| `governance/archive/configs/darwin_x_100m_cloud.yaml` | HISTORICO | 1596 | 1 | model_name: F51-Darwin-X-100M-Cloud |
| `governance/archive/configs/evolution_cycle.yaml` | HISTORICO | 399 | 1 | cycle_name: darwin_cycle_v0 |
| `governance/archive/dead_code/evolution_loop.py` | HISTORICO | 11949 | 2 | """ |
| `governance/archive/dead_code/evolution_sandbox.py` | HISTORICO | 8615 | 1 | from __future__ import annotations |
| `governance/archive/dead_code/expert_weight_router.py` | HISTORICO | 8430 | 2 | """ |
| `governance/archive/dead_code/ghost_token.py` | HISTORICO | 7394 | 1 | """ |
| `governance/archive/dead_code/jepa.py` | HISTORICO | 6021 | 1 | """ |
| `governance/archive/dead_code/organ_pipeline.py` | HISTORICO | 6563 | 1 | """F51 Organ Pipeline — Connects all 7 organs into one evolution loop. |
| `governance/archive/dead_code/sleep_consolidation.py` | HISTORICO | 13569 | 2 | """ |
| `governance/archive/dead_code/test_organ_pipeline.py` | HISTORICO | 2919 | 1 | from __future__ import annotations |
| `governance/archive/historical-docs/CHECKLIST_DAVI_FIXES.md` | HISTORICO | 2242 | 1 | # CHECKLIST - F51 Darwin Fixes |
| `governance/archive/historical-docs/F51_DARWIN_SSD_DIARIO_DE_BORDO.txt` | HISTORICO | 10215 | 1 | ================================================================================ |
| `governance/archive/historical-docs/LEIS_DE_DAVI.md` | HISTORICO | 12400 | 1 | # LEIS DE DAVI |
| `governance/archive/historical-docs/PLANO_INFERENCIA_TEMPO_REAL.md` | HISTORICO | 5833 | 1 | # PLANO: Aprendizado por Inferência em Tempo Real |
| `governance/archive/historical-docs/README.md` | HISTORICO | 270 | 1 | # Historical documents |
| `governance/archive/historical-docs/requirements-pre-gold.txt` | HISTORICO | 89 | 1 | torch>=2.0 |
| `governance/archive/historical-docs/ROADMAP_AUDIT.md` | HISTORICO | 8120 | 1 | # ROADMAP PÓS-AUDITORIA — F51-Darwin-SSD |
| `governance/archive/historical-docs/ROADMAP_DAVI_FIXES.md` | HISTORICO | 11264 | 1 | # ROADMAP - Correções Críticas Davi F51 |
| `governance/archive/historical-docs/ROADMAP_GATE0.md` | HISTORICO | 9370 | 1 | # F51 DARWIN-SSD — ROADMAP GATE 0 (Continual Learning) |
| `governance/archive/historical-docs/ROADMAP.md` | HISTORICO | 3959 | 1 | > **DOCUMENTO HISTORICO - NAO USAR COMO ESTADO VIVO** |
| `governance/archive/historical-evaluations/darwin_merged_v4.json` | HISTORICO | 3249 | 1 | { |
| `governance/archive/historical-evaluations/darwin_trained_v3.json` | HISTORICO | 3137 | 1 | { |
| `governance/archive/historical-evaluations/organism_cycle_235.json` | HISTORICO | 2594 | 13 | { |
| `governance/archive/historical-evaluations/organism_cycle_236.json` | HISTORICO | 2119 | 11 | { |
| `governance/archive/historical-evaluations/organism_cycle_237.json` | HISTORICO | 2822 | 19 | { |
| `governance/archive/historical-evaluations/organism_cycle_238.json` | HISTORICO | 2895 | 27 | { |
| `governance/archive/historical-evaluations/organism_cycle_239.json` | HISTORICO | 2783 | 15 | { |
| `governance/archive/historical-evaluations/README.md` | HISTORICO | 378 | 1 | # Historical evaluation snapshots |
| `governance/archive/historical-evaluations/tournament_summary.json` | HISTORICO | 20701 | 81 | [ |
| `governance/archive/legacy-scratch/clean_checkpoints.py` | HISTORICO | 412 | 1 | """HISTORICAL ONLY: destructive checkpoint loop; do not execute.""" |
| `governance/archive/legacy-scratch/README.md` | HISTORICO | 325 | 1 | # Legacy scratch scripts |
| `governance/archive/legacy-scratch/root/auto_train.bat` | HISTORICO | 998 | 1 | @echo off |
| `governance/archive/legacy-scratch/root/monitor_nucleo.ps1` | HISTORICO | 6908 | 2 | param( |
| `governance/archive/legacy-scratch/root/relaunch_training.bat` | HISTORICO | 611 | 1 | @echo off |
| `governance/archive/legacy-scratch/root/start_ghost_stream.bat` | HISTORICO | 656 | 1 | @echo off |
| `governance/archive/legacy-scratch/root/test_correcoes.py` | HISTORICO | 505 | 1 | """Legacy entrypoint retained without pretending printed output is a test. |
| `governance/archive/legacy-scratch/root/tokenize_fast.py` | HISTORICO | 15142 | 5 | #!/usr/bin/env python3 |
| `governance/archive/legacy-scratch/root/trainlord.bat` | HISTORICO | 2443 | 1 | @echo off |
| `governance/archive/legacy-scratch/root/unified_train.py` | HISTORICO | 22633 | 8 | #!/usr/bin/env python3 |
| `governance/archive/legacy-scratch/scripts/_bigctx.py` | HISTORICO | 1027 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_check_credit.py` | HISTORICO | 1027 | 1 | import urllib.request, json |
| `governance/archive/legacy-scratch/scripts/_check.py` | HISTORICO | 796 | 1 | import urllib.request, json |
| `governance/archive/legacy-scratch/scripts/_clean.py` | HISTORICO | 1087 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_cleanup_disk.ps1` | HISTORICO | 2899 | 1 | # HISTORICAL ONLY - DO NOT EXECUTE. |
| `governance/archive/legacy-scratch/scripts/_combo.py` | HISTORICO | 2708 | 3 | import subprocess, tempfile, tarfile, io, os |
| `governance/archive/legacy-scratch/scripts/_combo2.py` | HISTORICO | 2077 | 2 | import subprocess, tempfile, tarfile, io, os |
| `governance/archive/legacy-scratch/scripts/_ctx512.py` | HISTORICO | 1038 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_deploy_all.py` | HISTORICO | 2235 | 1 | import subprocess, tempfile, tarfile, io, gzip, os, time, threading |
| `governance/archive/legacy-scratch/scripts/_deploy.py` | HISTORICO | 1343 | 1 | import subprocess, os, tarfile, io |
| `governance/archive/legacy-scratch/scripts/_deploy2.py` | HISTORICO | 1677 | 1 | import subprocess, base64 |
| `governance/archive/legacy-scratch/scripts/_fix_fused.py` | HISTORICO | 1055 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_fix_nitro.py` | HISTORICO | 1139 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_fix_nitro2.py` | HISTORICO | 2092 | 7 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_fix_root.py` | HISTORICO | 1205 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_fix_sed.py` | HISTORICO | 3040 | 5 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_fix_soul2.py` | HISTORICO | 2410 | 3 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_force_cuda.py` | HISTORICO | 1596 | 3 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_fresh.py` | HISTORICO | 1700 | 1 | import urllib.request, json, time |
| `governance/archive/legacy-scratch/scripts/_launch_final.py` | HISTORICO | 1079 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_launch_moe.py` | HISTORICO | 1026 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/_rent.py` | HISTORICO | 1439 | 1 | import urllib.request, json, time |
| `governance/archive/legacy-scratch/scripts/_rent2.py` | HISTORICO | 2221 | 1 | import urllib.request, json, time, concurrent.futures |
| `governance/archive/legacy-scratch/scripts/_s1_code.py` | HISTORICO | 885 | 1 | import subprocess, tarfile, io |
| `governance/archive/legacy-scratch/scripts/_s2_corpus.py` | HISTORICO | 1394 | 1 | import subprocess, gzip, os, time |
| `governance/archive/legacy-scratch/scripts/_s2b.py` | HISTORICO | 469 | 1 | import urllib.request, json |
| `governance/archive/legacy-scratch/scripts/_s2c.py` | HISTORICO | 1841 | 1 | import subprocess, gzip, os, time |
| `governance/archive/legacy-scratch/scripts/_s3_launch.py` | HISTORICO | 1284 | 2 | import subprocess, tempfile |
| `governance/archive/legacy-scratch/scripts/run_neural_organism.py` | HISTORICO | 9543 | 2 | #!/usr/bin/env python |
| `governance/archive/legacy-scratch/scripts/run_organism_local.py` | HISTORICO | 972 | 1 | #!/usr/bin/env python3 |
| `governance/archive/legacy-state/f51/agent_memory.json` | HISTORICO | 173 | 1 | {"last_instance":"ssh5.vast.ai:32554","last_gpu":"RTX PRO 6000 WS 95GB","active_training":{"status":"running","ctx":256,"tok_s":100,"model":"1.6B Nitr |
| `governance/archive/legacy-state/f51/baseline/checkpoint_info.json` | HISTORICO | 4744 | 1 | { |
| `governance/archive/legacy-state/f51/baseline/tokenizer_metrics.json` | HISTORICO | 526 | 1 | { |
| `governance/archive/legacy-state/f51/partner/diff.txt` | HISTORICO | 32436 | 11 | diff --git a/src/scripts/darwin_organism.py b/src/scripts/darwin_organism.py |
| `governance/archive/legacy-state/f51/partner/review.md` | HISTORICO | 4149 | 1 | # Deep Review — Training Pipeline Patches (R1–R6) |
| `governance/archive/legacy-state/f51/research/architecture_weakness.md` | HISTORICO | 22017 | 1 | # F51 Darwin-X — Auditoria de Arquitetura e Pesquisa de Papers Subvalorizados |
| `governance/archive/legacy-state/f51/research/data_weakness.md` | HISTORICO | 22387 | 2 | # F51 Darwin-SSD — Fraquezas em Dados & Tokenizer |
| `governance/archive/legacy-state/f51/research/training_weakness.md` | HISTORICO | 17269 | 1 | # Training Pipeline - Fraquezas e Papers Subvalorizados |
| `governance/archive/README.md` | HISTORICO | 662 | 1 | # Archive — non-operational material |
| `governance/archive/scripts/_fast_gpus.py` | HISTORICO | 2222 | 2 | import json, sys |
| `governance/archive/scripts/_micro_infer.py` | HISTORICO | 4993 | 8 | """Micro-inference test — loads checkpoint on CPU, tests a few prompts.""" |
| `governance/archive/scripts/_parse_gpus.py` | HISTORICO | 2804 | 1 | import json, os, sys |
| `governance/archive/scripts/bob_agent.md` | HISTORICO | 412 | 1 | --- |
| `governance/audit/current/operational-surface-move-manifest.json` | GOVERNANCA / EVIDENCIA | 16030 | 1 | { |
| `governance/audit/current/physical-migration-manifest.json` | GOVERNANCA / EVIDENCIA | 3414 | 1 | { |
| `governance/audit/evidence/gitleaks-source-status.json` | GOVERNANCA / EVIDENCIA | 3673 | 1 | { |
| `governance/audit/evidence/supply-chain-index.json` | GOVERNANCA / EVIDENCIA | 4139 | 1 | { |
| `governance/audit/generated/.gitkeep` | GOVERNANCA / EVIDENCIA | 1 | 1 | — |
| `governance/audit/GOLD_REPORT_2026-07-17.md` | GOVERNANCA / EVIDENCIA | 5615 | 1 | # External audit simulation - 2026-07-17 |
| `governance/audit/policy/dependencies.json` | GOVERNANCA / EVIDENCIA | 3562 | 1 | { |
| `governance/audit/policy/distribution.json` | GOVERNANCA / EVIDENCIA | 976 | 1 | { |
| `governance/audit/policy/duplicates.json` | GOVERNANCA / EVIDENCIA | 1112 | 1 | { |
| `governance/audit/policy/operational-surface.json` | GOVERNANCA / EVIDENCIA | 34995 | 1 | { |
| `governance/audit/policy/physical-root.json` | GOVERNANCA / EVIDENCIA | 1973 | 1 | { |
| `governance/audit/policy/repository-governance.md` | GOVERNANCA / EVIDENCIA | 928 | 1 | # Repository governance |
| `governance/audit/provenance/evidence-attestation.json` | GOVERNANCA / EVIDENCIA | 1829 | 1 | { |
| `governance/audit/provenance/gold-lineage-summary.json` | GOVERNANCA / EVIDENCIA | 2859 | 1 | { |
| `governance/audit/provenance/nitro-runtime.json` | GOVERNANCA / EVIDENCIA | 4435 | 1 | { |
| `governance/audit/provenance/pre-migration.json` | GOVERNANCA / EVIDENCIA | 1162 | 1 | { |
| `governance/audit/provenance/retired-installers.json` | GOVERNANCA / EVIDENCIA | 2509 | 1 | { |
| `governance/audit/provenance/single-root-migration-summary.json` | GOVERNANCA / EVIDENCIA | 2289 | 1 | { |
| `governance/audit/provenance/source-manifest.json` | GOVERNANCA / EVIDENCIA | 4299 | 1 | { |
| `governance/audit/README.md` | GOVERNANCA / EVIDENCIA | 3765 | 1 | # Indice de auditoria |
| `governance/audit/sbom/cpu-audit-licenses.json` | GOVERNANCA / EVIDENCIA | 2672 | 1 | [ |
| `governance/audit/sbom/cpu-audit-vulnerabilities.json` | GOVERNANCA / EVIDENCIA | 912 | 1 | {"dependencies": [{"name": "colorama", "version": "0.4.6", "vulns": []}, {"name": "filelock", "version": "3.30.2", "vulns": []}, {"name": "fsspec", "v |
| `governance/audit/sbom/cpu-audit.cyclonedx.json` | GOVERNANCA / EVIDENCIA | 28214 | 1 | { |
| `governance/audit/sbom/nitro-runtime.cyclonedx.json` | GOVERNANCA / EVIDENCIA | 15904 | 1 | { |
| `governance/audit/schemas/gold-lineage.schema.json` | GOVERNANCA / EVIDENCIA | 615 | 1 | { |
| `governance/audit/schemas/gold-source-report.schema.json` | GOVERNANCA / EVIDENCIA | 6437 | 1 | { |
| `governance/audit/schemas/single-root-migration.schema.json` | GOVERNANCA / EVIDENCIA | 1525 | 1 | { |
| `governance/audit/schemas/supply-chain-index.schema.json` | GOVERNANCA / EVIDENCIA | 3225 | 1 | { |
| `governance/audit/test-reports/gold-source-f54a579.json` | GOVERNANCA / EVIDENCIA | 11174 | 147 | { |
| `governance/audit/test-reports/README.md` | GOVERNANCA / EVIDENCIA | 711 | 1 | # Source audit reports |
| `governance/audit/tooling/gitleaks-v8.30.1-windows-x64.json` | GOVERNANCA / EVIDENCIA | 1235 | 1 | { |
| `governance/audit/validators/assert_gold_source_report.ps1` | GOVERNANCA / EVIDENCIA | 6959 | 1 | [CmdletBinding()] |
| `CLAUDE.md` | GOVERNANCA / RAIZ | 4584 | 1 | # F51 Darwin-X - contrato de agentes |
| `src/configs/circuits/skills/induction-confirmation.json` | CONFIGURACAO | 275 | 1 | { |
| `src/configs/circuits/skills/induction-discovery.json` | CONFIGURACAO | 272 | 1 | { |
| `src/configs/circuits/skills/induction-offset-confirmation.json` | CONFIGURACAO | 289 | 1 | { |
| `src/configs/circuits/skills/induction-offset-discovery.json` | CONFIGURACAO | 286 | 1 | { |
| `src/configs/circuits/taps/distilgpt2-resid-pre.json` | CONFIGURACAO | 193 | 1 | { |
| `src/configs/darwin_transfer_distilgpt2_v1.yaml` | CONFIGURACAO | 931 | 1 | lineage_id: darwin-transfer-distilgpt2-v1 |
| `src/configs/darwin_x_1.6b_nitro.yaml` | CONFIGURACAO | 1358 | 1 | model_name: F51-Darwin-X-1.6B-Nitro |
| `src/configs/darwin_x_1.6b_smol_transplant.yaml` | CONFIGURACAO | 3348 | 1 | model_name: F51-Darwin-X-1.6B-Smol-Transplant-V1 |
| `src/configs/darwin_x_1.7b_smol_dense.yaml` | CONFIGURACAO | 2029 | 1 | model_name: F51-Darwin-X-1.7B-Smol-Native-Dense-V1 |
| `src/configs/darwin_x_1.7b_smol_exact.yaml` | CONFIGURACAO | 1995 | 1 | model_name: F51-Darwin-X-1.7B-Smol-Exact-Brain-V3 |
| `src/configs/darwin_x_100m_baseline_puro.yaml` | CONFIGURACAO | 2566 | 1 | # F51 Darwin-X 100M — Baseline puro (sem orgaos), pra medir ganho real |
| `src/configs/darwin_x_100m_dense_transformer_baseline.yaml` | CONFIGURACAO | 3040 | 1 | # F51 Darwin-X 100M — Transformer denso "de mercado" pra comparar contra o |
| `src/configs/darwin_x_100m_denso_full_organism.yaml` | CONFIGURACAO | 2842 | 1 | # F51 Darwin-X 100M — Transformer DENSO puro + órgãos full organism. |
| `src/configs/darwin_x_100m_full_organism.yaml` | CONFIGURACAO | 2735 | 1 | # F51 Darwin-X 100M — Full organism lineage. |
| `src/configs/darwin_x_100m.yaml` | CONFIGURACAO | 2246 | 1 | model_name: F51-Darwin-X-100M |
| `src/configs/darwin_x_600m.yaml` | CONFIGURACAO | 1883 | 1 | model_name: F51-Darwin-X-600M |
| `src/configs/data_factory.yaml` | CONFIGURACAO | 682 | 1 | factory_name: f51_data_immune_system |
| `DIARIO_DE_BORDO.md` | DIARIO / HISTORICO | 115730 | 1 | # Diário de Bordo — F51 Darwin-X |
| `governance/docs/_historico/AUDITORIA_CICLO_VIDA.md` | DOCUMENTO / HISTORICO | 6220 | 1 | # AUDITORIA: Conexões Neurais e Ciclo de Vida |
| `governance/docs/_historico/CEMITERIO_ICML.md` | DOCUMENTO / HISTORICO | 1994 | 1 | # Quarentena de referências não verificadas |
| `governance/docs/_historico/CLAUDE_OVERNIGHT_BENCHMARK_PROMPT_2026-07-15.md` | DOCUMENTO / HISTORICO | 16685 | 1 | # Prompt operacional para Claude — benchmark causal noturno F51 |
| `governance/docs/_historico/darwin_x_5b_spec.md` | DOCUMENTO / HISTORICO | 6256 | 1 | # F51-Darwin-X-5B-Nitro — Especificação Completa |
| `governance/docs/_historico/INDEX.md` | DOCUMENTO / HISTORICO | 884 | 1 | # Indice historico |
| `governance/docs/_historico/STATUS_ATUAL_2026-07-23.md` | DOCUMENTO / HISTORICO | 13199 | 1 | # Snapshot historico de 2026-07-23 -- F51 Darwin-X 100M |
| `governance/docs/_historico/TRAINING_PACK.md` | DOCUMENTO / HISTORICO | 2736 | 1 | # F51 Darwin-SSD — Training Pack para Nuvem |
| `governance/docs/arquitetura/BRAIN_COMPARISON_2025.md` | DOCUMENTO / ARQUITETURA | 7587 | 1 | # F51 Darwin-X vs Cérebro Humano |
| `governance/docs/arquitetura/DECISAO_ARQUITETURAL_LIVE_GENERATE.md` | DOCUMENTO / ARQUITETURA | 1570 | 1 | # Decisao Arquitetural: live_generate ↔ InferenceLearner |
| `governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md` | DOCUMENTO / ARQUITETURA | 3701 | 1 | # Implementacao atual |
| `governance/docs/arquitetura/NEUROENDOCRINE_IMPLEMENTATION.md` | DOCUMENTO / ARQUITETURA | 6215 | 1 | # Darwin-X neuroendócrino — estado científico verificável |
| `governance/docs/arquitetura/ORGANISMO_COMPLETO.md` | DOCUMENTO / ARQUITETURA | 17382 | 1 | # Organismo Darwin-X — Arquitetura completa |
| `governance/docs/arquitetura/REDESIGN_MODELO.md` | DOCUMENTO / ARQUITETURA | 17188 | 1 | # Redesign Completo — Darwin-X Organism v8 |
| `governance/docs/arquitetura/SSM_STATE_16_VS_64.md` | DOCUMENTO / ARQUITETURA | 2866 | 1 | # Decisao tecnica: 'ssm_state=16' versus '64' |
| `governance/docs/auditoria/2026-07-25/auditoria_14_orgaos_v9_cycle71.txt` | DOCUMENTO / CANONICO OU VARIAVEL | 24043 | 1 | ================================================================================ |
| `governance/docs/CANONICAL_MAP.md` | DOCUMENTO / CANONICO OU VARIAVEL | 4163 | 1 | # Mapa canonico Darwin-X |
| `governance/docs/BRAINSTORM_2026-07-30.md` | PLANO / NAO-AUTORIDADE | 2560 | 57 | # SUPERBRAINSTORM — F51 Darwin-X |
| `governance/docs/CONVERGENCE_ANALYSIS.md` | DOCUMENTO / CANONICO OU VARIAVEL | 3226 | 1 | # The Convergence: What Three Null Results Actually Prove |
| `governance/docs/dashboard_architecture_plan.md` | DOCUMENTO / CANONICO OU VARIAVEL | 38144 | 4 | # F51 Darwin-X Mission Control — Dashboard Architecture Plan |
| `governance/docs/dashboard_component_spec.json` | DOCUMENTO / CANONICO OU VARIAVEL | 62586 | 1 | { |
| `governance/docs/dashboard_data_flow.md` | DOCUMENTO / CANONICO OU VARIAVEL | 82002 | 23 | # F51 Darwin-X Mission Control -- Data Flow & Wire Protocol |
| `governance/docs/F51_DARWIN_BLUEBOOK.md` | DOCUMENTO / CANONICO OU VARIAVEL | 7124 | 1 | # F51 Darwin-X Bluebook |
| `governance/docs/F51_DARWIN_SYSTEM_BIBLE.md` | DOCUMENTO / CANONICO OU VARIAVEL | 12275 | 263 | # F51 Darwin-X — Bíblia do Sistema e Dicionário Técnico |
| `governance/docs/F51_DARWIN_SYSTEM_FILE_INDEX.md` | DOCUMENTO / CANONICO OU VARIAVEL | 36957 | 960 | # F51 Darwin-X — Índice Mecânico de Arquivos |
| `governance/docs/F51_DARWIN_SYSTEM_SYMBOL_DICTIONARY.md` | DOCUMENTO / CANONICO OU VARIAVEL | 179489 | 1 | # F51 Darwin-X — Dicionário de Símbolos Python |
| `governance/docs/F51_DARWIN_FILE_DICTIONARY.md` | DOCUMENTO / CANONICO OU VARIAVEL | 96772 | 871 | # F51 Darwin-X — Dicionário Semântico de Arquivos |
| `governance/docs/F51_DARWIN_PHYSICAL_INVENTORY.md` | DOCUMENTO / CANONICO OU VARIAVEL | 8448 | 152 | # F51 Darwin-X — Inventário Físico da Raiz |
| `src/tests/test_provenance_certificate.py` | TESTE / VALIDACAO | 8087 | 203 | A fronteira permutacao<->byte, e o acoplamento SHA + deriva. |
| `governance/docs/operacao/BENCHMARK_REFERENCE.md` | DOCUMENTO / OPERACAO | 2762 | 1 | # F51 Darwin-SSD - Referencia de avaliacao |
| `governance/docs/operacao/HISTORIA_GHOST.md` | DOCUMENTO / OPERACAO | 18799 | 1 | # Historia do orgao Ghost -- F51 Darwin-X |
| `governance/docs/operacao/HISTORIA_LINHAGENS.md` | DOCUMENTO / OPERACAO | 19176 | 1 | # Historia das Linhagens -- F51 Darwin-X 100M |
| `governance/docs/operacao/NATIVE_TTM_RECALL.md` | DOCUMENTO / OPERACAO | 5886 | 1 | # Operação — prova causal de memória TTM nativa |
| `governance/docs/operacao/OPERACAO_SEGURA.md` | DOCUMENTO / OPERACAO | 2891 | 1 | # Operacao segura |
| `governance/docs/operacao/SKILL_DEEP_REFLECTION.md` | DOCUMENTO / OPERACAO | 2174 | 1 | # 🧠 Deep Reflection Protocol — Codex v1 |
| `governance/docs/operacao/SOCIOS.md` | DOCUMENTO / OPERACAO | 1955 | 1 | # 🤝 Sociedade F51 |
| `governance/docs/operacao/STATUS_ATUAL.md` | DOCUMENTO / OPERACAO | 22793 | 1 | <!-- authority: current-status --> |
| `governance/docs/pesquisa/EXPERIMENTO_CONTINUAL_LEARNING.md` | DOCUMENTO / PESQUISA | 602 | 1 | # Experimento Controlado — Darwin-X Continual Learning |
| `governance/docs/pesquisa/F51_CONTINUAL_LEARNING_RND_PLAN.md` | DOCUMENTO / PESQUISA | 18584 | 1 | # F51 Continual Learning - Plano de R&D e Concept Note PARI-CNRC |
| `governance/docs/pesquisa/FONTES_PRIMARIAS_1940_1951.md` | DOCUMENTO / PESQUISA | 33035 | 1 | # Fontes Primárias 1940–1951 — Arquivo Documental |
| `governance/docs/pesquisa/FUSAO_EXPERTS_RESEARCH.md` | DOCUMENTO / PESQUISA | 3039 | 1 | # Fusão de experts — protocolo de pesquisa |
| `governance/docs/pesquisa/GHOST_RECURRENCE_RESEARCH.md` | DOCUMENTO / PESQUISA | 9925 | 1 | # Ghost recorrente — pesquisa, desenho e simulação |
| `governance/docs/pesquisa/ONLINE_LEARNING_RESEARCH.md` | DOCUMENTO / PESQUISA | 21235 | 1 | # Aprendizado online no Darwin-X: evidência, limites e direção de implementação |
| `governance/docs/pesquisa/PESQUISA_4_EIXOS.md` | DOCUMENTO / PESQUISA | 6974 | 1 | # Pesquisa 4 Eixos — Construindo o Organismo Vivo |
| `governance/docs/pesquisa/relatorio_eixo_1_termodinamica_shannon.md` | DOCUMENTO / PESQUISA | 4709 | 1 | # Eixo 1 — Termodinâmica, Entropia, Mecânica Estatística & Teoria da Informação (1940–1951) |
| `governance/docs/pesquisa/relatorio_eixo_2_wiener_cibernetica.md` | DOCUMENTO / PESQUISA | 5730 | 1 | # Eixo 2 — Cibernética, Feedback, Previsão & Comportamento Teleológico (1940–1951) |
| `governance/docs/pesquisa/relatorio_eixo_3_mcculloch_turing_vonneumann.md` | DOCUMENTO / PESQUISA | 6467 | 1 | # Eixo 3 — McCulloch-Pitts, Turing & von Neumann (1940–1952) |
| `governance/docs/pesquisa/relatorio_eixo_4_ashby_inferencia_causalidade.md` | DOCUMENTO / PESQUISA | 8964 | 1 | # Eixo 4 — Ashby, Homeostase, Inferência, Causalidade, Memória & Irreversibilidade (1940–1951) |
| `governance/docs/superpowers/plans/2026-07-15-16b-recovery-observability-canary.md` | PLANO / NAO-AUTORIDADE | 73565 | 8 | # Darwin-X 1.6B Recovery, Observability, and Canary Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-16-16b-a100-verified-resume-training.md` | PLANO / NAO-AUTORIDADE | 6710 | 1 | # Darwin-X 1.6B A100 Verified Resume Training Plan |
| `governance/docs/superpowers/plans/2026-07-16-darwin-active-gradient-engine.md` | PLANO / NAO-AUTORIDADE | 15602 | 1 | # Darwin Active Gradient Engine V1 Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-16-single-root-gold-audit.md` | PLANO / NAO-AUTORIDADE | 34440 | 1 | # F51 Darwin-X Single-Root GOLD Audit Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-17-internal-gold-hardening.md` | PLANO / NAO-AUTORIDADE | 15687 | 1 | # Internal Gold Hardening Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-18-causal-cycle-boundary.md` | PLANO / NAO-AUTORIDADE | 2878 | 1 | # Causal Cycle Boundary Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-18-causal-organism-v1.md` | PLANO / NAO-AUTORIDADE | 10389 | 1 | # F51 Causal Organism v1 Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-18-causal-organism-v2.md` | PLANO / NAO-AUTORIDADE | 43433 | 1 | # F51 Causal Organism v2 Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-18-checkpoint-lineage-isolation-recovery.md` | PLANO / NAO-AUTORIDADE | 15164 | 1 | # Checkpoint Lineage Isolation and Recovery Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-19-100m-full-organism-65b.md` | PLANO / NAO-AUTORIDADE | 7680 | 1 | # Darwin-X 100M Full Organism 65B Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-26-100m-cloud-v2.md` | PLANO / NAO-AUTORIDADE | 42880 | 1 | # Darwin-X 100M Cloud V2 Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-27-darwin-transfer-organism.md` | PLANO / NAO-AUTORIDADE | 6972 | 1 | # Darwin Transfer Organism Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-28-cognitive-circuit-vertical-slice.md` | PLANO / NAO-AUTORIDADE | 36304 | 1 | # Cognitive Circuit Vertical Slice Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-28-darwin-16b-smol-full-brain-transplant.md` | PLANO / NAO-AUTORIDADE | 47628 | 1 | # Darwin 1.6B Smol Full-Brain Transplant Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-28-two-donor-organ-transplant-first-slice.md` | PLANO / NAO-AUTORIDADE | 30624 | 1 | # Two-Donor Organ Transplant First Slice Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-29-darwin-organ-causal-qa.md` | PLANO / NAO-AUTORIDADE | 7636 | 1 | # Darwin Individual-Organ Causal QA Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-29-darwin-smol-native-dense-organism.md` | PLANO / NAO-AUTORIDADE | 21490 | 1 | # Darwin-Smol Native Dense Organism Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-29-native-ttm-channel-surgery.md` | PLANO / NAO-AUTORIDADE | 11677 | 1 | # Native TTM Channel Surgery Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-29-native-ttm-memory-recall.md` | PLANO / NAO-AUTORIDADE | 18127 | 1 | # Native TTM Memory Recall Implementation Plan |
| `governance/docs/superpowers/plans/2026-07-29-three-organ-cognition-foundation.md` | PLANO / NAO-AUTORIDADE | 60591 | 1 | # Three-Organ Cognition Foundation Implementation Plan |
| `governance/docs/superpowers/specs/2026-07-15-16b-recovery-observability-canary-design.md` | SPEC / NAO-AUTORIDADE | 12000 | 1 | # Darwin-X 1.6B Recovery, Observability, and Canary Design |
| `governance/docs/superpowers/specs/2026-07-16-darwin-active-gradient-engine-design.md` | SPEC / NAO-AUTORIDADE | 6750 | 1 | # Darwin Active Gradient Engine V1 — Design |
| `governance/docs/superpowers/specs/2026-07-16-documentation-reconciliation-design.md` | SPEC / NAO-AUTORIDADE | 14603 | 1 | # Darwin-X Documentation Reconciliation Design |
| `governance/docs/superpowers/specs/2026-07-16-single-root-gold-audit-design.md` | SPEC / NAO-AUTORIDADE | 14193 | 1 | # F51 Darwin-X Single-Root GOLD Audit Design |
| `governance/docs/superpowers/specs/2026-07-17-internal-gold-hardening-design.md` | SPEC / NAO-AUTORIDADE | 13514 | 1 | # Internal Gold Hardening Design |
| `governance/docs/superpowers/specs/2026-07-18-causal-organism-v1-design.md` | SPEC / NAO-AUTORIDADE | 7227 | 1 | # F51 Causal Organism v1 — Design |
| `governance/docs/superpowers/specs/2026-07-18-causal-organism-v2-design.md` | SPEC / NAO-AUTORIDADE | 18951 | 1 | # F51 Causal Organism v2 — Two-Timescale Design |
| `governance/docs/superpowers/specs/2026-07-18-checkpoint-lineage-isolation-recovery-design.md` | SPEC / NAO-AUTORIDADE | 9885 | 1 | # Checkpoint Lineage Isolation and Gold Recovery Design |
| `governance/docs/superpowers/specs/2026-07-18-f51-system-document-design.md` | SPEC / NAO-AUTORIDADE | 7972 | 1 | # F51 Darwin-X — Design do documento institucional e técnico |
| `governance/docs/superpowers/specs/2026-07-19-100m-full-organism-65b-design.md` | SPEC / NAO-AUTORIDADE | 4033 | 1 | # Darwin-X 100M Full Organism 65B Design |
| `governance/docs/superpowers/specs/2026-07-26-100m-cloud-v2-learning-progress-dopamine-design.md` | SPEC / NAO-AUTORIDADE | 17718 | 1 | # Darwin-X 100M Cloud V2: Motor Corrigido e Dopamina por Progresso |
| `governance/docs/superpowers/specs/2026-07-27-darwin-transfer-organism-design.md` | SPEC / NAO-AUTORIDADE | 5777 | 1 | # Darwin Transfer Organism — Design |
| `governance/docs/superpowers/specs/2026-07-27-learning-gain-simulation-design.md` | SPEC / NAO-AUTORIDADE | 12609 | 1 | # Learning-Gain Simulation — Design |
| `governance/docs/superpowers/specs/2026-07-28-cognitive-circuit-ablation-transplant-design.md` | SPEC / NAO-AUTORIDADE | 17526 | 1 | # Darwin Cognitive Circuit Ablation and Transplant - Design |
| `governance/docs/superpowers/specs/2026-07-28-darwin-16b-smol-full-brain-transplant-design.md` | SPEC / NAO-AUTORIDADE | 16939 | 1 | # Darwin 1.6B SmolLM Full-Brain Transplant Design |
| `governance/docs/superpowers/specs/2026-07-28-two-donor-organ-transplant-design.md` | SPEC / NAO-AUTORIDADE | 12339 | 1 | # Darwin Two-Donor Organ Transplant - Design |
| `governance/docs/superpowers/specs/2026-07-29-darwin-organ-causal-qa-design.md` | SPEC / NAO-AUTORIDADE | 4400 | 1 | # Darwin Individual-Organ Causal QA Design |
| `governance/docs/superpowers/specs/2026-07-29-darwin-smol-native-dense-organism-design.md` | SPEC / NAO-AUTORIDADE | 7399 | 1 | # Darwin-Smol Native Dense Organism — Design |
| `governance/docs/superpowers/specs/2026-07-29-native-ttm-channel-surgery-design.md` | SPEC / NAO-AUTORIDADE | 5174 | 1 | # Design — cirurgia do canal TTM nativo |
| `governance/docs/superpowers/specs/2026-07-29-native-ttm-memory-recall-design.md` | SPEC / NAO-AUTORIDADE | 6169 | 1 | # Native TTM Memory Recall — desenho experimental |
| `governance/docs/superpowers/specs/2026-07-29-three-universal-organs-design.md` | SPEC / NAO-AUTORIDADE | 23079 | 1 | # Design - tres orgaos cognitivos universais |
| `src/f51_darwin/__init__.py` | RUNTIME / PRODUTO | 3137 | 1 | """F51 Darwin-SSD public API with dependency-lazy exports. |
| `src/f51_darwin/__main__.py` | RUNTIME / PRODUTO | 132 | 1 | """'python -m f51_darwin' entrypoint.""" |
| `src/f51_darwin/_version.py` | RUNTIME / PRODUTO | 86 | 1 | """Distribution version without runtime dependency imports.""" |
| `src/f51_darwin/anti_woke_filter.py` | RUNTIME / PRODUTO | 2079 | 1 | from __future__ import annotations |
| `src/f51_darwin/artifact_manifest.py` | RUNTIME / PRODUTO | 7330 | 3 | from __future__ import annotations |
| `src/f51_darwin/artifacts.py` | RUNTIME / PRODUTO | 22655 | 1 | from __future__ import annotations |
| `src/f51_darwin/attention_block.py` | RUNTIME / PRODUTO | 2409 | 1 | from __future__ import annotations |
| `src/f51_darwin/benchmark.py` | RUNTIME / PRODUTO | 22235 | 3 | """ |
| `src/f51_darwin/brainstem.py` | RUNTIME / PRODUTO | 6741 | 1 | """F51 Brainstem — Homeostasis enforcer. |
| `src/f51_darwin/checkpoint_eval.py` | RUNTIME / PRODUTO | 5059 | 1 | from __future__ import annotations |
| `src/f51_darwin/checkpointing.py` | RUNTIME / PRODUTO | 418 | 1 | """Backward-compatible imports for the canonical organism checkpoint API.""" |
| `src/f51_darwin/circuits/__init__.py` | RUNTIME / PRODUTO | 2763 | 1 | """Canonical identities and manifests for portable Darwin-X circuits.""" |
| `src/f51_darwin/circuits/ablation.py` | RUNTIME / PRODUTO | 51353 | 1 | """Discovery-split ranking and independently anchored paired causal ablation.""" |
| `src/f51_darwin/circuits/compatibility.py` | RUNTIME / PRODUTO | 25132 | 1 | """Fail-closed static and executable compatibility preflight.""" |
| `src/f51_darwin/circuits/identity.py` | RUNTIME / PRODUTO | 432 | 1 | from __future__ import annotations |
| `src/f51_darwin/circuits/ledger.py` | RUNTIME / PRODUTO | 38912 | 4 | """Canonical append-only circuit ledger with externally anchored history.""" |
| `src/f51_darwin/circuits/manifest.py` | RUNTIME / PRODUTO | 10513 | 1 | from __future__ import annotations |
| `src/f51_darwin/circuits/package.py` | RUNTIME / PRODUTO | 12940 | 1 | """Verified, deterministic on-disk packages for portable Darwin-X circuits.""" |
| `src/f51_darwin/circuits/pointer.py` | RUNTIME / PRODUTO | 19809 | 1 | """Crash-safe compare-and-swap publication for the active circuit checkpoint.""" |
| `src/f51_darwin/circuits/rollback.py` | RUNTIME / PRODUTO | 30989 | 1 | """Immutable, identity-checked selective circuit rollback checkpoints.""" |
| `src/f51_darwin/circuits/skills.py` | RUNTIME / PRODUTO | 11167 | 1 | """Datasets determinísticos de habilidade para descoberta e confirmação causal. |
| `src/f51_darwin/circuits/taps.py` | RUNTIME / PRODUTO | 6501 | 1 | """Strict, allowlisted activation taps for portable circuits.""" |
| `src/f51_darwin/circuits/transaction.py` | RUNTIME / PRODUTO | 29642 | 1 | """Gate-zero circuit installation wrappers and immutable lifecycle records.""" |
| `src/f51_darwin/cli.py` | RUNTIME / PRODUTO | 1488 | 1 | """Audit-safe command line surface for the installed distribution.""" |
| `src/f51_darwin/cognition/__init__.py` | RUNTIME / PRODUTO | 978 | 1 | from .contracts import ( |
| `src/f51_darwin/cognition/adapters.py` | RUNTIME / PRODUTO | 3065 | 1 | from __future__ import annotations |
| `src/f51_darwin/cognition/contracts.py` | RUNTIME / PRODUTO | 4745 | 1 | from __future__ import annotations |
| `src/f51_darwin/cognition/executive.py` | RUNTIME / PRODUTO | 11963 | 1 | """UniversalExecutive — trajectory scoring and action selection organ. |
| `src/f51_darwin/cognition/memory.py` | RUNTIME / PRODUTO | 26235 | 1 | """UniversalMemory — persistent associative memory organ. |
| `src/f51_darwin/cognition/pulse.py` | RUNTIME / PRODUTO | 3188 | 1 | from __future__ import annotations |
| `src/f51_darwin/cognition/runtime.py` | RUNTIME / PRODUTO | 9132 | 1 | from __future__ import annotations |
| `src/f51_darwin/cognition/world_model.py` | RUNTIME / PRODUTO | 15111 | 1 | """HierarchicalWorldModel — semantic future prediction organ. |
| `src/f51_darwin/config.py` | RUNTIME / PRODUTO | 5406 | 1 | from __future__ import annotations |
| `src/f51_darwin/corpus_cloud_common.py` | RUNTIME / PRODUTO | 340 | 1 | from __future__ import annotations |
| `src/f51_darwin/corpus_cloud_ingest.py` | RUNTIME / PRODUTO | 4526 | 2 | from __future__ import annotations |
| `src/f51_darwin/corpus_cloud_token_index.py` | RUNTIME / PRODUTO | 6014 | 3 | from __future__ import annotations |
| `src/f51_darwin/corpus_cloud_token_verify.py` | RUNTIME / PRODUTO | 4977 | 1 | from __future__ import annotations |
| `src/f51_darwin/corpus_cloud_tokenize.py` | RUNTIME / PRODUTO | 6807 | 4 | from __future__ import annotations |
| `src/f51_darwin/corpus_factory_pipeline.py` | RUNTIME / PRODUTO | 7770 | — | binario; metadado estrutural apenas |
| `src/f51_darwin/corpus_pack_upload.py` | RUNTIME / PRODUTO | 4973 | 2 | from __future__ import annotations |
| `src/f51_darwin/corpus_pack_verifier.py` | RUNTIME / PRODUTO | 4459 | 1 | from __future__ import annotations |
| `src/f51_darwin/corpus_policy.py` | RUNTIME / PRODUTO | 7422 | 1 | from __future__ import annotations |
| `src/f51_darwin/creator_key.py` | RUNTIME / PRODUTO | 7708 | 4 | """ |
| `src/f51_darwin/curiosity.py` | RUNTIME / PRODUTO | 28363 | 1 | """ |
| `src/f51_darwin/dae_optimizer.py` | RUNTIME / PRODUTO | 6886 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/__init__.py` | RUNTIME / PRODUTO | 1003 | 1 | from f51_darwin.darwin_x_core.block import DarwinXBlock |
| `src/f51_darwin/darwin_x_core/block.py` | RUNTIME / PRODUTO | 2911 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/config.py` | RUNTIME / PRODUTO | 15196 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/estimation.py` | RUNTIME / PRODUTO | 2502 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/layers.py` | RUNTIME / PRODUTO | 12832 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/losses.py` | RUNTIME / PRODUTO | 4730 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/model.py` | RUNTIME / PRODUTO | 76170 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/moe.py` | RUNTIME / PRODUTO | 61543 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/neuroendocrine.py` | RUNTIME / PRODUTO | 30087 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_core/state.py` | RUNTIME / PRODUTO | 3962 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x_training.py` | RUNTIME / PRODUTO | 12529 | 1 | from __future__ import annotations |
| `src/f51_darwin/darwin_x.py` | RUNTIME / PRODUTO | 1311 | 1 | from __future__ import annotations |
| `src/f51_darwin/dashboard/__init__.py` | RUNTIME / PRODUTO | 347 | 1 | """F51 Darwin-X Dashboard — data layer package.""" |
| `src/f51_darwin/dashboard/data_layer.py` | RUNTIME / PRODUTO | 46219 | 13 | """ |
| `src/f51_darwin/data_factory.py` | RUNTIME / PRODUTO | 13391 | 4 | from __future__ import annotations |
| `src/f51_darwin/data_firewall.py` | RUNTIME / PRODUTO | 4523 | 1 | from __future__ import annotations |
| `src/f51_darwin/data.py` | RUNTIME / PRODUTO | 22662 | 17 | from __future__ import annotations |
| `src/f51_darwin/dataset_layout.py` | RUNTIME / PRODUTO | 5735 | 1 | from __future__ import annotations |
| `src/f51_darwin/dataset_states.py` | RUNTIME / PRODUTO | 967 | 1 | from __future__ import annotations |
| `src/f51_darwin/decision_engine.py` | RUNTIME / PRODUTO | 13548 | 1 | """ |
| `src/f51_darwin/distillation.py` | RUNTIME / PRODUTO | 1406 | 1 | from __future__ import annotations |
| `src/f51_darwin/duckduckgo_search.py` | RUNTIME / PRODUTO | 4496 | 3 | """Small, defensive DuckDuckGo search adapter. |
| `src/f51_darwin/evolution_gate.py` | RUNTIME / PRODUTO | 3855 | 1 | from __future__ import annotations |
| `src/f51_darwin/evolution_score.py` | RUNTIME / PRODUTO | 906 | 1 | from __future__ import annotations |
| `src/f51_darwin/expert_cache.py` | RUNTIME / PRODUTO | 8283 | 1 | """ |
| `src/f51_darwin/expert_pool.py` | RUNTIME / PRODUTO | 7141 | 1 | from __future__ import annotations |
| `src/f51_darwin/gaba_inhibition.py` | RUNTIME / PRODUTO | 27336 | 2 | """ |
| `src/f51_darwin/ghost_brain.py` | RUNTIME / PRODUTO | 17848 | 5 | """ |
| `src/f51_darwin/ghost_corpus.py` | RUNTIME / PRODUTO | 21684 | 9 | """ |
| `src/f51_darwin/grounded_extractor.py` | RUNTIME / PRODUTO | 8716 | 10 | """F51 Grounded Extractor — Codex as project microscope, not truth source. |
| `src/f51_darwin/hashing.py` | RUNTIME / PRODUTO | 2087 | 2 | """Canonical hashing utilities for F51 Darwin-X. |
| `src/f51_darwin/heartbeat.py` | RUNTIME / PRODUTO | 30354 | 1 | """ |
| `src/f51_darwin/identity_corpus.py` | RUNTIME / PRODUTO | 30877 | 6 | """ |
| `src/f51_darwin/inference_engine.py` | RUNTIME / PRODUTO | 15630 | 1 | """ |
| `src/f51_darwin/inference_learner.py` | RUNTIME / PRODUTO | 36729 | 12 | """Safe online learning for Davi. |
| `src/f51_darwin/ingestion/__init__.py` | RUNTIME / PRODUTO | 115 | 1 | """Quarantine-first local ingestion runtime.""" |
| `src/f51_darwin/ingestion/runtime.py` | RUNTIME / PRODUTO | 12667 | 23 | #!/usr/bin/env python3 |
| `src/f51_darwin/inter_hemispheric.py` | RUNTIME / PRODUTO | 12787 | 2 | """ |
| `src/f51_darwin/jepa_v2.py` | RUNTIME / PRODUTO | 31793 | 2 | """ |
| `src/f51_darwin/kv_cache.py` | RUNTIME / PRODUTO | 15876 | 1 | """ |
| `src/f51_darwin/legacy_layers.py` | RUNTIME / PRODUTO | 21678 | 4 | """ |
| `src/f51_darwin/lineage_tracker.py` | RUNTIME / PRODUTO | 13338 | 4 | """F51 Lineage Tracker — Árvore genealógica dos módulos do organismo. |
| `src/f51_darwin/math_genius.py` | RUNTIME / PRODUTO | 16820 | 20 | """ |
| `src/f51_darwin/math_organ.py` | RUNTIME / PRODUTO | 1575 | 1 | from __future__ import annotations |
| `src/f51_darwin/metrics.py` | RUNTIME / PRODUTO | 3926 | 2 | from __future__ import annotations |
| `src/f51_darwin/model.py` | RUNTIME / PRODUTO | 13573 | 1 | from __future__ import annotations |
| `src/f51_darwin/moe_layer.py` | RUNTIME / PRODUTO | 16408 | 1 | """ |
| `src/f51_darwin/moe_training.py` | RUNTIME / PRODUTO | 23567 | 3 | """ |
| `src/f51_darwin/olavo_curator.py` | RUNTIME / PRODUTO | 23998 | 20 | """F51 Olavo Super Dataset — Curadoria do corpus filosófico para o bebê F51. |
| `src/f51_darwin/operations/__init__.py` | RUNTIME / PRODUTO | 50 | 1 | """Repository-local operational launch assets.""" |
| `src/f51_darwin/operations/start_overnight_16b.ps1` | RUNTIME / PRODUTO | 32809 | 4 | param( |
| `src/f51_darwin/operations/start_smol_darwin_transplant.ps1` | RUNTIME / PRODUTO | 1807 | 1 | [CmdletBinding()] |
| `src/f51_darwin/orchestrator.py` | RUNTIME / PRODUTO | 11918 | 1 | """ |
| `src/f51_darwin/organism/__init__.py` | RUNTIME / PRODUTO | 1545 | 1 | """Operational Darwin organism runtime. |
| `src/f51_darwin/organism/autonomous_drives.py` | RUNTIME / PRODUTO | 27367 | 1 | """ |
| `src/f51_darwin/organism/blockchain_utils.py` | RUNTIME / PRODUTO | 2769 | 3 | """Blockchain-aware checkpoint utilities for the Darwin organism. |
| `src/f51_darwin/organism/blockchain.py` | RUNTIME / PRODUTO | 29871 | 6 | """Darwin-X Organ Blockchain — append-only block chain for organ state provenance. |
| `src/f51_darwin/organism/bootstrap.py` | RUNTIME / PRODUTO | 51745 | 3 | from __future__ import annotations |
| `src/f51_darwin/organism/causal_adapters.py` | RUNTIME / PRODUTO | 22123 | 1 | """Pure training adapters and narrow executors for causal interventions. |
| `src/f51_darwin/organism/causal_bus.py` | RUNTIME / PRODUTO | 40212 | 1 | """Typed, deterministic causal-intervention contracts for the organism runtime. |
| `src/f51_darwin/organism/causal_ledger.py` | RUNTIME / PRODUTO | 50573 | 10 | """Canonical append-only causal ledger with semantic online validation.""" |
| `src/f51_darwin/organism/checkpoint_mixin.py` | RUNTIME / PRODUTO | 21319 | 7 | from __future__ import annotations |
| `src/f51_darwin/organism/checkpoint_root.py` | RUNTIME / PRODUTO | 16762 | 1 | from __future__ import annotations |
| `src/f51_darwin/organism/checkpoint.py` | RUNTIME / PRODUTO | 30539 | 2 | """Single checkpoint authority for Darwin training and organism state. |
| `src/f51_darwin/organism/cli.py` | RUNTIME / PRODUTO | 28047 | 8 | from __future__ import annotations |
| `src/f51_darwin/organism/config.py` | RUNTIME / PRODUTO | 5802 | 1 | from __future__ import annotations |
| `src/f51_darwin/organism/control.py` | RUNTIME / PRODUTO | 13128 | 1 | from __future__ import annotations |
| `src/f51_darwin/organism/dependencies.py` | RUNTIME / PRODUTO | 2527 | 1 | """Shared dependency context for organism responsibility modules.""" |
| `src/f51_darwin/organism/layer_transplant.py` | RUNTIME / PRODUTO | 6270 | 1 | """Native Layer Transplant Manager — Direct Internal Block Integration. |
| `src/f51_darwin/organism/lifecycle.py` | RUNTIME / PRODUTO | 37599 | 2 | from __future__ import annotations |
| `src/f51_darwin/organism/organ_adapters.py` | RUNTIME / PRODUTO | 4182 | 1 | """Pure adapters for lagged causal-organ observation frames.""" |
| `src/f51_darwin/organism/organ_frames.py` | RUNTIME / PRODUTO | 5541 | 1 | """Immutable, delayed observation frames for causal organism feedback.""" |
| `src/f51_darwin/organism/organ_identity.py` | RUNTIME / PRODUTO | 10714 | 1 | """Per-organ SHA-256 content identities for the Darwin organism. |
| `src/f51_darwin/organism/organ_reputation.py` | RUNTIME / PRODUTO | 5628 | 1 | """ |
| `src/f51_darwin/organism/organ_rl.py` | RUNTIME / PRODUTO | 11922 | 1 | """Organ RL Optimizer — Aprende quais combinações de órgãos funcionam. |
| `src/f51_darwin/organism/organ_senate.py` | RUNTIME / PRODUTO | 11122 | 1 | """ |
| `src/f51_darwin/organism/runtime.py` | RUNTIME / PRODUTO | 379 | 1 | """Compatibility facade for the modular Darwin organism runtime.""" |
| `src/f51_darwin/organism/support.py` | RUNTIME / PRODUTO | 8554 | 1 | from __future__ import annotations |
| `src/f51_darwin/organism/training.py` | RUNTIME / PRODUTO | 46948 | 2 | from __future__ import annotations |
| `src/f51_darwin/organism/unified_mesh.py` | RUNTIME / PRODUTO | 20849 | 1 | from __future__ import annotations |
| `src/f51_darwin/provenance.py` | RUNTIME / PRODUTO | 15299 | 3 | from __future__ import annotations |
| `src/f51_darwin/pruning.py` | RUNTIME / PRODUTO | 1562 | 1 | from __future__ import annotations |
| `src/f51_darwin/quality_gate.py` | RUNTIME / PRODUTO | 10997 | 19 | """ |
| `src/f51_darwin/replay_buffer.py` | RUNTIME / PRODUTO | 4680 | 1 | from __future__ import annotations |
| `src/f51_darwin/rope.py` | RUNTIME / PRODUTO | 3079 | 1 | from __future__ import annotations |
| `src/f51_darwin/router.py` | RUNTIME / PRODUTO | 714 | 1 | from __future__ import annotations |
| `src/f51_darwin/semantic_probe.py` | RUNTIME / PRODUTO | 15195 | 1 | from __future__ import annotations |
| `src/f51_darwin/serving/__init__.py` | RUNTIME / PRODUTO | 62 | 1 | """Davi serving application and packaged static resources.""" |
| `src/f51_darwin/serving/runtime.py` | RUNTIME / PRODUTO | 29548 | 2 | #!/usr/bin/env python3 |
| `src/f51_darwin/serving/static/davi.html` | RUNTIME / PRODUTO | 13228 | 1 | <!doctype html> |
| `src/f51_darwin/soul.py` | RUNTIME / PRODUTO | 31151 | 15 | """ |
| `src/f51_darwin/spider_sense.py` | RUNTIME / PRODUTO | 16652 | 4 | """ |
| `src/f51_darwin/ssd_block.py` | RUNTIME / PRODUTO | 2091 | 1 | from __future__ import annotations |
| `src/f51_darwin/ssm_core.py` | RUNTIME / PRODUTO | 25367 | 1 | """ |
| `src/f51_darwin/state_identity.py` | RUNTIME / PRODUTO | 10296 | 1 | """Content identities for online-learning lineage. |
| `src/f51_darwin/synthetic_pt_generator.py` | RUNTIME / PRODUTO | 5374 | 1 | from __future__ import annotations |
| `src/f51_darwin/tokenizer_plan.py` | RUNTIME / PRODUTO | 901 | 1 | from __future__ import annotations |
| `src/f51_darwin/tokenizer.py` | RUNTIME / PRODUTO | 14539 | 9 | from __future__ import annotations |
| `src/f51_darwin/training_observability.py` | RUNTIME / PRODUTO | 26538 | 5 | from __future__ import annotations |
| `src/f51_darwin/training.py` | RUNTIME / PRODUTO | 11187 | 2 | from __future__ import annotations |
| `src/f51_darwin/transfer/__init__.py` | RUNTIME / PRODUTO | 229 | 1 | """Transformer-to-SSD architecture transfer for the Darwin organism.""" |
| `src/f51_darwin/transfer/checkpoint.py` | RUNTIME / PRODUTO | 3382 | 2 | from __future__ import annotations |
| `src/f51_darwin/transfer/data.py` | RUNTIME / PRODUTO | 6259 | 5 | from __future__ import annotations |
| `src/f51_darwin/transfer/donor.py` | RUNTIME / PRODUTO | 4645 | 2 | from __future__ import annotations |
| `src/f51_darwin/transfer/experience.py` | RUNTIME / PRODUTO | 4445 | 1 | from __future__ import annotations |
| `src/f51_darwin/transfer/jepa_predictor.py` | RUNTIME / PRODUTO | 1885 | 1 | """JEPA predictor minimal — integrado ao Darwin Transfer Runtime. |
| `src/f51_darwin/transfer/ngram.py` | RUNTIME / PRODUTO | 4832 | 1 | """Baseline de contagem para controlar holdouts degenerados. |
| `src/f51_darwin/transfer/runtime_v2.py` | RUNTIME / PRODUTO | 4738 | 1 | """Darwin Transfer Runtime V2 — TTM + JEPA + Backbone. |
| `src/f51_darwin/transfer/runtime.py` | RUNTIME / PRODUTO | 3691 | 1 | from __future__ import annotations |
| `src/f51_darwin/transfer/schedule.py` | RUNTIME / PRODUTO | 2719 | 1 | from __future__ import annotations |
| `src/f51_darwin/transfer/ssd_mixer.py` | RUNTIME / PRODUTO | 6042 | 1 | from __future__ import annotations |
| `src/f51_darwin/transfer/student.py` | RUNTIME / PRODUTO | 4037 | 1 | from __future__ import annotations |
| `src/f51_darwin/transfer/trainer.py` | RUNTIME / PRODUTO | 8554 | 2 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/__init__.py` | RUNTIME / PRODUTO | 261 | 1 | """Native SmolLM2-to-Darwin 1.6B weight-surgery package.""" |
| `src/f51_darwin/transplant_16b/assembly.py` | RUNTIME / PRODUTO | 45831 | 2 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/calibration.py` | RUNTIME / PRODUTO | 14458 | 4 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/checkpoint.py` | RUNTIME / PRODUTO | 7243 | 3 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/cli.py` | RUNTIME / PRODUTO | 8602 | 2 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/contracts.py` | RUNTIME / PRODUTO | 7804 | 3 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/dense_assembly.py` | RUNTIME / PRODUTO | 14772 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/dense_runtime.py` | RUNTIME / PRODUTO | 3866 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/exact_assembly.py` | RUNTIME / PRODUTO | 18592 | 2 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/layers.py` | RUNTIME / PRODUTO | 4262 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/ledger.py` | RUNTIME / PRODUTO | 11399 | 5 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/moe.py` | RUNTIME / PRODUTO | 9663 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/organ_causal_qa.py` | RUNTIME / PRODUTO | 5854 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/organs.py` | RUNTIME / PRODUTO | 7594 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/projection.py` | RUNTIME / PRODUTO | 4837 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/selection.py` | RUNTIME / PRODUTO | 11550 | 1 | """Reducao por selecao, dentro do grupo de simetria de permutacao. |
| `src/f51_darwin/transplant_16b/sources.py` | RUNTIME / PRODUTO | 2329 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/ssd_fit.py` | RUNTIME / PRODUTO | 6276 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/tokenizer.py` | RUNTIME / PRODUTO | 5345 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant_16b/verification.py` | RUNTIME / PRODUTO | 1695 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/__init__.py` | RUNTIME / PRODUTO | 2470 | 1 | """Two-donor organ transplantation primitives.""" |
| `src/f51_darwin/transplant/bundle.py` | RUNTIME / PRODUTO | 11769 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/checkpoint.py` | RUNTIME / PRODUTO | 8673 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/experiment.py` | RUNTIME / PRODUTO | 3972 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/ledger.py` | RUNTIME / PRODUTO | 7649 | 2 | from __future__ import annotations |
| `src/f51_darwin/transplant/lifecycle.py` | RUNTIME / PRODUTO | 3699 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/organs.py` | RUNTIME / PRODUTO | 14145 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/recipient.py` | RUNTIME / PRODUTO | 14687 | 1 | from __future__ import annotations |
| `src/f51_darwin/transplant/slots.py` | RUNTIME / PRODUTO | 16023 | 1 | from __future__ import annotations |
| `src/f51_darwin/turbo_tokenizer.py` | RUNTIME / PRODUTO | 6769 | 4 | """ |
| `src/f51_darwin/turbo.py` | RUNTIME / PRODUTO | 6129 | 1 | """F51 CUDA TURBO — Geração autoregressiva com KV cache (SSD h + conv + attn KV).""" |
| `src/f51_darwin/wolfram_bridge.py` | RUNTIME / PRODUTO | 7210 | 2 | """F51 Wolfram Bridge — The organism's mathematical cortex extension. |
| `src/f51_darwin/workspace_migration.py` | RUNTIME / PRODUTO | 23657 | 3 | from __future__ import annotations |
| `F51-JEPA-100M-2.0` | GITLINK / FRONTEIRA | — | — | gitlink; consultar inventario fisico e checkout separado |
| `LICENSE` | GOVERNANCA / RAIZ | 830 | 1 | F51 LABS PROPRIETARY LICENSE |
| `MANIFEST.in` | PACOTE / DEPENDENCIA | 236 | 1 | include LICENSE |
| `NOTICE` | GOVERNANCA / RAIZ | 458 | 1 | F51 Darwin-X / F51-Darwin-SSD |
| `pyproject.toml` | PACOTE / DEPENDENCIA | 1227 | 1 | [build-system] |
| `README.md` | ARQUIVO / RAIZ | 3895 | 1 | # F51 Darwin-X |
| `requirements-cpu-audit-dev.in` | PACOTE / DEPENDENCIA | 111 | 1 | # CPU-safe source audit environment; exact direct pins. |
| `requirements-cpu-audit-dev.lock` | PACOTE / DEPENDENCIA | 22876 | 1 | # |
| `requirements-cpu-audit.in` | PACOTE / DEPENDENCIA | 125 | 1 | # Supported local Darwin runtime; exact direct pins, resolved in requirements.lock. |
| `requirements-cpu-audit.lock` | PACOTE / DEPENDENCIA | 21624 | 1 | # |
| `research/bench_expert_cache.py` | PESQUISA / EXPERIMENTO | 2529 | 1 | #!/usr/bin/env python |
| `research/bench_ssd.py` | PESQUISA / EXPERIMENTO | 888 | 1 | import torch, time |
| `research/benchmark_darwin.py` | PESQUISA / EXPERIMENTO | 13604 | 13 | #!/usr/bin/env python3 |
| `research/build_classical_corpus.py` | PESQUISA / EXPERIMENTO | 40651 | 21 | #!/usr/bin/env python3 |
| `research/darwin_school.py` | PESQUISA / EXPERIMENTO | 13403 | 7 | #!/usr/bin/env python3 |
| `research/download_classical_corpus.py` | PESQUISA / EXPERIMENTO | 19927 | 12 | """Download classical liberal / conservative corpus into F51 candidate staging. |
| `research/download_olavo_sources.py` | PESQUISA / EXPERIMENTO | 12639 | 8 | """Download Olavo de Carvalho free sources into F51 candidate staging. |
| `research/evolve_darwin.py` | PESQUISA / EXPERIMENTO | 14972 | 6 | #!/usr/bin/env python3 |
| `research/experiment_runner_v2.py` | PESQUISA / EXPERIMENTO | 15779 | 3 | #!/usr/bin/env python3 |
| `research/experiment_runner.py` | PESQUISA / EXPERIMENTO | 14848 | 3 | #!/usr/bin/env python3 |
| `research/gen_status_report.py` | PESQUISA / EXPERIMENTO | 12110 | 1 | #!/usr/bin/env python3 |
| `research/generate_bible.py` | PESQUISA / EXPERIMENTO | 5678 | 69 | #!/usr/bin/env python |
| `research/generate_dataset_candidates.py` | PESQUISA / EXPERIMENTO | 2673 | 1 | from __future__ import annotations |
| `research/generate_generic_corpus.py` | PESQUISA / EXPERIMENTO | 28149 | 68 | #!/usr/bin/env python |
| `research/generate_mass_math.py` | PESQUISA / EXPERIMENTO | 12139 | 38 | #!/usr/bin/env python |
| `research/generate_math_corpus.py` | PESQUISA / EXPERIMENTO | 13924 | 12 | #!/usr/bin/env python |
| `research/generate_math_monster.py` | PESQUISA / EXPERIMENTO | 7063 | 23 | #!/usr/bin/env python |
| `research/generate_philosophers.py` | PESQUISA / EXPERIMENTO | 5165 | 34 | #!/usr/bin/env python |
| `research/generate_python_50m.py` | PESQUISA / EXPERIMENTO | 29006 | 9 | #!/usr/bin/env python |
| `research/generate_synthetic_pt_20k.py` | PESQUISA / EXPERIMENTO | 3150 | 4 | from __future__ import annotations |
| `research/generate_timeline_pdf.py` | PESQUISA / EXPERIMENTO | 43258 | 1 | #!/usr/bin/env python3 |
| `research/ghost_feeder.py` | PESQUISA / EXPERIMENTO | 9126 | 1 | #!/usr/bin/env python3 |
| `research/ghost_stream_runner.ps1` | PESQUISA / EXPERIMENTO | 6674 | 1 | <# |
| `research/ghost_stream.py` | PESQUISA / EXPERIMENTO | 17299 | 23 | #!/usr/bin/env python3 |
| `research/grow_model.py` | PESQUISA / EXPERIMENTO | 8231 | 1 | #!/usr/bin/env python |
| `research/install_ghost_stream_task.ps1` | PESQUISA / EXPERIMENTO | 5258 | 1 | <# |
| `research/learning_gain/__init__.py` | PESQUISA / EXPERIMENTO | 333 | 1 | """Learning-Gain Simulation — isolated scientific experiment. |
| `research/learning_gain/evaluator.py` | PESQUISA / EXPERIMENTO | 15159 | 1 | """Independent evaluator — hidden ground truth, probes, metrics, leakage checks. |
| `research/learning_gain/inference_demo.py` | PESQUISA / EXPERIMENTO | 7687 | 5 | """Inference demo — GainAdaptive vs Frozen on S01+S02+S05. |
| `research/learning_gain/policies.py` | PESQUISA / EXPERIMENTO | 12744 | 1 | """Learning policies: Frozen, AlwaysUpdate, SurpriseOnly, GainAdaptive, RandomGate. |
| `research/learning_gain/runner.py` | PESQUISA / EXPERIMENTO | 20989 | 6 | """Paired-seed execution and artifact assembly. |
| `research/learning_gain/scenarios.py` | PESQUISA / EXPERIMENTO | 15577 | 1 | """Deterministic scenario generators S01–S10. |
| `research/learning_gain/state.py` | PESQUISA / EXPERIMENTO | 13227 | 1 | """Learner state, bounded memory, update actions, and cost accounting. |
| `research/maquinista.py` | PESQUISA / EXPERIMENTO | 20057 | 2 | #!/usr/bin/env python |
| `research/math_overdrive.py` | PESQUISA / EXPERIMENTO | 26128 | 31 | #!/usr/bin/env python |
| `research/mine_corpus_agents.py` | PESQUISA / EXPERIMENTO | 26134 | 7 | #!/usr/bin/env python3 |
| `research/prepare_base_training.py` | PESQUISA / EXPERIMENTO | 3876 | 2 | from __future__ import annotations |
| `research/quick_ppl.py` | PESQUISA / EXPERIMENTO | 9814 | 1 | #!/usr/bin/env python3 |
| `research/README.md` | PESQUISA / EXPERIMENTO | 342 | 1 | # Research and experiments |
| `research/reingest_olavo.py` | PESQUISA / EXPERIMENTO | 3090 | 1 | """Purge failed Olavo firewall records and re-import from staging + audit.""" |
| `research/run_evolution_cycle.py` | PESQUISA / EXPERIMENTO | 2637 | 2 | from __future__ import annotations |
| `research/semantic_probe.py` | PESQUISA / EXPERIMENTO | 3835 | 1 | #!/usr/bin/env python3 |
| `research/serve_f51.py` | PESQUISA / EXPERIMENTO | 18560 | 4 | #!/usr/bin/env python |
| `research/serve_model.py` | PESQUISA / EXPERIMENTO | 11619 | 3 | #!/usr/bin/env python |
| `research/simulate_ghost_recurrence.py` | PESQUISA / EXPERIMENTO | 20777 | 2 | """Deterministic simulation of recurrent Ghost memory. |
| `research/split_ghost_corpus.py` | PESQUISA / EXPERIMENTO | 4196 | 1 | #!/usr/bin/env python |
| `research/stream_all.py` | PESQUISA / EXPERIMENTO | 1944 | 2 | import sys, struct, os |
| `research/suck_wolfram.py` | PESQUISA / EXPERIMENTO | 21481 | 15 | #!/usr/bin/env python |
| `research/test_10_cenarios.py` | PESQUISA / EXPERIMENTO | 483 | 1 | #!/usr/bin/env python3 |
| `research/test_live_generate.py` | PESQUISA / EXPERIMENTO | 2455 | 2 | #!/usr/bin/env python3 |
| `research/tokenize_classical_corpus.py` | PESQUISA / EXPERIMENTO | 5047 | 2 | #!/usr/bin/env python3 |
| `research/train_base.py` | PESQUISA / EXPERIMENTO | 8456 | 1 | from __future__ import annotations |
| `research/train_curriculum.py` | PESQUISA / EXPERIMENTO | 7139 | 1 | #!/usr/bin/env python3 |
| `research/train_darwin_x.py` | PESQUISA / EXPERIMENTO | 9189 | 1 | #!/usr/bin/env python3 |
| `research/train_seed_toy.py` | PESQUISA / EXPERIMENTO | 2861 | 3 | from __future__ import annotations |
| `research/train_tokenizer.py` | PESQUISA / EXPERIMENTO | 2837 | 1 | from __future__ import annotations |
| `research/turbo_train.py` | PESQUISA / EXPERIMENTO | 14204 | 4 | #!/usr/bin/env python3 |
| `research/uninstall_ghost_stream_task.ps1` | PESQUISA / EXPERIMENTO | 1722 | 1 | <# |
| `research/weight_heist.py` | PESQUISA / EXPERIMENTO | 14092 | 3 | #!/usr/bin/env python3 |
| `src/scripts/__init__.py` | SCRIPT / SUPERFICIE MISTA | 81 | 1 | """Supported command adapters; invoke them with ''python -m scripts.<name>''.""" |
| `research/ablation_memory_bridge.py` | PESQUISA / EXPERIMENTO | 5862 | 3 | #!/usr/bin/env python3 |
| `research/analyze_margins.py` | PESQUISA / EXPERIMENTO | 4195 | 1 | import os, torch |
| `research/benchmark_darwin_organs_qa.py` | PESQUISA / EXPERIMENTO | 19358 | 2 | #!/usr/bin/env python3 |
| `research/benchmark_darwin_vs_baseline.py` | PESQUISA / EXPERIMENTO | 4539 | 6 | """Script de Benchmark Comparativo — F51 Darwin-X 100M vs. Dense Transformer Baseline. |
| `research/benchmark_native_ttm_recall.py` | PESQUISA / EXPERIMENTO | 51708 | 6 | #!/usr/bin/env python3 |
| `research/benchmark_smol_dense_qa.py` | PESQUISA / EXPERIMENTO | 23391 | 7 | #!/usr/bin/env python3 |
| `research/benchmark_universal_memory_recall.py` | PESQUISA / EXPERIMENTO | 17522 | 13 | #!/usr/bin/env python3 |
| `src/scripts/bob.ps1` | SCRIPT / SUPERFICIE MISTA | 1587 | 1 | # ── F51 Bob Agent (GLM 5.2 via z.ai) ───────────────────────────────────────── |
| `src/tools/build_circuit_registry.py` | FERRAMENTA / MANUTENCAO | 5215 | 4 | #!/usr/bin/env python3 |
| `research/calibrate_smol_darwin_transplant.py` | PESQUISA / EXPERIMENTO | 3763 | 2 | #!/usr/bin/env python3 |
| `research/causal_domain_ablation.py` | PESQUISA / EXPERIMENTO | 15321 | 24 | #!/usr/bin/env python3 |
| `src/tools/checkpoint_repath_source.py` | FERRAMENTA / MANUTENCAO | 2497 | 1 | #!/usr/bin/env python3 |
| `research/circuit_ablate.py` | PESQUISA / EXPERIMENTO | 13814 | 2 | #!/usr/bin/env python3 |
| `research/compare_memory_adapters.py` | PESQUISA / EXPERIMENTO | 13158 | 9 | #!/usr/bin/env python3 |
| `src/tools/corpus_compress_for_transfer.py` | FERRAMENTA / MANUTENCAO | 6941 | 1 | #!/usr/bin/env python3 |
| `src/tools/corpus_decompress_on_receive.py` | FERRAMENTA / MANUTENCAO | 2652 | 1 | #!/usr/bin/env python3 |
| `src/tools/corpus_verify_roundtrip.py` | FERRAMENTA / MANUTENCAO | 2060 | 1 | #!/usr/bin/env python3 |
| `src/tools/darwin_dashboard_web.py` | FERRAMENTA / MANUTENCAO | 57841 | 20 | #!/usr/bin/env python3 |
| `src/tools/darwin_dashboard.py` | FERRAMENTA / MANUTENCAO | 11560 | 5 | #!/usr/bin/env python3 |
| `src/scripts/darwin_inventory.py` | SCRIPT / SUPERFICIE MISTA | 4111 | 1 | from __future__ import annotations |
| `src/scripts/darwin_organism.py` | ENTRYPOINT / OPERACAO | 1078 | 1 | #!/usr/bin/env python3 |
| `research/diagnose_native_ttm_recall.py` | PESQUISA / EXPERIMENTO | 18618 | 5 | #!/usr/bin/env python3 |
| `research/direction_vs_neuron_domain.py` | PESQUISA / EXPERIMENTO | 16654 | 27 | #!/usr/bin/env python3 |
| `research/discover_best_direction.py` | PESQUISA / EXPERIMENTO | 8281 | 8 | """4-METHOD DIRECTION DISCOVERY for identity circuit editing. |
| `research/discover_fact_channels.py` | PESQUISA / EXPERIMENTO | 17774 | 11 | #!/usr/bin/env python3 |
| `research/discover_identity_layer.py` | PESQUISA / EXPERIMENTO | 7229 | 5 | #!/usr/bin/env python3 |
| `research/eval_darwin_transfer.py` | PESQUISA / EXPERIMENTO | 1976 | 1 | from __future__ import annotations |
| `research/evaluate_smol_darwin_transplant.py` | PESQUISA / EXPERIMENTO | 1015 | 1 | #!/usr/bin/env python3 |
| `research/evaluate_smol_dense_candidate.py` | PESQUISA / EXPERIMENTO | 2493 | 1 | #!/usr/bin/env python3 |
| `research/evaluate_smol_exact_candidate.py` | PESQUISA / EXPERIMENTO | 2181 | 2 | #!/usr/bin/env python3 |
| `src/tools/extract_darwin_organs.py` | FERRAMENTA / MANUTENCAO | 1559 | 1 | #!/usr/bin/env python3 |
| `research/full_holdout_eval.py` | PESQUISA / EXPERIMENTO | 14831 | 1 | #!/usr/bin/env python3 |
| `src/tools/gen_organ_ablation_configs.ps1` | FERRAMENTA / MANUTENCAO | 2686 | 1 | # Gera 11 configs de ablacao (baseline + 1 orgao ligado por vez) a partir de |
| `research/gradient_alignment.py` | PESQUISA / EXPERIMENTO | 13634 | 2 | #!/usr/bin/env python3 |
| `research/gradient_attribution.py` | PESQUISA / EXPERIMENTO | 24888 | 18 | #!/usr/bin/env python3 |
| `research/inference_test.py` | PESQUISA / EXPERIMENTO | 4715 | 1 | #!/usr/bin/env python3 |
| `src/scripts/ingest_pipeline.py` | SCRIPT / SUPERFICIE MISTA | 477 | 1 | #!/usr/bin/env python3 |
| `research/inspect_ffn.py` | ENTRYPOINT / OPERACAO | 1718 | 1 | import os, torch |
| `src/scripts/inspect_organism_checkpoint.py` | ENTRYPOINT / OPERACAO | 8138 | 1 | #!/usr/bin/env python3 |
| `research/massive_identity_swap.py` | PESQUISA / EXPERIMENTO | 5426 | 5 | """MASSIVE TEST: Full identity swap with linear probe direction (d=11.41, 100% acc). |
| `research/memit_identity.py` | PESQUISA / EXPERIMENTO | 7184 | 6 | """MEMIT — Mass-Editing Memory in a Transformer. |
| `research/memory_entity_sim.py` | PESQUISA / EXPERIMENTO | 38698 | 9 | #!/usr/bin/env python3 |
| `research/native_smol_darwin_smoke.py` | PESQUISA / EXPERIMENTO | 5066 | 1 | #!/usr/bin/env python3 |
| `research/probe_donor_basis_alignment.py` | PESQUISA / EXPERIMENTO | 17348 | 3 | #!/usr/bin/env python3 |
| `research/probe_three_organ_shadow.py` | PESQUISA / EXPERIMENTO | 8970 | 1 | #!/usr/bin/env python3 |
| `research/probe_training_distribution.py` | PESQUISA / EXPERIMENTO | 25943 | 2 | #!/usr/bin/env python |
| `research/quick_validate_v2.py` | PESQUISA / EXPERIMENTO | 4589 | 4 | import os, torch |
| `src/scripts/README.md` | SCRIPT / SUPERFICIE MISTA | 821 | 1 | # Scripts |
| `research/rome_aggressive.py` | PESQUISA / EXPERIMENTO | 2535 | 2 | import os, torch |
| `research/rome_identity_edit_instruct.py` | PESQUISA / EXPERIMENTO | 22886 | 8 | #!/usr/bin/env python3 |
| `research/rome_identity_edit.py` | PESQUISA / EXPERIMENTO | 8277 | 8 | #!/usr/bin/env python3 |
| `research/rome_instruct_identity.py` | PESQUISA / EXPERIMENTO | 7069 | 7 | #!/usr/bin/env python3 |
| `research/rome_proper_identity.py` | PESQUISA / EXPERIMENTO | 19899 | 7 | #!/usr/bin/env python3 |
| `research/router_dual_path_sim.py` | PESQUISA / EXPERIMENTO | 24803 | 3 | #!/usr/bin/env python3 |
| `research/run_learning_gain_sim.py` | PESQUISA / EXPERIMENTO | 6635 | 5 | #!/usr/bin/env python3 |
| `research/run_two_donor_canary.py` | PESQUISA / EXPERIMENTO | 33772 | 2 | from __future__ import annotations |
| `research/scan_ffn_neurons.py` | PESQUISA / EXPERIMENTO | 18534 | 3 | #!/usr/bin/env python3 |
| `research/scan_model_circuits.py` | PESQUISA / EXPERIMENTO | 18443 | 6 | #!/usr/bin/env python3 |
| `src/scripts/serve_davi.py` | ENTRYPOINT / OPERACAO | 995 | 1 | #!/usr/bin/env python3 |
| `research/simulate_organ_combinatorics.py` | PESQUISA / EXPERIMENTO | 7577 | 5 | #!/usr/bin/env python3 |
| `src/scripts/start_100m_65b.ps1` | ENTRYPOINT / OPERACAO | 10582 | 1 | [CmdletBinding()] |
| `src/scripts/start_100m_auto.ps1` | ENTRYPOINT / OPERACAO | 17947 | 1 | param( |
| `src/tools/start_darwin_transfer.ps1` | ENTRYPOINT / OPERACAO | 4188 | 1 | param( |
| `src/scripts/start_overnight_16b.ps1` | ENTRYPOINT / OPERACAO | 1137 | 1 | param( |
| `src/scripts/start_smol_darwin_transplant.ps1` | ENTRYPOINT / OPERACAO | 261 | 1 | [CmdletBinding()] |
| `src/tools/start_two_donor_transplant.ps1` | ENTRYPOINT / OPERACAO | 5782 | 1 | param( |
| `src/tools/static/chart.min.js` | FERRAMENTA / MANUTENCAO | 205222 | 3 | /** |
| `src/tools/static/dashboard.css` | FERRAMENTA / MANUTENCAO | 39633 | 1 | /* ========================================================================== |
| `src/tools/static/htmx.min.js` | FERRAMENTA / MANUTENCAO | 50917 | 2 | var htmx=function(){"use strict";const Q={onLoad:null,process:null,on:null,off:null,trigger:null,ajax:null,find:null,findAll:null,closest:null,values: |
| `src/tools/templates/dashboard.html` | FERRAMENTA / MANUTENCAO | 57295 | 3 | <!DOCTYPE html> |
| `research/test_identity_circuit_real.py` | PESQUISA / EXPERIMENTO | 10233 | 10 | #!/usr/bin/env python3 |
| `research/test_identity_contrast.py` | PESQUISA / EXPERIMENTO | 7204 | 13 | #!/usr/bin/env python3 |
| `research/test_sae_domain.py` | PESQUISA / EXPERIMENTO | 7735 | 11 | """SAE domain separation test. Trains SAE on Python + Medicine activations |
| `research/train_darwin_transfer.py` | PESQUISA / EXPERIMENTO | 16256 | 2 | from __future__ import annotations |
| `research/train_memory_adapters.py` | PESQUISA / EXPERIMENTO | 6179 | 3 | #!/usr/bin/env python3 |
| `research/train_memory_real.py` | PESQUISA / EXPERIMENTO | 23021 | 4 | #!/usr/bin/env python3 |
| `research/train_paraphrase_invariance.py` | PESQUISA / EXPERIMENTO | 14116 | 7 | #!/usr/bin/env python3 |
| `research/train_tiny_sae.py` | PESQUISA / EXPERIMENTO | 2467 | 1 | import os, torch |
| `research/transplant_circuit_atomic.py` | PESQUISA / EXPERIMENTO | 11150 | 5 | #!/usr/bin/env python3 |
| `research/transplant_domain.py` | PESQUISA / EXPERIMENTO | 5120 | 7 | """Cross-checkpoint domain transplant: SmolLM2-Instruct -> SmolLM2-Base. |
| `research/transplant_internal_layer.py` | PESQUISA / EXPERIMENTO | 4681 | 3 | #!/usr/bin/env python3 |
| `research/transplant_layer_direct.py` | PESQUISA / EXPERIMENTO | 4850 | 7 | """Direct layer transplant: SmolLM2-Instruct -> SmolLM2-Base. |
| `research/transplant_layer_svd.py` | PESQUISA / EXPERIMENTO | 7001 | 8 | """Layer transplant: Llama-3.2-3B FFN -> SmolLM2-1.7B via SVD projection. |
| `research/transplant_smol_dense_brain.py` | PESQUISA / EXPERIMENTO | 1320 | 1 | #!/usr/bin/env python3 |
| `research/transplant_smol_exact_brain.py` | PESQUISA / EXPERIMENTO | 1187 | 1 | #!/usr/bin/env python3 |
| `src/tools/transplant_smol_to_darwin_1_6b.py` | FERRAMENTA / MANUTENCAO | 128 | 1 | #!/usr/bin/env python3 |
| `research/use_darwin_transfer_v2.py` | PESQUISA / EXPERIMENTO | 9300 | 7 | #!/usr/bin/env python3 |
| `research/use_darwin_transfer.py` | PESQUISA / EXPERIMENTO | 3834 | 1 | from __future__ import annotations |
| `research/validate_memory_full_cycle.py` | PESQUISA / EXPERIMENTO | 13343 | 4 | #!/usr/bin/env python3 |
| `research/validate_memory_training.py` | PESQUISA / EXPERIMENTO | 3968 | 2 | import os, gc, torch |
| `research/validate_v2_paraphrases.py` | PESQUISA / EXPERIMENTO | 6331 | 4 | import os, gc, json, torch |
| `src/tools/watch_training_v2.ps1` | FERRAMENTA / MANUTENCAO | 7993 | 1 | param( |
| `src/tools/watch_training.ps1` | FERRAMENTA / MANUTENCAO | 1701 | 1 | param( |
| `research/weight_sim.py` | PESQUISA / EXPERIMENTO | 14482 | 2 | #!/usr/bin/env python3 |
| `SECURITY.md` | GOVERNANCA / RAIZ | 1247 | 1 | # Security policy |
| `src/tests/fixtures/corpus/sample.txt` | TESTE / VALIDACAO | 201 | 1 | F51 Darwin-SSD grows only when necessary. |
| `src/tests/test_100m_65b_launcher.py` | TESTE / VALIDACAO | 1394 | 1 | from __future__ import annotations |
| `src/tests/test_100m_full_config.py` | TESTE / VALIDACAO | 2533 | 1 | from __future__ import annotations |
| `src/tests/test_adapt_checkpoint_darwin_x.py` | TESTE / VALIDACAO | 1595 | 1 | from __future__ import annotations |
| `src/tests/test_architecture_boundaries.py` | TESTE / VALIDACAO | 7099 | 26 | from __future__ import annotations |
| `src/tests/test_artifact_resolution.py` | TESTE / VALIDACAO | 14225 | 1 | from __future__ import annotations |
| `src/tests/test_audit_controls.py` | TESTE / VALIDACAO | 7782 | 5 | from __future__ import annotations |
| `src/tests/test_canary_launcher_contract.py` | TESTE / VALIDACAO | 9162 | 2 | from pathlib import Path |
| `src/tests/test_canonical_docs.py` | TESTE / VALIDACAO | 2767 | 4 | from __future__ import annotations |
| `src/tests/test_causal_ablation.py` | TESTE / VALIDACAO | 6107 | 2 | from __future__ import annotations |
| `src/tests/test_causal_checkpoint_contract.py` | TESTE / VALIDACAO | 12912 | 2 | from __future__ import annotations |
| `src/tests/test_causal_cognitive_organs.py` | TESTE / VALIDACAO | 24514 | 1 | from __future__ import annotations |
| `src/tests/test_causal_cycle_boundary.py` | TESTE / VALIDACAO | 8653 | 1 | from __future__ import annotations |
| `src/tests/test_causal_ledger_deadlock_repro.py` | TESTE / VALIDACAO | 7979 | 1 | """Regression test for the CausalLedger self-deadlock (FIXED). |
| `src/tests/test_causal_ledger.py` | TESTE / VALIDACAO | 13104 | 3 | from __future__ import annotations |
| `src/tests/test_causal_training_phases.py` | TESTE / VALIDACAO | 17147 | 1 | from __future__ import annotations |
| `src/tests/test_checkpoint_eval.py` | TESTE / VALIDACAO | 5066 | 2 | from __future__ import annotations |
| `src/tests/test_checkpoint_resume.py` | TESTE / VALIDACAO | 3141 | 1 | from dataclasses import replace |
| `src/tests/test_checkpoint_root_policy.py` | TESTE / VALIDACAO | 4532 | 3 | from __future__ import annotations |
| `src/tests/test_circuit_ablation_runtime.py` | TESTE / VALIDACAO | 14435 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_ablation_tap.py` | TESTE / VALIDACAO | 5688 | 1 | """run_paired_ablation com TapContract explícito. |
| `src/tests/test_circuit_compatibility.py` | TESTE / VALIDACAO | 16257 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_identity.py` | TESTE / VALIDACAO | 550 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_ledger_pointer.py` | TESTE / VALIDACAO | 55498 | 6 | from __future__ import annotations |
| `src/tests/test_circuit_manifest.py` | TESTE / VALIDACAO | 4029 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_package.py` | TESTE / VALIDACAO | 6922 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_pipeline.py` | TESTE / VALIDACAO | 9925 | 1 | """Integration tests: scan → stamp → transplant → verify. |
| `src/tests/test_circuit_rollback.py` | TESTE / VALIDACAO | 20684 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_skills.py` | TESTE / VALIDACAO | 9256 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_taps.py` | TESTE / VALIDACAO | 4250 | 1 | from __future__ import annotations |
| `src/tests/test_circuit_transaction.py` | TESTE / VALIDACAO | 18019 | 1 | from __future__ import annotations |
| `src/tests/test_classical_corpus.py` | TESTE / VALIDACAO | 2977 | 1 | import json |
| `src/tests/test_clean_checkout_ci.py` | TESTE / VALIDACAO | 1542 | 1 | from __future__ import annotations |
| `src/tests/test_cognition_active_integration.py` | TESTE / VALIDACAO | 3518 | 1 | """Integration tests for the full three-organ active path.""" |
| `src/tests/test_cognition_config_identity.py` | TESTE / VALIDACAO | 2986 | 1 | from __future__ import annotations |
| `src/tests/test_cognition_contracts.py` | TESTE / VALIDACAO | 1952 | 1 | from __future__ import annotations |
| `src/tests/test_cognition_model_integration.py` | TESTE / VALIDACAO | 3324 | 1 | from __future__ import annotations |
| `src/tests/test_cognition_runtime.py` | TESTE / VALIDACAO | 1328 | 1 | from __future__ import annotations |
| `src/tests/test_cognitive_adapters.py` | TESTE / VALIDACAO | 2078 | 1 | from __future__ import annotations |
| `src/tests/test_cognitive_pulse.py` | TESTE / VALIDACAO | 1464 | 1 | from __future__ import annotations |
| `src/tests/test_corpus_cloud_common.py` | TESTE / VALIDACAO | 501 | 1 | import pytest |
| `src/tests/test_corpus_cloud_ingest.py` | TESTE / VALIDACAO | 2674 | 1 | from f51_darwin.corpus_cloud_ingest import build_cloud_ingest_plan, ingest_remote_pack |
| `src/tests/test_corpus_cloud_token_index.py` | TESTE / VALIDACAO | 2562 | 1 | from f51_darwin.corpus_cloud_token_index import ( |
| `src/tests/test_corpus_cloud_token_verify.py` | TESTE / VALIDACAO | 2310 | 1 | from f51_darwin.corpus_cloud_token_verify import ( |
| `src/tests/test_corpus_cloud_tokenize.py` | TESTE / VALIDACAO | 2569 | 1 | from f51_darwin.corpus_cloud_tokenize import build_cloud_tokenize_plan, tokenize_remote_batch |
| `src/tests/test_corpus_factory_pipeline.py` | TESTE / VALIDACAO | 6119 | — | binario; metadado estrutural apenas |
| `src/tests/test_corpus_pack_upload.py` | TESTE / VALIDACAO | 3002 | 1 | from pathlib import Path |
| `src/tests/test_corpus_policy.py` | TESTE / VALIDACAO | 3243 | 1 | from f51_darwin.corpus_policy import CorpusDecision, evaluate_corpus_source |
| `src/tests/test_curiosity_causal.py` | TESTE / VALIDACAO | 8186 | 1 | from __future__ import annotations |
| `src/tests/test_dae_optimizer.py` | TESTE / VALIDACAO | 2772 | 1 | from __future__ import annotations |
| `src/tests/test_darwin_config.py` | TESTE / VALIDACAO | 2662 | 1 | import pytest |
| `src/tests/test_darwin_forward.py` | TESTE / VALIDACAO | 2471 | 1 | import math |
| `src/tests/test_darwin_inventory.py` | TESTE / VALIDACAO | 585 | 1 | from pathlib import Path |
| `src/tests/test_darwin_x_public_contract.py` | TESTE / VALIDACAO | 4453 | 1 | from __future__ import annotations |
| `src/tests/test_darwin_x_training.py` | TESTE / VALIDACAO | 7293 | 4 | from __future__ import annotations |
| `src/tests/test_darwin_x.py` | TESTE / VALIDACAO | 5609 | 1 | from pathlib import Path |
| `src/tests/test_data_firewall.py` | TESTE / VALIDACAO | 3965 | 1 | from pathlib import Path |
| `src/tests/test_data_loader.py` | TESTE / VALIDACAO | 5021 | 1 | from pathlib import Path |
| `src/tests/test_dataset_layout.py` | TESTE / VALIDACAO | 3410 | 1 | from pathlib import Path |
| `src/tests/test_dataset_states.py` | TESTE / VALIDACAO | 591 | 1 | import pytest |
| `src/tests/test_dense_darwin_model.py` | TESTE / VALIDACAO | 2926 | 1 | from __future__ import annotations |
| `src/tests/test_dense_swiglu.py` | TESTE / VALIDACAO | 3028 | 1 | from __future__ import annotations |
| `src/tests/test_distribution_policy.py` | TESTE / VALIDACAO | 8581 | 28 | from __future__ import annotations |
| `src/tests/test_dual_gpu_split.py` | TESTE / VALIDACAO | 1297 | 1 | from __future__ import annotations |
| `src/tests/test_duplicate_policy.py` | TESTE / VALIDACAO | 4861 | 22 | from __future__ import annotations |
| `src/tests/test_evolution_gate.py` | TESTE / VALIDACAO | 1926 | 1 | from __future__ import annotations |
| `src/tests/test_evolution_score.py` | TESTE / VALIDACAO | 779 | 1 | import pytest |
| `src/tests/test_evolve_darwin_harness.py` | TESTE / VALIDACAO | 1311 | 1 | from __future__ import annotations |
| `src/tests/test_executive.py` | TESTE / VALIDACAO | 5521 | 1 | from __future__ import annotations |
| `src/tests/test_find_gold_checkpoint.py` | TESTE / VALIDACAO | 1243 | 1 | from __future__ import annotations |
| `src/tests/test_full_holdout_eval.py` | TESTE / VALIDACAO | 3772 | 1 | from __future__ import annotations |
| `src/tests/test_gaba_causal.py` | TESTE / VALIDACAO | 11759 | 1 | from __future__ import annotations |
| `src/tests/test_ghost_causal.py` | TESTE / VALIDACAO | 14255 | 1 | from __future__ import annotations |
| `src/tests/test_ghost_recurrence_simulation.py` | TESTE / VALIDACAO | 1794 | 1 | from research.simulate_ghost_recurrence import ( |
| `src/tests/test_gold_harness_production.py` | TESTE / VALIDACAO | 7864 | 18 | from __future__ import annotations |
| `src/tests/test_gold_lineage.py` | TESTE / VALIDACAO | 6366 | — | binario; metadado estrutural apenas |
| `src/tests/test_gradient_mutational_state.py` | TESTE / VALIDACAO | 13849 | 1 | from __future__ import annotations |
| `src/tests/test_isolate_100m_checkpoint.py` | TESTE / VALIDACAO | 2507 | 1 | from __future__ import annotations |
| `src/tests/test_kv_cache.py` | TESTE / VALIDACAO | 3560 | 1 | from __future__ import annotations |
| `src/tests/test_learning_gain_sim.py` | TESTE / VALIDACAO | 13777 | 1 | """Validation and falsification gate tests for learning_gain simulation. |
| `src/tests/test_legacy_layers.py` | TESTE / VALIDACAO | 2341 | 1 | import json |
| `src/tests/test_live_inference_runtime.py` | TESTE / VALIDACAO | 5090 | 1 | from __future__ import annotations |
| `src/tests/test_loss_semantics.py` | TESTE / VALIDACAO | 13704 | 1 | from __future__ import annotations |
| `src/tests/test_math_organ.py` | TESTE / VALIDACAO | 858 | 1 | from __future__ import annotations |
| `src/tests/test_memory_e2e_recall.py` | TESTE / VALIDACAO | 7704 | 1 | """End-to-end test: UniversalMemory teach -> save -> reload -> recall. |
| `src/tests/test_native_layer_transplant.py` | TESTE / VALIDACAO | 2251 | 1 | """Tests for Native Layer Transplant Manager.""" |
| `src/tests/test_native_ttm_diagnostic.py` | TESTE / VALIDACAO | 4816 | 1 | from __future__ import annotations |
| `src/tests/test_native_ttm_recall.py` | TESTE / VALIDACAO | 11374 | 1 | from __future__ import annotations |
| `src/tests/test_neuroendocrine_moe.py` | TESTE / VALIDACAO | 4892 | 1 | import math |
| `src/tests/test_online_learning.py` | TESTE / VALIDACAO | 12147 | 3 | from __future__ import annotations |
| `src/tests/test_operational_surface.py` | TESTE / VALIDACAO | 5949 | 1 | from __future__ import annotations |
| `src/tests/test_organ_causal_bus.py` | TESTE / VALIDACAO | 20535 | 1 | from __future__ import annotations |
| `src/tests/test_organ_causal_qa.py` | TESTE / VALIDACAO | 6532 | 1 | from __future__ import annotations |
| `src/tests/test_organism_causal_runtime.py` | TESTE / VALIDACAO | 26200 | — | binario; metadado estrutural apenas |
| `src/tests/test_physical_hygiene.py` | TESTE / VALIDACAO | 3300 | 5 | from __future__ import annotations |
| `src/tests/test_probe_training_distribution.py` | TESTE / VALIDACAO | 5768 | 1 | from __future__ import annotations |
| `src/tests/test_protected_sleep_evolution.py` | TESTE / VALIDACAO | 29629 | 1 | from __future__ import annotations |
| `src/tests/test_provenance_ledger.py` | TESTE / VALIDACAO | 759 | 1 | from pathlib import Path |
| `src/tests/test_realtime_ingest_flow.py` | TESTE / VALIDACAO | 8284 | 1 | from __future__ import annotations |
| `src/tests/test_replay_buffer.py` | TESTE / VALIDACAO | 3225 | 1 | from __future__ import annotations |
| `src/tests/test_repository_governance.py` | TESTE / VALIDACAO | 2244 | 1 | import subprocess |
| `src/tests/test_rope_llama.py` | TESTE / VALIDACAO | 707 | 1 | from __future__ import annotations |
| `src/tests/test_runtime_data_migration.py` | TESTE / VALIDACAO | 1629 | 4 | from __future__ import annotations |
| `src/tests/test_runtime_modules.py` | TESTE / VALIDACAO | 3436 | 1 | from pathlib import Path |
| `src/tests/test_semantic_probe.py` | TESTE / VALIDACAO | 5412 | 1 | from __future__ import annotations |
| `src/tests/test_serve_davi.py` | TESTE / VALIDACAO | 10547 | 1 | from __future__ import annotations |
| `src/tests/test_single_root_migration.py` | TESTE / VALIDACAO | 9732 | 1 | from __future__ import annotations |
| `src/tests/test_smol_dense_assembly.py` | TESTE / VALIDACAO | 1838 | 1 | from __future__ import annotations |
| `src/tests/test_smol_dense_brain_config.py` | TESTE / VALIDACAO | 1495 | 1 | from __future__ import annotations |
| `src/tests/test_smol_dense_qa_benchmark.py` | TESTE / VALIDACAO | 1267 | 1 | from __future__ import annotations |
| `src/tests/test_smol_dense_runtime.py` | TESTE / VALIDACAO | 1297 | 1 | from __future__ import annotations |
| `src/tests/test_smol_exact_assembly.py` | TESTE / VALIDACAO | 1328 | 1 | from __future__ import annotations |
| `src/tests/test_smol_exact_brain_config.py` | TESTE / VALIDACAO | 2837 | 1 | from __future__ import annotations |
| `src/tests/test_smol_transplant_launcher.py` | TESTE / VALIDACAO | 1118 | 1 | from __future__ import annotations |
| `src/tests/test_soul_desire_runtime.py` | TESTE / VALIDACAO | 971 | 1 | from __future__ import annotations |
| `src/tests/test_ssm_core.py` | TESTE / VALIDACAO | 2127 | 1 | import torch |
| `src/tests/test_start_100m_auto_launcher.py` | TESTE / VALIDACAO | 3352 | 1 | from __future__ import annotations |
| `src/tests/test_state_identity.py` | TESTE / VALIDACAO | 2681 | 1 | from __future__ import annotations |
| `src/tests/test_supply_chain_controls.py` | TESTE / VALIDACAO | 13570 | 15 | from __future__ import annotations |
| `src/tests/test_supported_entrypoint_smoke.py` | TESTE / VALIDACAO | 1341 | 1 | from __future__ import annotations |
| `src/tests/test_temporal_organ_frames.py` | TESTE / VALIDACAO | 4682 | 1 | from dataclasses import replace |
| `src/tests/test_three_organ_shadow_probe.py` | TESTE / VALIDACAO | 1380 | 1 | from __future__ import annotations |
| `src/tests/test_tokenizer_interface.py` | TESTE / VALIDACAO | 1445 | 1 | from pathlib import Path |
| `src/tests/test_topology_manifest_v7.py` | TESTE / VALIDACAO | 8621 | 1 | from __future__ import annotations |
| `src/tests/test_training_data_contract.py` | TESTE / VALIDACAO | 2168 | 1 | from __future__ import annotations |
| `src/tests/test_training_observability.py` | TESTE / VALIDACAO | 19603 | 1 | from __future__ import annotations |
| `src/tests/test_training_token_budget.py` | TESTE / VALIDACAO | 1690 | 1 | from __future__ import annotations |
| `src/tests/test_transfer_donor.py` | TESTE / VALIDACAO | 1246 | 1 | from __future__ import annotations |
| `src/tests/test_transfer_experience.py` | TESTE / VALIDACAO | 3395 | 1 | from pathlib import Path |
| `src/tests/test_transfer_launcher.py` | TESTE / VALIDACAO | 1010 | 1 | from pathlib import Path |
| `src/tests/test_transfer_ngram.py` | TESTE / VALIDACAO | 3428 | 1 | from __future__ import annotations |
| `src/tests/test_transfer_schedule.py` | TESTE / VALIDACAO | 2999 | 1 | from f51_darwin.transfer.schedule import ( |
| `src/tests/test_transfer_ssd_mixer.py` | TESTE / VALIDACAO | 1940 | 1 | from __future__ import annotations |
| `src/tests/test_transfer_student.py` | TESTE / VALIDACAO | 2945 | 1 | from pathlib import Path |
| `src/tests/test_transfer_trainer.py` | TESTE / VALIDACAO | 4167 | 1 | from pathlib import Path |
| `src/tests/test_transplant_16b_checkpoint.py` | TESTE / VALIDACAO | 2641 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_cli.py` | TESTE / VALIDACAO | 3480 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_config.py` | TESTE / VALIDACAO | 1328 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_donor_contract.py` | TESTE / VALIDACAO | 5251 | 1 | """Regressoes do contrato de doador do transplante 1.6B. |
| `src/tests/test_transplant_16b_layers.py` | TESTE / VALIDACAO | 1079 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_ledger.py` | TESTE / VALIDACAO | 2106 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_moe.py` | TESTE / VALIDACAO | 5462 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_organs.py` | TESTE / VALIDACAO | 1608 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_projection.py` | TESTE / VALIDACAO | 1470 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_selection.py` | TESTE / VALIDACAO | 4537 | 1 | """Invariantes da reducao por selecao. |
| `src/tests/test_transplant_16b_sources.py` | TESTE / VALIDACAO | 1301 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_ssd_fit.py` | TESTE / VALIDACAO | 1251 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_tokenizer.py` | TESTE / VALIDACAO | 2037 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_16b_verification.py` | TESTE / VALIDACAO | 2156 | 2 | from __future__ import annotations |
| `src/tests/test_transplant_bundle.py` | TESTE / VALIDACAO | 5990 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_checkpoint.py` | TESTE / VALIDACAO | 5163 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_launcher.py` | TESTE / VALIDACAO | 638 | 1 | from pathlib import Path |
| `src/tests/test_transplant_ledger.py` | TESTE / VALIDACAO | 4448 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_lifecycle.py` | TESTE / VALIDACAO | 4938 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_organs.py` | TESTE / VALIDACAO | 7756 | 1 | from __future__ import annotations |
| `src/tests/test_transplant_recipient.py` | TESTE / VALIDACAO | 11395 | 1 | from __future__ import annotations |
| `src/tests/test_universal_memory.py` | TESTE / VALIDACAO | 7952 | 1 | from __future__ import annotations |
| `src/tests/test_workspace_boundary.py` | TESTE / VALIDACAO | 672 | 1 | from pathlib import Path |
| `src/tests/test_world_model.py` | TESTE / VALIDACAO | 6257 | 1 | from __future__ import annotations |
| `THIRD_PARTY_NOTICES.md` | GOVERNANCA / RAIZ | 2014 | 1 | # Third-party dependency notices |
| `src/tools/adapt_checkpoint_darwin_x.py` | FERRAMENTA / AUDITORIA | 10161 | 1 | #!/usr/bin/env python3 |
| `src/tools/audit_dataset_candidates.py` | FERRAMENTA / AUDITORIA | 2335 | 1 | from __future__ import annotations |
| `src/tools/auto_ingest.py` | FERRAMENTA / AUDITORIA | 13466 | 14 | #!/usr/bin/env python3 |
| `src/tools/auto_pipeline.py` | FERRAMENTA / AUDITORIA | 4338 | 3 | #!/usr/bin/env python3 |
| `src/tools/balance_corpus.py` | FERRAMENTA / AUDITORIA | 41529 | 20 | #!/usr/bin/env python |
| `src/tools/build_corpus_pack.py` | FERRAMENTA / AUDITORIA | 1829 | 1 | from __future__ import annotations |
| `src/tools/build_distribution.py` | FERRAMENTA / AUDITORIA | 6958 | 6 | #!/usr/bin/env python3 |
| `src/tools/check_architecture_boundaries.py` | FERRAMENTA / AUDITORIA | 16150 | 1 | #!/usr/bin/env python3 |
| `src/tools/check_canonical_docs.py` | FERRAMENTA / AUDITORIA | 2928 | 3 | #!/usr/bin/env python3 |
| `src/tools/check_ckpt.py` | FERRAMENTA / AUDITORIA | 844 | 1 | import sys |
| `src/tools/check_dependency_policy.py` | FERRAMENTA / AUDITORIA | 19610 | — | binario; metadado estrutural apenas |
| `src/tools/check_distribution.py` | FERRAMENTA / AUDITORIA | 15044 | 1 | #!/usr/bin/env python3 |
| `src/tools/check_docs_links.py` | FERRAMENTA / AUDITORIA | 4092 | 5 | #!/usr/bin/env python3 |
| `src/tools/check_duplicates.py` | FERRAMENTA / AUDITORIA | 5707 | 4 | #!/usr/bin/env python3 |
| `src/tools/check_nitro_runtime.py` | FERRAMENTA / AUDITORIA | 6035 | 1 | #!/usr/bin/env python3 |
| `src/tools/check_nucleo_ready.ps1` | FERRAMENTA / AUDITORIA | 6254 | 2 | param( |
| `src/tools/check_operational_surface.py` | FERRAMENTA / AUDITORIA | 4830 | 1 | #!/usr/bin/env python3 |
| `src/tools/check_physical_hygiene.py` | FERRAMENTA / AUDITORIA | 5840 | 1 | #!/usr/bin/env python3 |
| `src/tools/check_python_compilation.py` | FERRAMENTA / AUDITORIA | 1364 | 1 | #!/usr/bin/env python3 |
| `src/tools/check_source_secrets.py` | FERRAMENTA / AUDITORIA | 6622 | 3 | #!/usr/bin/env python3 |
| `src/tools/commit_loop_fechado.ps1` | FERRAMENTA / AUDITORIA | 2358 | 1 | # Script de commit — Loop Fechado F51-Darwin-X |
| `src/tools/compare_checkpoints.py` | FERRAMENTA / AUDITORIA | 2279 | 2 | #!/usr/bin/env python3 |
| `src/tools/consolidate_corpus.py` | FERRAMENTA / AUDITORIA | 4727 | 7 | #!/usr/bin/env python |
| `src/tools/convert_legacy_checkpoint.py` | FERRAMENTA / AUDITORIA | 11873 | 3 | #!/usr/bin/env python |
| `src/tools/create_admin_shortcut.ps1` | FERRAMENTA / AUDITORIA | 624 | 1 | $desktopPath = [Environment]::GetFolderPath("Desktop") |
| `src/tools/eval_checkpoint.py` | FERRAMENTA / AUDITORIA | 13938 | 1 | #!/usr/bin/env python3 |
| `src/tools/evaluate_darwin.py` | FERRAMENTA / AUDITORIA | 1124 | 1 | from __future__ import annotations |
| `src/tools/find_gold_checkpoint.py` | FERRAMENTA / AUDITORIA | 4533 | 1 | #!/usr/bin/env python3 |
| `src/tools/find_gold_in_vss.ps1` | FERRAMENTA / AUDITORIA | 1014 | 1 | #Requires -RunAsAdministrator |
| `src/tools/gradient_diagnostics.py` | FERRAMENTA / AUDITORIA | 7990 | 8 | """Gradient diagnostics for Darwin-X 100M — run a few steps with gradient hooks.""" |
| `src/tools/inspect_checkpoint.py` | FERRAMENTA / AUDITORIA | 2911 | 11 | """Inspect a training checkpoint.""" |
| `src/tools/inspect_darwin_x.py` | FERRAMENTA / AUDITORIA | 2991 | 1 | #!/usr/bin/env python3 |
| `src/tools/inspect_seed.py` | FERRAMENTA / AUDITORIA | 3838 | 1 | from __future__ import annotations |
| `src/tools/install_nucleo_dataset_watchdog_startup.ps1` | FERRAMENTA / AUDITORIA | 1172 | 1 | param( |
| `src/tools/install_nucleo_dataset_watchdog_task.ps1` | FERRAMENTA / AUDITORIA | 1943 | 1 | param( |
| `src/tools/isolate_100m_checkpoint.py` | FERRAMENTA / AUDITORIA | 4492 | 1 | #!/usr/bin/env python3 |
| `src/tools/measure_tokenizer_metrics.py` | FERRAMENTA / AUDITORIA | 8902 | 11 | #!/usr/bin/env python3 |
| `src/tools/migrate_repo_runtime_data.py` | FERRAMENTA / AUDITORIA | 4547 | 3 | #!/usr/bin/env python3 |
| `src/tools/migrate_single_root_workspace.ps1` | FERRAMENTA / AUDITORIA | 986 | 1 | [CmdletBinding(DefaultParameterSetName = 'Plan')] |
| `src/tools/nucleo_dataset_organism.py` | FERRAMENTA / AUDITORIA | 22695 | 7 | from __future__ import annotations |
| `src/tools/promote_approved_data.py` | FERRAMENTA / AUDITORIA | 2192 | 1 | from __future__ import annotations |
| `src/tools/README.md` | FERRAMENTA / AUDITORIA | 316 | 1 | # Maintenance tools |
| `src/tools/realtime_ingest.py` | FERRAMENTA / AUDITORIA | 14902 | 24 | #!/usr/bin/env python3 |
| `src/tools/run_causal_ablation.py` | FERRAMENTA / AUDITORIA | 27920 | 2 | """Run a paired, local-only causal ablation on a deterministic tiny model. |
| `src/tools/run_gold_source_audit.ps1` | FERRAMENTA / AUDITORIA | 10555 | 1 | [CmdletBinding()] |
| `src/tools/start_nucleo_dataset_organism.ps1` | FERRAMENTA / AUDITORIA | 2273 | 1 | param( |
| `src/tools/start_nucleo_dataset_watchdog.ps1` | FERRAMENTA / AUDITORIA | 1734 | 1 | param( |
| `src/tools/start_nucleo_ready_watcher.ps1` | FERRAMENTA / AUDITORIA | 1616 | 1 | param( |
| `src/tools/status_nucleo_dataset_organism.ps1` | FERRAMENTA / AUDITORIA | 318 | 1 | param( |
| `src/tools/verify_corpus_pack.py` | FERRAMENTA / AUDITORIA | 764 | 1 | from __future__ import annotations |
| `src/tools/verify_gold_lineage.py` | FERRAMENTA / AUDITORIA | 9264 | 1 | #!/usr/bin/env python3 |
| `src/tools/watch_nucleo_ready.ps1` | FERRAMENTA / AUDITORIA | 1439 | 1 | param( |
| `src/tools/watchdog_nucleo_dataset_organism.ps1` | FERRAMENTA / AUDITORIA | 2404 | 1 | param( |

## Estado de cobertura

Os 859 caminhos rastreados estão na tabela principal. O arquivo
`src/tests/test_provenance_certificate.py` está TRACKED em `df9f250`; sua presença
e seus testes não equivalem a uma descoberta científica validada.
