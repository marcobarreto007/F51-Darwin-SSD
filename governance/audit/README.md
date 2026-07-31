# Indice de auditoria

## Escopo

A auditoria cobre o source local-only, a unica raiz `workspace/`, o corpus
feast_v2, `organism_cycle_071.pt` (cycle 71, step 40751), supply chain,
seguranca, duplicidade e reproducao. Artefatos grandes ficam ignorados pelo Git
e sao representados por manifests e SHA-256.

## Politicas e proveniencia

- `governance/audit/policy/repository-governance.md` - governanca Git e runtime;
- `governance/audit/policy/operational-surface.json` - seis entrypoints suportados;
- `governance/audit/policy/duplicates.json` - conteudo e autoridade sem duplicacao;
- `governance/audit/policy/dependencies.json` - CPU-audit e Nitro-CUDA separados;
- `governance/audit/policy/distribution.json` - build imutavel, offline e smoke em venv;
- `governance/audit/policy/physical-root.json` - allowlist e higiene da unica raiz;
- `governance/audit/provenance/single-root-migration-summary.json` - migracao e rollback;
- `governance/audit/provenance/gold-lineage-summary.json` - corpus/checkpoint verificados;
- `governance/audit/evidence/gitleaks-source-status.json` - arvore limpa e bloqueio
  historico.
- `governance/audit/provenance/source-manifest.json` - manifest do source congelado;
- `governance/audit/provenance/evidence-attestation.json` - atestado posterior que registra
  `gold_source_commit` e `gold_evidence_commit`;
- `governance/audit/test-reports/gold-source-f54a579.json` - relatorio integral redigido,
  schema-valid e commit-bound dos 12 gates;
- `governance/audit/GOLD_REPORT_2026-07-17.md` - simulacao externa e veredito requisito a
  requisito.

## Reproducao CPU-safe

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m tools.check_docs_links --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_canonical_docs --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_duplicates --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_dependency_policy --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_operational_surface --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_physical_hygiene --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_architecture_boundaries --root .
.\.venv_nitro\Scripts\python.exe -m tools.check_distribution --root .
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider
powershell -ExecutionPolicy Bypass -File src\\tools\\run_gold_source_audit.ps1
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

O source audit possui 12 gates e publica JSON atomico validado por schema e
semantica. A hierarquia de prova exige que source commit, report commit e
observed end commit coincidam.

O source e a evidencia sao commits separados. Resolva o evidence commit mais
recente com `git log -1 --format=%H -- governance/audit/provenance/source-manifest.json`.
Hashes de evidencia usam bytes do Git blob e registram o blob OID; reproduza-os
com `git cat-file blob <evidence-commit>:<path>` para evitar deriva LF/CRLF.

No source `f54a579802e8e56b819eba37ce81b69c444fd781`, 11 gates passaram
e somente `secret_scan` falhou. Foram coletados 384 testes, o pytest retornou 0
e 250/250 arquivos Python ativos compilaram.

## Veredito de seguranca

A arvore rastreada atual teve zero achados. O historico Git alcancavel possui
nove achados redigidos, sem revogacao comprovada e sem history rewrite. Isso e
um bloqueio externo: o estado permanece **NOT GOLD**. Nao ha allowlist para
credenciais reais e o canary com launch permanece proibido enquanto o source
audit falhar.

Quando o escopo solicitado exclui credenciais, o relatório deve apresentar o
`secret_scan` vermelho sem ocultá-lo e avaliar separadamente os outros 14
gates. “Organizacionalmente GOLD” significa 14/14 nesses controles; nunca
significa que o sistema recebeu aprovação integral de segurança.

O rollback da raiz e reproduzido somente pelo comando de
`governance/docs/operacao/OPERACAO_SEGURA.md`.
