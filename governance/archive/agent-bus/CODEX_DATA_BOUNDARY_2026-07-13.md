# F51 data boundary cutover — 2026-07-13

## Objetivo entregue

Manter codigo no `F51-Darwin-SSD` e todo o ciclo operacional de dados em
`C:\Users\marco\Desktop\F51-Dataset-Organizado`, sem reconstruir o corpus
principal e sem reiniciar o treino vivo.

## Estado fisico

- Corpus principal preservado:
  `01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast.bin`
  (`72,567,226,368` bytes; `18,141,806,592` tokens int32).
- `F51-Darwin-SSD/data` agora e uma junction para
  `F51-Dataset-Organizado/02_CORPUS`.
- `F51-Darwin-SSD/checkpoints` agora e uma junction para
  `F51-Dataset-Organizado/03_CHECKPOINTS`; o caminho estava ausente e nenhum
  checkpoint repo-local precisou ser movido.
- Dados repo-local foram mesclados sem sobrescrever conflitos:
  `730` arquivos copiados, `1,437` linhas JSONL incorporadas na primeira
  passagem; a segunda passagem retornou `730` identicos e `0` linhas pendentes.
- Backup reversivel preservado em
  `02_CORPUS/_MIGRATION_BACKUPS/repo_data_20260713_115306`.
- `gutenberg_117.txt` (318 bytes, metadado de arquivos MIDI, sem prosa) foi
  preservado em `00_BRUTOS/_quarantine/non_text_gutenberg/`; o corpus classico
  selado de 240 arquivos e seu token bin nao foram reconstruidos.

## Contrato de codigo

- `DataFactoryPaths.from_project()` resolve obrigatoriamente o workspace externo.
- `run247`, Ghost Stream, Ghost Feeder, ingestao, auditoria e promocao usam a
  fronteira externa.
- Resolucao padrao de tokens escolhe o feast externo; fallback repo-local exige
  opt-in explicito e permanece somente para fixtures/cloud staging isolado.
- A migracao idempotente e auditavel esta em
  `scripts/migrate_repo_runtime_data.py`.

## Validacao

- Suite completa CPU-safe:
  `.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider`
- Resultado final: `194 passed`.
- Processo `run247` permaneceu vivo desde `2026-07-13 05:27`.
- O ledger externo recebeu nova escrita as `11:58`, depois do corte; o backup
  repo-local original permaneceu congelado em `11:42`.
- Checkpoint observado depois do corte: `organism_cycle_064.pt`, cycle 64,
  step 32000, salvo externamente.
- GPUs observadas: RTX 5060 Ti `12,782 MiB / 16,311 MiB`; RTX 3060
  `10,569 MiB / 12,288 MiB`.
- `darwin_inventory.py` agora resolve o alvo relativo de
  `03_CHECKPOINTS/organism_latest.json` ao lado do ponteiro; status final `ok`
  e `target_exists: true`.

## Verdade que permanece

O processo vivo usa `configs/darwin_x_1.6b_nitro.yaml`. Isso nao prova nem
substitui a linhagem 2.5B desejada.
