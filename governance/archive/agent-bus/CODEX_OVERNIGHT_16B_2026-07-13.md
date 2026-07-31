# Codex - Overnight 1.6B canonico - 2026-07-13

## Decisao de Marco

- usar sempre o `F51-Darwin-X-1.6B-Nitro` no hardware local;
- deixar a classe 2.5B para um futuro distante na nuvem.

## Linhagem entregue

- origem: `organism_cycle_065.pt`, v6, cycle 65, step 32500;
- migrado: `organism_cycle_066.pt`, v7, cycle 66, step 32501;
- parametros reais: 1.764.019.648;
- checkpoint v7: 10.876.174.001 bytes;
- base id v7: `darwin-model-core-v1:165c056485235c8417c1b328c74ffa9bfd587547d96481ceaa6991d612fac3c9`;
- config, shapes e identidade integral aprovados;
- Topology Manifest reaberto com 0 reparos;
- AdamW reaberto com momentum;
- corpus externo: 18.141.806.592 tokens int32.

## Migracao v6 para v7

O v6 tinha config exata, mas antecedia os estados neuroendocrinos persistentes. A migracao foi fechada por allowlist: 256 chaves novas conhecidas, nenhuma chave inesperada e nenhuma divergencia de shape. Foram restaurados 981 estados AdamW; somente 16 `baseline_dopamine` novos comecaram sem momentum. O checkpoint v7 embute `checkpoint_migration`.

## Correcoes operacionais

- `heartbeat_enabled` restaurado para `true` na config canonica;
- hash de identidade aceita scalar BF16 sem alterar bytes dos tensores nao escalares;
- resume v6 nao finge validar um hash produzido por schema historico;
- resume v7 continua estrito;
- todos os ciclos completos publicam checkpoint;
- rotacao destrutiva removida;
- gate de espaco ocorre antes do save e encerra antes de esgotar disco;
- LegacyLayers usa temp unico e retry para lock transiente do Windows;
- launcher oficial criado em `scripts/start_overnight_16b.ps1`.

## Runtime iniciado

- launch: 2026-07-13 23:56 EDT;
- PID launcher: 26520;
- run dir: `runs/overnight_16b/20260713_235641`;
- ciclo: 67, 2.000 steps;
- primeiro step: 32502;
- loss 3.449, LM 2.564, PPL 13.0, 17 tok/s;
- VRAM observada: 12.180 MiB GPU0 e 9.929 MiB GPU1;
- checkpoints preservados;
- reserva de disco no launch: 186,88 GiB.

## Prova

- suite CPU completa aprovada;
- smoke CUDA e save v7 aprovados;
- reabertura estrita do v7 aprovada;
- stderr do run noturno vazio no primeiro step.
