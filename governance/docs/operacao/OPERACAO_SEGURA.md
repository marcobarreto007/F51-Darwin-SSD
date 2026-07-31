# Operacao segura

## Preflight obrigatorio

Na raiz do repositorio, configure o namespace Python antes dos comandos de
modulo:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

1. `git status --short` deve estar limpo.
2. Confirme ausencia de processo Darwin concorrente com `Get-CimInstance
   Win32_Process`.
3. Rode `nvidia-smi` e confirme as duas GPUs locais.
4. Rode `.\.venv_nitro\Scripts\python.exe -m scripts.darwin_inventory`.
5. Verifique o checkpoint ativo com
   `.\.venv_nitro\Scripts\python.exe -m scripts.inspect_organism_checkpoint
    workspace\03_CHECKPOINTS_100M_FULL_V9\organism_cycle_003.pt
    --config src\\configs\\darwin_x_100m.yaml`.
   (O gold 1.6B-Nitro cycle_071 nao esta acessivel localmente.)
6. Rode a auditoria em `src/tools/run_gold_source_audit.ps1`.
7. Confirme que readiness, pointer, config, corpus feast_v2 e source commit
   possuem a mesma linhagem.

## Gate oficial e launch

O gate oficial e local-only e nao cria processo de treino:

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_overnight_16b.ps1 -Canary
```

`-Canary -Launch` e permitido apenas se todos os source gates estiverem verdes,
o worktree estiver limpo, nao houver processo concorrente, houver espaco em
disco e a readiness estiver vinculada ao HEAD. O canary deve usar raiz isolada,
ser finito e nunca alterar `organism_latest.json`.

`run247` nao faz parte deste procedimento. Nao confunda loss finita com melhora
de qualidade.

Para a linhagem 100M FULL V9, o gate e:

```powershell
powershell -ExecutionPolicy Bypass -File src\\scripts\\start_100m_65b.ps1 -Canary
```

O `-Launch` inicia um supervisor oculto vinculado ao HEAD limpo. Ele executa
canarios CONTROL, SHADOW e ENFORCE antes de iniciar o comando finito
`train-budget` com meta de 65.000.000.000 tokens. Falha em qualquer canario
impede o treino longo.

## Ingestao

Use `.\.venv_nitro\Scripts\python.exe -m scripts.ingest_pipeline`. O fluxo e
pesquisar, colocar em quarentena, obter aprovacao explicita e consolidar.
Nenhum dado externo vai diretamente ao corpus canonico ou aos pesos.

## Rollback da raiz

Antes do rollback, pare processos que usem `workspace/`, verifique o manifest e
execute apenas:

```powershell
powershell -ExecutionPolicy Bypass -File src\\tools\\migrate_single_root_workspace.ps1 `
  -ProjectRoot . -Rollback `
  -Manifest workspace\04_MANIFESTOS\single_root_migration.json
```

O rollback e uma renomeacao no mesmo volume. Nao copie os 240 GB, nao remova o
workspace manualmente e nao execute rollback se o destino estiver ocupado ou
conflitante. O manifest e os hashes sao a recovery path.

## Proibicoes

- nao iniciar treino concorrente;
- nao promover checkpoint por mtime, nome ou cycle;
- nao apagar artefato unico;
- nao gravar dataset/checkpoint no Git;
- nao usar cloud como fonte de verdade;
- nao reescrever historico, executar prune/gc ou mascarar segredo;
- nao declarar GOLD com gate faltante, vermelho ou evidencia de outro commit.
