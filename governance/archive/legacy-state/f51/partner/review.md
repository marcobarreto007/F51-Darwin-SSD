# Deep Review — Training Pipeline Patches (R1–R6)

Arquivos: `scripts/darwin_organism.py`, `scripts/start_overnight_25b.ps1`, `tests/test_organism_causal_runtime.py`
Veredito: **APROVADO COM RISCO** — aplicar o fix da race do `heartbeat_state` (R4) antes de commitar em producao. Os demais achados sao opcionais.

## Achado BLOCKER — R4 race em `heartbeat_state` (async default)

`async_checkpoint` agora default `True`. A thread de background executa `torch.save(payload)` enquanto o proximo ciclo ja esta treinando. O payload tem dois campos ja isolados em CPU (`model_state` via `.cpu()`, `optimizer_state_dict` via `_optimizer_state_to_cpu`). Mas **`heartbeat_state` nao foi isolado**:

- `Heartbeat.state_dict()` (`f51_darwin/heartbeat.py:335`) retorna `ff_stack_state_dict` e `tt_memory_state_dict`, que sao `nn.Module.state_dict()` = **referencias a tensores GPU vivos**.
- `ForwardForwardLayer.forward` muta `self.weight.data += ...` em cada passo (`heartbeat.py:35`). Logo, a thread le esses tensores enquanto o treino os escreve.
- Resultado: snapshot tornado do heartbeat no checkpoint e/ou erro de CUDA (acesso concorrente ao mesmo tensor por duas threads).

**Fix exigido:** deep-copy `heartbeat_state` para CPU antes de iniciar a thread, espelhando `_optimizer_state_to_cpu`. Substituir no payload:
```python
"heartbeat_state": _tensors_to_cpu(heartbeat_state) if heartbeat_state is not None else None,
```
Aplicar tambem na construcao de `heartbeat_state` (ja que `model.heartbeat_state_dict()` aliasa tensores).

## Achados menores (nao bloqueantes)

1. **`model_state` nao clona em CPU** (pre-existente): `{k: v.cpu() ...}` retorna o mesmo tensor se o modelo ja esta em CPU. No cenario alvo (GPU) `.cpu()` copia, entao e seguro; mas para robustez usar `.detach().cpu().clone()`. Nao introduzido por este patch.

2. **Warmup/accum contam por micro-batch**: `_apply_lr_schedule(self.total_steps)` avanca por micro-batch, nao por optimizer step. Com `accum_steps>1` o warmup termina "mais cedo" em optimizer-steps. Intencional e aceitavel; registrar no CLAUDE.md se vira padrao.

3. **Windows SIGTERM**: `signal.signal(SIGTERM)` no Windows nao dispara em `TerminateProcess`. SIGINT (Ctrl+C) funciona. O fallback `except KeyboardInterrupt` cobre. Documentado no docstring.

4. **RuntimeError de save async nao tratado no run247**: se `_join_pending_save` re-raises (ex.: disco cheio durante write), a excecao escapa de `run_cycle`, nao e pega pelo `except KeyboardInterrupt` e derruba o run247 sem shutdown gracioso. Mesmo padrao do codigo original (torch.save podia falhar); o gate de disco pre-save mitiga. Opcional: envolver `run_cycle` em try/except no run247.

## Achados positivos (sem bug)

- **R1 `remap_optimizer_state`**: chaveamento por `(name, shape)` e correto. `old_optimizer.state.get(param)` em defaultdict nao dispara a factory (`.get` nunca cria). Preserva `exp_avg`/`exp_avg_sq`/`step` via clone; neurogenese fica lazy; prune descarta; expansao (shape mudou) vira fresh. Test cobre os 3 casos. OK.
- **R2 warmup**: formula `base*(step+1)/warmup` alcancando `base` no ultimo step; no-op quando `warmup_steps<=0`. Re-aplicado apos rebuild R1. OK.
- **R3 accum**: `(loss/N).backward()` + step na fronteira; `accum_steps=1` reproduz o legado. Rejeicao brainstem/micro-batch ruim reseta a janela (seguro). OK.
- **R4 `_join_pending_save`**: ordenacao de memoria garantida por `thread.join()` antes de ler `_save_error` (happens-before). `_save_thread` so e tocado na main thread. Sem deadlock/livelock: join so bloqueia se a thread anterior ainda vive, e toda `_save_cycle` comeca por join. OK.
- **R5 handler cooperativo**: so seta flag (async-signal-safe); loop cheeca por step. Nao reentra. OK.
- **R6 brainstem**: guarda com `getattr(self,"brainstem",None)` compativel com mocks `__new__`. OK.
- **Testes**: 3 novos cobrem R1 (preserved/dropped/initialised) e R2 (rampa + plateau + no-op). 178/178 passam. OK.
- **Preserva alteracoes de outros agentes**: diff toca apenas os 3 arquivos do Bob.

## Conclusao
Aplique o fix do `heartbeat_state` (isolamento em CPU) e pode commitar. O resto e polimento opcional.
