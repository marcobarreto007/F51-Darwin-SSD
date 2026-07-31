# Native TTM Channel Surgery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o TTM do Darwin-Smol recuperar associações alinhadas, injetar um residual numericamente limitado em `T-1` e repetir o gate causal sem treinar o cérebro.

**Architecture:** `TestTimeMemory` ganha escrita explícita com fontes separadas de chave e valor. O modelo consulta top-1 usando o hidden bruto da última posição e injeta somente nessa posição após limitar a norma; o benchmark grava a chave do prompt e o valor do trecho de resposta observado.

**Tech Stack:** Python 3.12, PyTorch, pytest, checkpoint Darwin-Smol Native Dense V1 em BF16, duas GPUs CUDA locais.

## Global Constraints

- Nenhum parâmetro ou shape novo no checkpoint.
- Nenhum token/texto da resposta armazenado; somente tensores hidden.
- Nenhum backward ou alteração do cérebro.
- Dose máxima `0.30`; residual limitado à norma do hidden de referência.
- O protocolo causal, seus controles e thresholds não podem ser relaxados.
- Preservar as mudanças JEPA não commitadas em `config.py` e `model.py`; commits TTM não podem incluí-las.
- O checkpoint publicado nunca é reescrito.

---

### Task 1: Associação explícita e bounding numérico

**Files:**
- Modify: `src/f51_darwin/heartbeat.py:67-184`
- Modify: `src/tests/test_causal_cognitive_organs.py`

**Interfaces:**
- Consumes: `SurpriseMemorySlot`, `TestTimeMemory.proj_key`, tensors `[B, D]`.
- Produces: `TestTimeMemory.write_association(key_source, value_source, *, jepa_error, domain) -> bool` e `bound_memory_residual(value, reference) -> Tensor`.

- [ ] **Step 1: Escrever testes vermelhos para fontes separadas**

Adicionar testes que zeram/configuram `proj_key` de forma determinística, usam
`key_source != value_source` e verificam:

```python
wrote = memory.write_association(
    key_source,
    value_source,
    jepa_error=1.0,
    domain="fact-a",
)
assert wrote
assert torch.allclose(memory.slots[0].key, memory.proj_key(key_source)[0])
assert torch.allclose(memory.slots[0].value, value_source[0])
```

Também rejeitar shapes diferentes de `[B, D]`, batch diferente de 1 e tensor
não finito.

- [ ] **Step 2: Executar testes e confirmar RED**

```powershell
python -m pytest src/tests/test_causal_cognitive_organs.py -k "write_association or bound_memory" -q
```

Esperado: falha porque as interfaces ainda não existem.

- [ ] **Step 3: Implementar escrita mínima**

Em `TestTimeMemory`:

```python
def write_association(
    self,
    key_source,
    value_source,
    *,
    jepa_error=0.0,
    domain="",
):
    if jepa_error < self.surprise_threshold:
        return False
    if key_source.ndim != 2 or value_source.ndim != 2:
        raise ValueError("association sources must have shape [batch, d_model]")
    if key_source.shape != value_source.shape or key_source.shape != (1, self.d_model):
        raise ValueError("association sources must both have shape [1, d_model]")
    if not torch.isfinite(key_source).all() or not torch.isfinite(value_source).all():
        raise ValueError("association sources must be finite")
    with torch.no_grad():
        key = self.proj_key(key_source)[0]
        value = value_source[0]
    self._append_slot(key, value, jepa_error=jepa_error, domain=domain)
    return True
```

Extrair a lógica de capacidade/append hoje duplicada em `write_if_surprised`
para `_append_slot`, sem mudar o comportamento legado.

- [ ] **Step 4: Implementar bounding puro**

No mesmo módulo:

```python
def bound_memory_residual(value, reference, eps=1e-6):
    if value.shape != reference.shape:
        raise ValueError("memory value and reference must share shape")
    if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
        raise ValueError("memory value and reference must be finite")
    value_norm = value.float().norm(dim=-1, keepdim=True)
    reference_norm = reference.float().norm(dim=-1, keepdim=True)
    factor = (reference_norm / value_norm.clamp_min(eps)).clamp(max=1.0)
    return value * factor.to(dtype=value.dtype)
```

- [ ] **Step 5: Executar testes verdes e regressões Heartbeat**

```powershell
python -m pytest src/tests/test_causal_cognitive_organs.py src/tests/test_live_inference_runtime.py -q
```

Esperado: todos passam.

- [ ] **Step 6: Commit isolado**

```powershell
git add -- src/f51_darwin/heartbeat.py src/tests/test_causal_cognitive_organs.py
git commit -m "feat(memory): add aligned TTM associations"
```

---

### Task 2: Readout top-1 limitado em T-1

**Files:**
- Modify: `src/f51_darwin/darwin_x_core/model.py:747-792`
- Modify: `src/tests/test_causal_cognitive_organs.py`

**Interfaces:**
- Consumes: `bound_memory_residual`, `TestTimeMemory.retrieve(..., top_k=1, already_pooled=True)`.
- Produces: leitura nativa em `hidden[:, -1, :]`, injeção somente em `hidden_for_logits[:, -1, :]`.

- [ ] **Step 1: Escrever teste vermelho do caminho T-1**

Construir modelo pequeno causal, monkeypatchar `retrieve` para registrar query e
retornar valor finito. Verificar:

```python
assert recorded_query.shape == (batch, d_model)
assert torch.equal(recorded_query, raw_hidden[:, -1, :])
assert output.ttm_memory_retrieved is True
assert output.logits[:, -1].ne(control.logits[:, -1]).any()
assert torch.equal(output.logits[:, :-1], control.logits[:, :-1])
```

O braço gate zero deve manter logits bit a bit.

- [ ] **Step 2: Confirmar RED**

```powershell
python -m pytest src/tests/test_causal_cognitive_organs.py -k "ttm_last_position" -q
```

Esperado: caminho atual consulta quatro posições Spider e o teste falha.

- [ ] **Step 3: Implementar readout**

Substituir somente o ramo `ttm_entity_addressing`:

```python
query = hidden[:, -1, :].detach()
value = self.heartbeat.tt_memory.retrieve(
    query,
    top_k=1,
    already_pooled=True,
)
if value is not None:
    bounded = bound_memory_residual(value, query)
    residual_scale = self.ttm_residual_scale().to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    hidden_for_logits = hidden.clone()
    hidden_for_logits[:, -1, :] = (
        hidden[:, -1, :]
        + residual_scale * bounded.to(hidden.device, hidden.dtype)
    )
    ttm_memory_retrieved = True
    ttm_residual_applied = bool(residual_scale.detach().ne(0).item())
```

Não alterar o ramo sem entity addressing nem o bloco JEPA localizado depois da
linha 1348.

- [ ] **Step 4: Rodar testes verdes**

```powershell
python -m pytest src/tests/test_causal_cognitive_organs.py src/tests/test_organ_causal_qa.py -q
```

Esperado: todos passam.

- [ ] **Step 5: Commitar somente hunks TTM**

Como `model.py` já contém alterações JEPA do operador, preparar/stagear apenas
o hunk 747-792 e os testes. Confirmar antes do commit:

```powershell
git diff --cached -- src/f51_darwin/darwin_x_core/model.py
```

O diff staged não pode conter `jepa_temporal_mask_ratio`, `_jepa_loss` ou
`temporal_mask_ratio`.

Commit:

```powershell
git commit -m "fix(memory): route bounded TTM residual to readout"
```

---

### Task 3: Ensino alinhado no benchmark

**Files:**
- Modify: `src/scripts/benchmark_native_ttm_recall.py`
- Modify: `src/scripts/diagnose_native_ttm_recall.py`
- Modify: `src/tests/test_native_ttm_recall.py`
- Modify: `src/tests/test_native_ttm_diagnostic.py`

**Interfaces:**
- Consumes: `write_association`, prompt IDs e sequência teacher-forced.
- Produces: exatamente cinco slots com chave em prompt `T-1` e valor médio no span da resposta.

- [ ] **Step 1: Escrever testes vermelhos do extrator**

Criar helper puro:

```python
def association_sources(prompt_hidden, teaching_hidden, prompt_length, answer_length):
    key = prompt_hidden[:, prompt_length - 1, :]
    value = teaching_hidden[:, prompt_length:prompt_length + answer_length, :].mean(dim=1)
    return key, value
```

Testar índices exatos, shapes `[1, D]`, resposta vazia e não finitos.

- [ ] **Step 2: Confirmar RED**

```powershell
python -m pytest src/tests/test_native_ttm_recall.py -k association_sources -q
```

- [ ] **Step 3: Capturar hidden bruto e gravar associação**

Adicionar helper que registra hook temporário em `model.norm`, executa um
forward e retorna clone do hidden bruto. Em `_process_teaching_sequences`:

1. slots vazios;
2. forward prompt-only para chave;
3. forward teacher-forced para valor;
4. `write_association`;
5. restaurar slots anteriores.

Registrar `key_sha256`, `value_sha256`, `prompt_length` e `answer_length`.

- [ ] **Step 4: Atualizar o diagnóstico**

O probe deixa de depender de posições Spider como condição de sucesso e mede:

- `native_memory_retrieved`;
- top-1 e domínio;
- `T-1` como posição de injeção por contrato;
- diferença de logits;
- razão do residual limitado;
- cérebro invariável.

- [ ] **Step 5: Rodar testes focados**

```powershell
python -m pytest src/tests/test_native_ttm_recall.py src/tests/test_native_ttm_diagnostic.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add -- src/scripts/benchmark_native_ttm_recall.py src/scripts/diagnose_native_ttm_recall.py src/tests/test_native_ttm_recall.py src/tests/test_native_ttm_diagnostic.py
git commit -m "test(memory): teach aligned TTM associations"
```

---

### Task 4: Prova real, classificação e documentação

**Files:**
- Runtime replace: `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/*`
- Modify: `governance/docs/operacao/NATIVE_TTM_RECALL.md`
- Modify: `governance/docs/operacao/STATUS_ATUAL.md`
- Runtime modify: `workspace/runtime/history/agent_bus/2026-07-29-native-ttm-recall.md`

**Interfaces:**
- Consumes: benchmark e diagnóstico corrigidos.
- Produces: classificação científica e causa mecânica verificáveis por hashes.

- [ ] **Step 1: Gate de ambiente**

```powershell
git status --short
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match 'darwin_organism|run247|start_overnight'
}
```

Esperado: somente os dois arquivos JEPA conhecidos aparecem modificados; GPUs
livres; nenhum treinador concorrente.

- [ ] **Step 2: Testes de fonte**

```powershell
python -m pytest src/tests/test_native_ttm_diagnostic.py src/tests/test_native_ttm_recall.py src/tests/test_smol_dense_qa_benchmark.py src/tests/test_organ_causal_qa.py src/tests/test_live_inference_runtime.py src/tests/test_causal_cognitive_organs.py -q
python -m compileall -q src/scripts/benchmark_native_ttm_recall.py src/scripts/diagnose_native_ttm_recall.py
git diff --check
```

- [ ] **Step 3: Executar diagnóstico e benchmark real**

```powershell
python src/scripts/benchmark_native_ttm_recall.py
python src/scripts/diagnose_native_ttm_recall.py
python src/scripts/benchmark_native_ttm_recall.py --verify-report
```

O benchmark cria os novos slots alinhados e decide comportamento. O diagnóstico
executado depois fecha o canal e reconcilia o relatório. Não promover se apenas
logits mudarem.

- [ ] **Step 4: Atualizar autoridade de status**

Registrar números medidos, classificação, causa mecânica, hashes e qualquer
gate reprovado. Nunca substituir `no_memory_effect` por sucesso sem o contraste
correta versus desligada versus embaralhada.

- [ ] **Step 5: Commit de documentação**

```powershell
git add -- governance/docs/operacao/NATIVE_TTM_RECALL.md governance/docs/operacao/STATUS_ATUAL.md
git commit -m "docs(memory): publish TTM channel surgery result"
```

- [ ] **Step 6: Verificação final**

Repetir testes de fonte, `--verify-report`, `git diff --check`,
`git status --short` e `git log --oneline -8`. O worktree pode manter somente
as alterações JEPA pré-existentes e não commitadas.
