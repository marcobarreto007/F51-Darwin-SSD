# AUDITORIA: Conexões Neurais e Ciclo de Vida

> Registro histórico pré-correção. Em 2026-07-13, ações estruturais foram
> retiradas do backward e passaram a ser propostas auditáveis; o vínculo
> vertical tornou-se transitório; a janela de sono é shape-safe e não afirma
> replay; fusão destrutiva está desativada. Para o estado atual, consulte
> `governance/docs/NEUROENDOCRINE_IMPLEMENTATION.md`.

## ❌ FALHAS CRÍTICAS

### 1. Desconexão Entre Camadas (Arquitetura)

**Problema**: Cada `DeepSeekStyleMoE` opera como uma ilha isolada.

```python
# darwin_x.py, linha ~643
for block in self.blocks:
    x, aux = block(x)  # Cada block tem seu próprio MoE, isolado
```

**Cérebro humano**: Neurônios em L2 se conectam com L1, L3 e lateralmente.
**Darwin-X**: Expert #5 em layer 3 não tem nenhuma conexão com expert #5 em layer 4.

**Impacto**: Não há continuidade de representação através das camadas. Um "expert de matemática" no layer 3 não conversa com o "expert de matemática" no layer 4.

---

### 2. Neuroendócrino Não Se Expande

**Problema**: `_create_expert()` adiciona expert mas quebra o sistema hormonal.

```python
# NeuroendocrineSystem.__init__
self.register_buffer('dopamine', torch.zeros(num_experts))  # TAMANHO FIXO

# _create_expert() linha 768
self.fine_experts.append(new_expert)  # Adiciona expert...
# Mas neuroendocrine.dopamine continua com tamanho antigo!
```

**Resultado**: Quando tenta acessar `dopamine[expert_idx]` com idx >= num_experts original → IndexError.

**Cérebro humano**: Hipotálamo se adapta quando novos neurônios nascem.
**Darwin-X**: Sistema hormonal ignora neurogênese.

---

### 3. Expansão Quebra Referências

**Problema**: `_expand_expert()` substitui o expert inteiro.

```python
# linha 745
self.fine_experts[expert_idx] = new_expert  # Substituição completa
```

**Consequências**:
- Expert antigo é descartado (perda de conhecimento)
- Neuroendócrino não é notificado (dopamina/BDNF do expert antigo somem)
- Nitro GPU tracking pode apontar para device fantasma

**Cérebro humano**: Crescimento neuronal é gradual — dendritos se estendem, sinapses se fortalecem.
**Darwin-X**: "Crescimento" é destruição + reconstrução.

---

### 4. Prune Não É Morte Real

**Problema**: Expert podado continua existindo.

```python
# _prune_expert() linha 712-720
def _prune_expert(self, expert_idx: int) -> None:
    # Move to CPU, freezes grads
    self.fine_experts[expert_idx].to('cpu')
    for p in self.fine_experts[expert_idx].parameters():
        p.requires_grad = False
    # Reduce router bias
    self.fine_router.expert_bias.data[expert_idx] -= 3.0
```

**O que falta**:
- Expert ainda está em `self.fine_experts[]`
- Consome memória (pesos congelados)
- Pode ser "ressuscitado" pelo Nitro (`_restore_expert`) mas isso não é ressurreição biológica
- Não há remoção real como apoptose

**Cérebro humano**: Neurônios mortos são removidos por micróglia, espaço é reutilizado.
**Darwin-X**: "Morte" é apenas congelamento.

---

### 5. Sleep Phase Incompleto

**Problema**: `_enter_sleep_phase()` apenas seta flags.

```python
# linha 795-799
def _enter_sleep_phase(self) -> None:
    self.neuroendocrine.trigger_sleep_phase()
    self._sleep_active = True  # Apenas uma flag!
    self._sleep_steps_remaining = 10
```

**O que falta**:
- Não há replay de exemplos antigos
- Não há poda sináptica durante sono
- Não há fusão de experts redundantes
- Não há verificação de correspondência醒来

**Cérebro humano**: Sono REM/NREM consolida memória através de replay hipocampal.
**Darwin-X**: Sono é um booleano.

---

## ⚠️ FALHAS MODERADAS

### 6. Falta Sinapses Inter-Layer

No cérebro, um neurônio em L3 tem axônios que conectam com L4, L5 e L2.
Em Darwin-X, não há nenhuma conexão direta entre experts de layers diferentes.

**Possível solução**: "Vertical connections" onde experts da mesma especialização em layers adjacentes compartilham estado.

---

### 7. Plasticidade Apenas LTP, Sem LTD

**Dopamina alta** → gate amplificado (LTP: long-term potentiation)
**Dopamina baixa** → gate atenuado

Mas não há mecanismo explícito de **LTD** (long-term depression) onde sinapses são explicitamente enfraquecidas por falta de uso.

O sistema depende de "esquecimento" via decaimento exponencial, mas não há poda ativa de pesos individuais.

---

### 8. Homeostase Frágil

Se dopamina explodir (ex: gradiente NaN → valores enormes), o sigmoid clampa em [0, 1], mas:
- Não há mecanismo de "reset" em caso de disfunção
- Não há detecção de "doença" sistêmica
- Não há quarentena de experts doentes

---

## ✅ O QUE ESTÁ SÓLIDO

1. **Organismo interno**: Lei 1 cumprida — backward hook automático
2. **Decaimento exponencial**: τ específico por hormônio — biologicamente plausível
3. **Sinais locais**: Gradientes + entropy + usage — não depende de loss externa
4. **Gate multiplicativo**: `sigmoid(DA) * expert_out` — mecanismo plausível

---

## 📊 COMPARAÇÃO COM CÉREBRO HUMANO

| Aspecto | Cérebro Humano | Darwin-X | Status |
|---------|----------------|----------|--------|
| Neurogênese | Hipocampo, ~700 novos neurônios/dia | `_create_expert()` | ⚠️ Quebra sistema hormonal |
| Apoptose | Morte programada, 50-70% de neurônios morrem | `_prune_expert()` | ❌ Não remove, apenas congela |
| Plasticidade | LTP + LTD simultâneos | Apenas LTP via dopamina | ⚠️ LTD implícito apenas |
| Sono | REM/NREM, replay, consolidação | `_enter_sleep_phase()` | ❌ Flag sem implementação |
| Conexões | Dendritos+axônios, rede pequeno mundo | MoE isolado por camada | ❌ Ilhas sem pontes |
| Homeostase | Glicose, oxigênio, temperatura | Tau de decaimento | ⚠️ Sem mecanismos de proteção |

---

## 🔧 PRIORIDADES DE CORREÇÃO

1. **CRÍTICO**: Neuroendócrino dinâmico (se expande com neurogênese)
2. **CRÍTICO**: Expansão sem destruição (crescimento gradual)
3. **ALTA**: Sleep phase real (replay, poda, fusão)
4. **ALTA**: Prune real (remoção + reutilização de espaço)
5. **MÉDIA**: Sinapses inter-layer (conexões verticais)
6. **BAIXA**: LTD explícito (poda de pesos por falta de uso)
