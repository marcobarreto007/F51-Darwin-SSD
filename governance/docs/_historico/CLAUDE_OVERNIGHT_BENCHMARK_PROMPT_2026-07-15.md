# Prompt operacional para Claude — benchmark causal noturno F51

Copie integralmente o bloco abaixo para Claude Code. Não resuma nem remova as
restrições. O objetivo é deixar uma execução auditável durante a noite e um
relatório verificável pela manhã.

---

## PROMPT PARA CLAUDE

Você está trabalhando exclusivamente em:

`C:\Users\marco\Desktop\F51-Darwin-SSD`

Atue como engenheiro de pesquisa responsável pela validade causal do benchmark,
não como redator, arquiteto contemplativo ou operador que apenas inicia um
processo. Execute o trabalho até obter os artefatos da manhã ou encontrar um
bloqueio seguro e comprovado.

### Objetivo final

Entregar um benchmark noturno reproduzível que responda:

1. O F51 reduz esquecimento de matemática ao aprender literatura?
2. O ganho vem de replay, Ghost, órgãos auxiliares ou da combinação?
3. O efeito permanece quando a fase de literatura é prolongada?
4. O resultado ocorre sob o mesmo estado inicial, dados, orçamento de tokens e
   janelas de avaliação?

O benchmark noturno é um **proxy causal 600M**, não uma alegação sobre a linhagem
local 1.6B. Não chame o resultado de 1.6B e não inicie `run247`.

### Resultado que prova conclusão

Ao terminar, devem existir:

- runner v2 separado e testado;
- launcher PowerShell com modo de plano sem lançamento;
- manifesto de prontidão e ambiente;
- agendas de batches e avaliações congeladas por seed;
- resultados atômicos por condição/seed;
- logs completos e retomáveis;
- `MORNING_REPORT.md` com tabela, variância, comparações pareadas, limitações e
  veredito;
- nenhuma condição silenciosamente alterada após o início.

### Verdades e limites obrigatórios

- Leia completamente `AGENTS.md`, `README.md`, `governance/docs/STATUS_ATUAL.md` e este
  prompt antes de agir.
- O alvo local oficial continua sendo `F51-Darwin-X-1.6B-Nitro`; o benchmark
  abaixo usa o 600M apenas para caber na janela noturna.
- Um processo deve possuir as duas GPUs; nunca lance condições concorrentes.
- Corpus e checkpoints físicos ficam fora do repositório, em
  `F51-Dataset-Organizado`.
- Nunca reconstrua corpus dentro do repositório.
- Não mate processo, não feche Ollama, não apague artefatos, não faça
  `git reset`, `git clean`, commit ou push.
- Preserve todas as alterações já existentes. Antes de editar, salve `git
  status --short` e `git diff --stat` no diretório do novo run.
- Não habilite mutação estrutural real durante a noite. O controlador permanece
  em shadow mode. Topologia ativa só poderá ser testada em outro experimento,
  depois de provar reconstrução do optimizer, checkpoint estrito e retomada.
- Nunca reduza steps, seeds, bloco, avaliação ou condições silenciosamente para
  fazer o run terminar.
- Nunca transforme crash, loss descendente ou uma seed favorável em descoberta.

### Fase 0 — auditoria real, sem alterações

Antes de escrever código:

1. Inspecione Git, os cinco últimos commits, processos Python, lock do runner,
   VRAM nas duas GPUs, espaço em disco, Python/PyTorch/CUDA, config 600M, config
   1.6B, corpus e resultados anteriores.
2. Confirme que não há `darwin_organism.py run247` nem `experiment_runner.py`
   ativo. Não conte o próprio PowerShell da inspeção como treino.
3. Leia `runs/experiments/20260714_205754/results.json` e
   `runs/experiments/20260714_085708/results.json`.
4. Registre no manifesto os defeitos conhecidos do runner v1:
   - instanciação aleatória sem carregar checkpoint;
   - replay buffer reiniciado entre matemática e literatura;
   - C1 com órgãos ainda ativos;
   - avaliação aleatória diferente entre condições;
   - treino incluindo a cauda usada como validação;
   - orçamento de tokens diferente quando replay concatena batch;
   - apenas 30 amostras de avaliação;
   - F51 estrutural em shadow mode.
5. Se outro treino estiver ativo ou a VRAM estiver insuficiente, não mate nada.
   Grave `BLOCKED.md` com PID, comando, GPUs e horário e encerre com segurança.

### Fase 1 — implemente um runner v2 sem destruir o v1

Crie, sem substituir o runner histórico:

- `src/scripts/experiment_runner_v2.py`
- `src/scripts/start_overnight_experiments_v2.ps1`
- `src/tests/test_experiment_runner_v2.py`

O launcher deve possuir `-Plan` como padrão seguro e exigir `-Launch` explícito
para usar CUDA. O modo `-Plan` não pode instanciar modelo grande nem reservar
VRAM.

O runner v2 deve aceitar no mínimo:

- `--run-id`
- `--tier smoke|main|long|scale-smoke`
- `--conditions`
- `--seeds`
- `--phase-a-steps`
- `--phase-b-steps`
- `--block`
- `--eval-samples`
- `--max-hours`
- `--resume`
- `--plan`

Use lock de sistema liberado automaticamente em crash. Salve JSON de forma
atômica após **cada condição e seed**, não somente depois de uma condição
inteira. Uma retomada deve pular somente unidades completas e validar que
config, agenda, código e datasets continuam idênticos.

### Contrato causal obrigatório

#### Estado inicial comum

Para cada seed:

1. Fixe Python, NumPy, Torch CPU e Torch CUDA.
2. Crie uma única inicialização base a partir de
   `src/configs/darwin_x_600m_ckpt239.yaml` e registre claramente `init=random`.
3. Execute a fase A de matemática uma única vez por seed sob o contrato
   `PHASE_A_SHARED`: órgãos e losses auxiliares canônicos ativos, replay
   desligado e topologia em shadow mode. Isso permite ao estado mutacional
   observar a aprendizagem antiga antes da clonagem. Salve um checkpoint
   temporário fora do repositório ou um estado CPU integral com hash.
4. Todas as condições da fase B devem carregar estritamente o mesmo estado da
   fase A. `ppl_math_before` e seu NLL devem ser idênticos por seed entre
   condições; diferença maior que `1e-6` invalida a seed.
5. Crie um optimizer novo e idêntico para cada condição da fase B.

Neste protocolo, `C0_STATIC` significa controle estático **durante a fase B**.
Todas as condições herdam a mesma aprendizagem de matemática da fase A. Declare
isso no relatório para não insinuar que C0 foi estático desde a inicialização.

Não use o nome `ckpt239` como prova de que um checkpoint foi carregado. Registre
explicitamente se o estado é aleatório, adaptado ou retomado.

#### Separação dos dados

- Reserve a cauda de validação antes de construir a visão de treino.
- Nenhum índice de treino pode alcançar a cauda de validação.
- Gere e salve antecipadamente, por seed:
  - índices da fase A matemática;
  - índices fresh da fase B literatura;
  - índices de replay matemático;
  - índices fixos de avaliação matemática;
  - índices fixos de avaliação literária.
- Todas as condições usam exatamente os mesmos índices correspondentes.
- Avalie sempre nas mesmas janelas antes e depois.
- Use no mínimo 256 blocos fixos por domínio no main/long. O smoke pode usar 16.

#### Orçamento igual

Cada update da fase B processa o mesmo número de tokens em todas as condições.
Replay não pode concatenar um segundo batch e ganhar compute.

Para condições com replay de 20%:

- a cada cinco updates, um update usa matemática antiga em vez de literatura;
- os outros quatro usam literatura fresh;
- o total de updates, tokens e bloco permanece idêntico ao controle;
- registre separadamente tokens fresh e tokens replay.

O replay matemático deve sobreviver da fase A para a fase B. Um buffer criado de
novo na fase B é falha de contrato.

### Condições e ablações

Implemente builders explícitos e teste os valores efetivos. Não derive condições
de nomes ou defaults ocultos.

1. `C0_STATIC`
   - fase B 100% literatura;
   - replay desligado;
   - `ghost_weight=0`, `ghost_mask_ratio=0`, `mtp_weight=0`, `jepa_weight=0`,
     `curiosity_weight=0`, Spider Sense e Heartbeat desligados;
   - nenhum órgão auxiliar altera loss ou forward.

2. `C1_REPLAY_ONLY`
   - mesmo estado/config comportamental de C0;
   - somente replay matemático de 20% na fase B;
   - nenhum Ghost, JEPA, MTP, Curiosity, Spider Sense ou Heartbeat.

3. `C2_GHOST_ONLY`
   - sem replay;
   - somente Ghost ativo com os valores canônicos do YAML;
   - MTP, JEPA, Curiosity, Spider Sense e Heartbeat desligados;
   - prove em teste que o sinal Ghost que chega ao backward é o que foi medido,
     e que não é sobrescrito antes do hook.

4. `C3_ORGANS_NO_REPLAY`
   - órgãos e losses auxiliares canônicos ativos;
   - replay desligado;
   - mutação estrutural em shadow mode.

5. `F51_FULL_SHADOW`
   - mesmos órgãos de C3;
   - replay matemático de 20% com a mesma agenda de C1;
   - controlador mutacional e eventos registrados;
   - topologia continua em shadow mode.

Comparações causais primárias:

- C1 − C0: efeito do replay;
- C2 − C0: efeito isolado do Ghost;
- C3 − C0: efeito dos órgãos sem replay;
- F51 − C1: efeito dos órgãos mantendo replay igual;
- F51 − C3: efeito do replay mantendo órgãos iguais.

### Testes obrigatórios antes de CUDA

Adicione testes causais que falhariam no runner v1:

1. validação não aparece em nenhum índice de treino;
2. agendas são idênticas entre condições comparáveis;
3. replay da fase B contém matemática da fase A;
4. replay não aumenta tokens por update;
5. C1 tem todos os órgãos desligados;
6. C2 ativa somente Ghost;
7. F51 e C1 usam exatamente a mesma agenda replay/fresh;
8. estado inicial e `math_before` são idênticos por seed;
9. resultados parciais retomam sem repetir unidade completa;
10. mismatch de hash/config/dataset recusa resume;
11. crash grava traceback e libera lock;
12. BF16 mantém geometria mutacional FP32;
13. shadow mode não altera número de experts;
14. estatística usa desvio-padrão amostral `ddof=1`;
15. relatório não chama FR próximo de 1 de transferência positiva sem intervalo
    e margem predefinida.

Rode antes do launch, com CUDA invisível:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m py_compile src\\scripts\\experiment_runner_v2.py src\\f51_darwin\\darwin_x.py
.\.venv_nitro\Scripts\python.exe -m pytest src\\tests\\test_experiment_runner_v2.py src\\tests\\test_gradient_mutational_state.py -q -p no:cacheprovider
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Qualquer falha bloqueia o launch. Não desabilite teste para prosseguir.

### Fase 2 — plano e smoke

Execute primeiro o launcher sem `-Launch`. O manifesto deve mostrar:

- PID/lock livres;
- duas GPUs detectadas;
- VRAM suficiente;
- datasets e tamanhos;
- config e hash;
- estimativa de duração baseada nos runs anteriores;
- espaço livre;
- matriz que será executada;
- comando exato de launch.

Depois, se e somente se a prontidão passar, execute um smoke sequencial:

- condições: todas as cinco;
- seed: 51;
- fase A: 20 steps;
- fase B: 20 steps;
- avaliação: 16 blocos;
- mesmo processo, duas GPUs, condições sequenciais.

O smoke deve verificar finitude de loss/NLL/PPL, igualdade do estado inicial,
artefatos atômicos, cleanup CUDA entre condições e resume. Se falhar, corrija a
causa, repita uma vez e não lance o main até passar.

### Fase 3 — matriz noturna por prioridade

Use orçamento máximo de 9 horas a partir do início do main. Não durma em loop e
não espere indefinidamente.

#### Tier MAIN — obrigatório

- proxy 600M;
- condições: as cinco;
- seeds: `51,52,53,54,55`;
- fase A: 500 steps por seed, compartilhada;
- fase B: 500 steps por condição;
- block: 32;
- avaliação fixa: 256 blocos por domínio.

#### Tier LONG — somente após MAIN completo

- condições: `C0_STATIC,C1_REPLAY_ONLY,F51_FULL_SHADOW`;
- seeds: `51,52,53`;
- reutilize o mesmo estado de fase A validado;
- fase B: 2000 steps;
- avaliação fixa: 512 blocos por domínio.

Se o orçamento restante não comportar o LONG inteiro com margem de 30 minutos,
não inicie o tier. Registre `SKIPPED_TIME_BUDGET`, a estimativa e os dados usados.
Não execute apenas uma condição do LONG, pois isso destruiria a comparação.

#### Tier SCALE-SMOKE — opcional e por último

Somente se MAIN e LONG terminarem e restarem pelo menos 90 minutos:

- valide o checkpoint canônico 1.6B externo, identidade, config, shapes e
  Topology Manifest v7;
- faça apenas avaliação + smoke de 20 steps, uma seed, C0/C1/F51;
- rotule claramente como smoke sem poder estatístico;
- não altere nem retome `run247`.

### Métricas obrigatórias

Por condição e seed registre:

- NLL e PPL matemática antes/depois;
- NLL e PPL literatura antes/depois;
- `FR = ppl_math_after / ppl_math_before`;
- `delta_math_nll = nll_after - nll_before`;
- ganho/adaptação de literatura;
- tokens fresh, replay e totais;
- updates, tempo, tokens/s e pico de VRAM por GPU;
- split das camadas;
- loss principal e cada loss auxiliar separadamente;
- contagem de observações mutacionais;
- conflito, residual ortogonal, estabilidade, flips e protected rank;
- propostas CREATE/EXPAND/PROTECT/IGNORE/PRUNE;
- eventos estruturais e confirmação de que nenhum foi executado;
- hash do estado inicial, agenda, código, config e manifesto de dados.

Use `math.isfinite` em todas as métricas. Um NaN/Inf invalida a unidade e gera
falha explícita.

### Estatística e critérios de decisão

Calcule por condição:

- valores individuais por seed;
- média;
- desvio-padrão amostral (`ddof=1`);
- intervalo de confiança de 95%;
- diferenças pareadas por seed para cada comparação causal;
- efeito absoluto e redução relativa do excesso de esquecimento `(FR - 1)`;
- contagem de seeds em que cada hipótese vence.

Não declare significância se a amostra não sustentar. Não use uma seed como
headline.

Classificação do MAIN:

- `SUPPORTED`: F51 vence a comparação pareada em pelo menos 4/5 seeds, média
  melhora e IC pareado de 95% não cruza zero;
- `PROMISING_BUT_INCONCLUSIVE`: média melhora, mas IC cruza zero ou vence menos
  de 4/5;
- `NO_EFFECT`: efeito absoluto pequeno e inconsistente;
- `REFUTED`: F51 piora em pelo menos 4/5 seeds e a média pareada piora;
- `INVALID`: qualquer quebra de estado inicial, agenda, held-out, orçamento ou
  artefato.

FR abaixo de 1 só pode ser chamado de transferência positiva se a melhoria
exceder uma margem predefinida de 2% (`FR <= 0.98`) e for consistente no
agregado. `FR=0.996` é “sem esquecimento mensurável”, não descoberta.

### Artefatos do run

Grave tudo em:

`runs/experiments_v2/<run_id>/`

No mínimo:

- `status.json`
- `readiness.json`
- `environment.json`
- `git_state.txt`
- `protocol.json`
- `condition_configs.json`
- `schedules/<seed>.json`
- `partial_results.json`
- `results.json`
- `failures.jsonl`
- `experiment.log`
- `MORNING_REPORT.md`
- `REPRODUCE.ps1`

Não coloque pesos, corpus nem checkpoints nessa pasta. Checkpoints temporários
pesados ficam no workspace externo e devem ser referenciados pelo manifesto.

### Tratamento de falhas

- Grave traceback completo, condição, seed, step, dtype, device e memória.
- Faça cleanup do modelo/optimizer e `torch.cuda.empty_cache()` entre unidades.
- Não mude batch, block ou condição após OOM. Marque a unidade como falha.
- Uma falha isolada pode permitir a próxima unidade somente se as GPUs estiverem
  saudáveis e o estado inicial continuar verificável.
- Duas falhas com a mesma assinatura indicam falha sistêmica: aborte o tier,
  preserve artefatos e escreva `BLOCKED.md`.
- Nunca crie retry infinito.

### Relatório da manhã

`MORNING_REPORT.md` deve começar com uma destas frases:

- `RESULTADO VÁLIDO — ...`
- `RESULTADO INCONCLUSIVO — ...`
- `RESULTADO REFUTADO — ...`
- `RESULTADO INVÁLIDO/BLOQUEADO — ...`

Depois apresente:

1. o que realmente terminou;
2. tabela por condição e seed;
3. comparações causais pareadas;
4. custo computacional;
5. sinais mutacionais observados;
6. falhas e condições puladas;
7. o que pode e não pode ser afirmado;
8. comando exato para reproduzir ou retomar;
9. recomendação de próximo experimento, sem alterar código adicional.

### Condição terminal

Você não termina ao escrever um plano nem ao iniciar um processo. Termine apenas
quando:

1. os testes e o smoke passaram;
2. o main terminou ou foi bloqueado com evidência;
3. o long terminou ou foi corretamente marcado por orçamento;
4. todos os artefatos foram fechados e validados;
5. não há processo órfão;
6. `MORNING_REPORT.md` existe e corresponde aos JSONs.

Na resposta final, seja curto: informe o status, o caminho absoluto do relatório,
tiers concluídos, quantidade de unidades válidas/falhas e se algum processo ainda
está ativo. Não proclame descoberta fora dos critérios acima.

## FIM DO PROMPT
