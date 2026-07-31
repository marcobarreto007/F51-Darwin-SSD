# F51 Darwin-X — Design do documento institucional e técnico

Data: 2026-07-18

Público: investidores, parceiros e equipe técnica

Artefato final: `C:\Users\marco\Desktop\F51-Darwin-X-O-Que-E-O-Nosso-Sistema.pdf`

## Objetivo

Produzir um documento único que explique o que é o F51 Darwin-X em linguagem
clara, sem perder rigor técnico. A primeira parte deve permitir que um
investidor ou parceiro compreenda o propósito, o diferencial e o potencial do
sistema. A segunda parte deve permitir que uma equipe técnica entenda a
arquitetura, o fluxo operacional, os órgãos, as evidências existentes e os
limites atuais.

O documento não é uma peça de hype. Toda afirmação deve ser compatível com o
estado atual do repositório, dos testes, dos manifests e dos artefatos
verificados.

## Mensagem central

O F51 Darwin-X é uma plataforma local de pesquisa e engenharia para treinar,
avaliar, evoluir, retomar e servir modelos de linguagem híbridos
SSD + Attention + MoE. Sua direção de ruptura é transformar mecanismos
inspirados em órgãos — como GABA, Ghost, Curiosity, Soul e Decision — em
componentes computacionais causais, mensuráveis, persistentes e sujeitos a
controle negativo.

O diferencial não será descrito como “biologia artificial comprovada”. Os
nomes biológicos representam funções de engenharia. O valor técnico está na
separação entre observação, proposta, decisão, aplicação futura, persistência,
ablação e rollback.

## Regra de verdade

Toda capacidade relevante receberá uma destas classificações:

- **OPERACIONAL**: integra a superfície suportada e possui prova operacional
  atual.
- **IMPLEMENTADO E TESTADO**: existe no código e possui testes, mas pode ainda
  não ter sido validado em treino longo.
- **EM INTEGRAÇÃO**: possui implementação parcial ou está sendo conectado ao
  ciclo completo.
- **PESQUISA**: hipótese, experimento ou direção ainda não promovida a
  capacidade operacional.

Documentação antiga, nomes de arquivos, loss isolada, ciclo maior ou mtime não
serão usados como prova de qualidade, linhagem ou prontidão.

## Estrutura do PDF

### 1. Capa e resumo executivo

- nome F51 Darwin-X;
- descrição em uma frase;
- missão;
- estágio atual em linguagem honesta;
- quatro classificações de maturidade.

### 2. O problema

- limitações de treinos convencionais baseados apenas em minimizar loss;
- diferença entre memorizar replay e generalizar em dados fresh/heldout;
- dificuldade de controlar adaptação, evolução e memória sem vazamento causal;
- necessidade de linhagem, rollback e evidência reproduzível.

### 3. O que é o sistema

- plataforma local-only;
- modelo híbrido SSD + Attention + MoE;
- corpus governado;
- treinamento, avaliação, checkpoint, resume e serving;
- organismo causal como camada de coordenação e evidência.

### 4. Diferencial

- órgãos tratados como mecanismos de engenharia;
- observação pós-outcome;
- propostas válidas apenas para `t+1` ou fronteira de ciclo;
- `CONTROL`, `SHADOW` e `APPLY`;
- loss crua separada de loss efetiva;
- checkpoint e ledger com identidade;
- fail-closed, rollback e decisão humana nos limites operacionais.

### 5. Arquitetura

- plano tensorial;
- plano causal;
- plano executivo;
- plano de persistência;
- plano operacional;
- relação entre corpus, modelo, órgãos, ledger e checkpoints.

Um diagrama mostrará:

`corpus aprovado → wake/forward → outcome imutável → órgãos observam →
Soul propõe → Decision aprova/adia/veta → intervenção t+1 → checkpoint/sleep`.

### 6. Órgãos

Cada órgão terá:

- função de engenharia;
- entrada;
- saída;
- estado persistente;
- momento permitido de atuação;
- maturidade atual;
- teste ou evidência;
- limitação conhecida.

Órgãos e mecanismos cobertos:

- GABA;
- Ghost Token;
- Curiosity;
- Soul;
- Decision Engine;
- Ghost Brain;
- JEPA;
- Spider-Sense;
- Heartbeat/TTM;
- controle neuroendócrino MoE;
- evolução estrutural;
- protected sleep.

### 7. Fluxo causal do organismo

- estado congelado antes da tentativa;
- forward e coleta de evidência;
- outcome;
- criação de frame imutável;
- proposta;
- decisão;
- aplicação futura;
- ledger;
- checkpoint;
- rollback quando necessário.

O documento deixará explícito que o organismo não pode recompensar ou alterar
o próprio passo usando informação produzida nesse mesmo passo.

### 8. Dados, treinamento e checkpoints

- ingestão com quarentena e aprovação;
- corpus `feast_v2`;
- linhagens 100M e 1.6B tratadas separadamente;
- fresh, replay e heldout;
- checkpoint atômico;
- pointer verificado;
- resume estrito;
- promoção nunca baseada somente em loss, nome ou cycle.

Números de checkpoint, hash, espaço em disco e resultados voláteis só serão
incluídos quando verificados novamente durante a produção do PDF.

### 9. Evidências atuais

- testes de source;
- ablações causais;
- identidade bitwise de braços inativos;
- provas específicas de GABA, Ghost e Curiosity;
- compatibilidade de checkpoint;
- limites dos benchmarks atuais.

O documento não alegará ganho de perplexidade, estado da arte ou superioridade
sobre baselines sem benchmark externo equivalente.

### 10. Aplicações e visão de produto

- pesquisa de modelos adaptativos locais;
- treinamento soberano;
- experimentação causal de mecanismos cognitivos;
- serving local;
- plataforma de avaliação e evolução controlada;
- futura aplicação em agentes e sistemas especializados.

As aplicações serão descritas como direção de produto quando ainda não
operacionais.

### 11. O que falta

- integração completa dos órgãos restantes;
- contrato cognitivo v2 completo;
- treino causal longo;
- benchmark externo real;
- medição de custo/benefício por órgão;
- validação de 1.6B;
- segurança e prontidão antes de qualquer promoção.

### 12. Apêndice técnico

- raiz operacional;
- entrypoints suportados;
- comandos seguros de inventário, inspeção e canary sem launch;
- hierarquia de prova;
- documentos canônicos;
- glossário.

## Forma visual

- PDF em português;
- estética técnica e sóbria;
- capa limpa;
- hierarquia forte de títulos;
- caixas de maturidade;
- tabelas pequenas para órgãos e evidências;
- um diagrama de arquitetura;
- um diagrama do ciclo causal;
- rodapé com data e estado do snapshot;
- sem imagens decorativas ou alegações biológicas literais.

## Fontes de verdade

Durante a produção, a ordem de autoridade será:

1. runtime vivo e artefatos carregados;
2. manifests, pointers e hashes;
3. código e testes;
4. `governance/docs/operacao/STATUS_ATUAL.md`;
5. documentos arquiteturais canônicos;
6. histórico apenas como contexto.

O estado será atualizado por inspeção real antes de fechar o conteúdo.

## Critérios de aceitação

- o leitor não técnico entende o sistema sem ler o apêndice;
- o leitor técnico consegue identificar componentes, fluxos e contratos;
- toda capacidade tem maturidade explícita;
- nenhuma visão futura aparece como entrega pronta;
- números voláteis são verificados no mesmo snapshot;
- o PDF abre corretamente e possui renderização visual revisada;
- o arquivo final existe na Área de Trabalho;
- a fonte do documento fica versionada no repositório;
- não ocorre treino, canary com launch, alteração de checkpoint ou mudança em
  configuração operacional durante a produção.

## Fora de escopo

- alterar código do organismo;
- iniciar treino;
- selecionar ou promover checkpoint;
- publicar externamente;
- criar alegações de benchmark não comprovadas;
- declarar equivalência biológica;
- reescrever documentação operacional autoritativa.
