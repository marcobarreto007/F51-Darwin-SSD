# CODEX - reconciliacao documental - 2026-07-13

## Objetivo

Alinhar a documentacao canonica com o runtime real e impedir que atividade da linhagem 1.76B seja confundida com retomada do alvo 2.5B.

## Arquivos corrigidos

- `README.md`
- `docs/STATUS_ATUAL.md`
- `AGENTS.md`
- `CLAUDE.md`
- `scripts/README.md`
- `ROADMAP.md` recebeu aviso de documento historico

## Verdade persistida

- objetivo: checkpoint treinado da classe 2.5B, corpus externo, duas GPUs, todos os orgaos e resume fiel;
- processo observado: 1.760B, config `darwin_x_1.6b_nitro.yaml`, inicializacao aleatoria, sem resume;
- ciclo 1 completou 500 steps e publicou externamente `organism_cycle_001.pt` com 10.874.301.839 bytes;
- `organism_latest.json` passou a apontar para essa linhagem 1.76B;
- o checkpoint 239 continua sendo legado 576M e nenhum dos dois e o checkpoint 2.5B;
- Ghost Stream gravou candidatos/rejeitados dentro do repo, violando a fronteira externa;
- `scripts/start_overnight_25b.ps1` permanece o gate oficial e recusa substituto menor.

## Validacao

- links locais principais existem;
- afirmacoes antigas de treino parado, split fixo 7/5 e 600M como alvo atual foram removidas dos documentos canonicos;
- nenhum processo foi parado e nenhum artefato foi apagado.
