# Recuperacao do crash dual-GPU — 2026-07-15 EDT

## Resultado operacional

O `run247` iniciado em 2026-07-15 08:52:04 EDT encerrou abruptamente durante
o ciclo 69. O ultimo log foi o step 36751, mas o ultimo checkpoint atomico
publicado e recuperavel e `organism_cycle_068.pt`, cycle 68, step 36501.
Nenhum relaunch de treino foi feito nesta investigacao.

## Evidencia do crash

- o processo Python desapareceu sem shutdown do organismo;
- `train.stderr.log` permaneceu vazio;
- a VRAM das duas GPUs foi liberada;
- em 2026-07-15 13:09:45 EDT o System Event Log registrou `nvlddmkm`, Event ID
  153, `Error occurred on GPUID: 400`;
- `GPUID 400` corresponde ao barramento `04:00.0`, RTX 3060;
- a auditoria do crash anterior, em 2026-07-14, registrou o mesmo Event ID 153
  em `GPUID 100`, barramento `01:00.0`, RTX 5060 Ti;
- nao houve evento WHEA correlacionado.

Isso torna uma falha intermitente do stack driver/WDDM/caminho dual mais
provavel que defeito permanente de apenas uma placa. Ainda nao e causa raiz
provada.

## Checkpoint 68

- caminho: `F51-Dataset-Organizado/03_CHECKPOINTS/organism_cycle_068.pt`;
- tamanho: 10.876.779.141 bytes;
- SHA-256: `4416A9AF6E1A3105FCF207616E7A134D2484CB3E0B9EE36281921446230CF6CC`;
- checkpoint v7, `F51-Darwin-X-1.6B-Nitro`, 16 layers, d_model 1920;
- training state: cycle 68, step 36501;
- base checkpoint ID:
  `darwin-model-core-v1:4d5d97e7fd0c38f6e703e0db0d35d5bb2b65d233623486da39123a48bdb12906`;
- 1.515 tensores no model state;
- AdamW: 997 entradas de estado para 997 parametros no grupo;
- carregamento de metadados por `torch.load(..., mmap=True)` aprovado.

## Testes CUDA controlados

Todos os testes usaram o ambiente `.venv_nitro`, PyTorch
`2.12.0.dev20260408+cu128`, sem carregar o Darwin-X.

1. RTX 3060 isolada:
   - 6.979.321.856 bytes preenchidos e verificados;
   - 45 s de workload `dX/dW` com tres GEMMs por iteracao;
   - 2.844 iteracoes, exit 0, sem novo `nvlddmkm`.
2. RTX 5060 Ti isolada:
   - 9.395.240.960 bytes preenchidos e verificados;
   - 45 s do mesmo workload;
   - 4.901 iteracoes, exit 0, sem novo `nvlddmkm`.
3. Dual-GPU curto:
   - `cudaDeviceCanAccessPeer` falso nos dois sentidos;
   - copia 5060 Ti -> 3060: 6,336 GB/s;
   - copia 3060 -> 5060 Ti: 6,267 GB/s;
   - GEMMs simultaneas e troca de ativacoes por 45,93 s, exit 0.
4. Dual-GPU sustentado:
   - aproximadamente 8 GiB alocados na 5060 Ti e 6 GiB na 3060;
   - 300,72 s, 422.734 iteracoes, exit 0;
   - pico observado: 74 C na 5060 Ti e 70 C na 3060;
   - RTX 3060 atingiu 100% e aproximadamente 170 W;
   - nenhum novo evento `nvlddmkm`.

## Ambiente que permanece suspeito

- driver NVIDIA 595.97 WHQL em ambas as placas;
- Windows WDDM;
- plano de energia `Balanced`;
- PCIe Link State Power Management em `Moderate power savings` no AC;
- sem valores TDR customizados no registro;
- P2P CUDA indisponivel entre as placas.

O driver nao foi alterado, o plano de energia nao foi alterado e nenhum reboot
foi solicitado. O proximo teste deve mudar apenas uma variavel.

## Defeito de observabilidade confirmado

O replay usa razao 0,20 e ocorre deterministicamente nos passos multiplos de
5. O logger imprime nos passos multiplos de 50. Assim, quase toda `loss`, `lm`
e `ppl` mostrada no stdout e de batch de replay. `ppl=1.0` nao e validacao
held-out e nao prova qualidade.

Antes do canario, o organismo precisa separar metricas de batch fresco,
replay de treino e holdout posicional realmente excluido do treino, persistindo
as avaliacoes em artefato estruturado.
