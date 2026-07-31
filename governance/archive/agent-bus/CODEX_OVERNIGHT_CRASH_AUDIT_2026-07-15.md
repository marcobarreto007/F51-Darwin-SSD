# Auditoria do crash noturno — 2026-07-15 EDT

## Objetivo

Verificar por evidencia por que o `run247` da linhagem
`F51-Darwin-X-1.6B-Nitro` parou e se existe estado recuperavel, sem iniciar ou
matar processos e sem alocar CUDA.

## Estado vivo

- Nenhum processo `darwin_organism.py run247` ativo.
- GPUs ociosas para treino; RTX 5060 Ti com apenas memoria grafica de desktop e
  RTX 3060 sem memoria alocada.
- Ultimo stdout: `2026-07-14 08:46:06 EDT`, step 56351 do ciclo 78.
- Ultimo checkpoint publicado: `organism_cycle_077.pt`, cycle 77, step 54501,
  11.914.584.035 bytes, publicado em `2026-07-14 08:05:31 EDT`.
- Nenhum `.tmp` abandonado no diretorio de checkpoints.

## Causa do encerramento

O stdout termina abruptamente, sem pausa, traceback, erro de CUDA em Python ou
gate de disco. O stderr contem apenas um aviso antigo de autenticacao do HF Hub.
Oito segundos depois do ultimo write do stdout, o System Event Log registrou:

- `2026-07-14 08:46:14 EDT`
- provider `nvlddmkm`, Event ID 153
- `\Device\Video8`
- `Error occurred on GPUID: 100`

`GPUID 100` corresponde ao PCI bus `01:00.0`, GPU 0, RTX 5060 Ti. Ha historico
recorrente: 105 eventos `nvlddmkm` nos ultimos 30 dias, 88 deles no GPUID 100,
incluindo eventos TDR. Nao houve reboot no horario da queda; o reboot posterior
foi planejado pelo Windows Update em 2026-07-15 02:14 EDT.

Conclusao: causa raiz operacional mais provavel e falha/reset do driver NVIDIA
na RTX 5060 Ti, que encerrou externamente o processo CUDA.

## Recuperabilidade do checkpoint 77

O checkpoint abriu por `torch.load(..., mmap=True)` e passou:

- checkpoint v7, config embutida igual ao YAML canonico;
- identidade declarada e integral recalculada exatamente iguais:
  `darwin-model-core-v1:f6870753c784beb8201546dd7e7450ac998f4663f8c4741f17481e86b87ba56b`;
- Topology Manifest v7 restaurou 66 mudancas estruturais;
- migracao atual inicializou 208 buffers mutacionais adicionados depois do run;
- load estrito: 0 missing, 0 unexpected;
- 1.000 shapes de momentum AdamW alinhados em ordem aos 1.000 parametros do
  modelo restaurado.

O checkpoint 77 e recuperavel.

## Bloqueios antes de retomar

1. O launcher oficial falha falsamente no inspetor: `inspect_organism_checkpoint.py`
   compara o checkpoint evoluido com a anatomia-base sem aplicar o Topology
   Manifest. Ele retorna `resume_shape_compatible=false`, apesar de o bootstrap
   real restaurar e carregar estritamente o mesmo checkpoint.
2. Disco C: com 71,66 GiB livres; o launcher exige pelo menos 100 GiB.
   Os checkpoints externos ocupam 137,91 GiB e nao foram apagados.
3. A instabilidade recorrente do driver/GPU 0 precisa ser mitigada antes de novo
   treino longo; relancar sem isso pode repetir a queda.

## Validacoes

- `py_compile` de runtime, modelo e inspetor: aprovado.
- suite completa CPU (`CUDA_VISIBLE_DEVICES=-1`): aprovada, exit code 0.
- `scripts/darwin_inventory.py`: corpus principal OK, ponteiro para cycle 77 OK,
  checkpoint reconhecido como torch ZIP.
- Gate oficial executado sem `-Launch`: recusado no falso negativo de anatomia;
  nenhum processo foi iniciado.

Nenhum codigo existente foi alterado nesta auditoria. Nenhum checkpoint foi
apagado, movido ou criado e nenhuma GPU foi usada para treino.
