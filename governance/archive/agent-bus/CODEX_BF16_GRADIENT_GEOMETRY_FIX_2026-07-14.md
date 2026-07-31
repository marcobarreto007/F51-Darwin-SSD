# Codex — correção BF16 da geometria mutacional (2026-07-14)

## Sintoma

O experimento C1 caiu em `NeuroendocrineSystem._measure_gradient_geometry()`
com `RuntimeError: addmv input tensors must have the same dtype`, envolvendo
`Float` e `BFloat16`.

## Causa reproduzida

`DeepSeekStyleMoE.to(dtype=torch.bfloat16)` convertia também os buffers
`protected_subspace` e demais sketches mutacionais para BF16. A coleta de
assinaturas, corretamente, produz sketches FP32. O primeiro produto matricial
com uma base protegida de rank maior que zero misturava os dois dtypes.

## Correção

- `NeuroendocrineSystem._apply()` preserva os buffers geométricos mutacionais
  em FP32 durante mudanças de dispositivo/dtype.
- Pesos e ativações dos experts continuam BF16.
- Normas e uso recebidos pelo controlador são normalizados para FP32.
- Foi adicionado teste causal que cria uma base protegida, mede conflito e
  reconsolida memória dentro de um MoE convertido para BF16.

## Verificação

- reprodução mínima anterior: falha determinística em `darwin_x.py:604`;
- teste BF16 causal: passou;
- suítes relacionadas: 44 testes passaram;
- suíte CPU completa: 226 testes passaram;
- `py_compile`: passou;
- `git diff --check`: passou (somente avisos de LF/CRLF).

Nenhum treino ou experimento foi reiniciado nesta sessão.
