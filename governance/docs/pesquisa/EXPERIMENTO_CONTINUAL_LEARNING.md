# Experimento Controlado — Darwin-X Continual Learning

## Hipótese
O organismo Darwin-X reduz catastrophic forgetting vs baseline estático.

## Condições
- C0 Static: sem organismo, sem replay, sem hormonios
- C1 Replay: C0 + replay buffer no gradiente  
- F51 Full: organismo completo

## Domínios (ordem fixa)
- Fase A: Matematica (OpenWebMath2)
- Fase B: Literatura (Gutenberg classics)

## Métrica Primária
Forgetting Ratio = PPL_A_apos_B / PPL_A_antes_B
Sucesso: FR < 1.2 E menor que C0 (p < 0.1, 3 seeds)

## Protocolo
50.000 steps por fase. 3 condicoes x 3 seeds = 900.000 steps (~4h).