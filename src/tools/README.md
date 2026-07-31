# Maintenance tools

Esta pasta contem verificadores, migradores e utilitarios de manutencao.
Nenhum arquivo daqui e um launcher de producao ou uma superficie suportada.
Execute modulos Python a partir da raiz com o namespace `src` configurado, por
exemplo:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m tools.check_operational_surface
```

A classificacao normativa esta em
`governance/audit/policy/operational-surface.json`.

Dashboards, verificadores de checkpoint, utilitarios de transferencia e
launchers experimentais ficam aqui para nao se confundirem com os entrypoints
oficiais de `src/scripts/`. Benchmarks, simulacoes, probes e treinamento
experimental ficam em `research/`.
