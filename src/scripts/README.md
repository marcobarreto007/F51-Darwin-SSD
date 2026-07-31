# Scripts

A lista normativa esta em `governance/audit/policy/operational-surface.json`.

## Entry points suportados

- `bob.ps1`
- `start_overnight_16b.ps1`
- `start_100m_65b.ps1`
- `start_100m_auto.ps1`
- `start_smol_darwin_transplant.ps1`
- `darwin_organism.py`
- `serve_davi.py`
- `inspect_organism_checkpoint.py`
- `darwin_inventory.py`
- `ingest_pipeline.py`

Os adaptadores Python sao invocados a partir da raiz como módulos. Configure o
namespace `src` antes da primeira chamada:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv_nitro\Scripts\python.exe -m scripts.darwin_organism --help
```

A forma de invocar um arquivo diretamente nao faz parte do contrato, pois muda
a raiz de imports do Python.

Nao ha outros executaveis nesta pasta. Manutencao fica em `src/tools/`; pesquisa e
experimentos ficam em `research/`. Eles nao constituem launchers alternativos
e nao devem ser promovidos por nome. Material aposentado fica em `governance/archive/` e
nao deve ser executado.
