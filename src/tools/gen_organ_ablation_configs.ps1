# Gera 11 configs de ablacao (baseline + 1 orgao ligado por vez) a partir de
# src/configs/darwin_x_100m_baseline_puro.yaml, cada um com checkpoint_root proprio.
$ErrorActionPreference = "Stop"
$baseline = Get-Content "src\\configs\\darwin_x_100m_baseline_puro.yaml" -Raw

$organs = @{
    "mtp" = @{
        "mtp_weight: 0.0" = "mtp_weight: 0.15"
    }
    "jepa" = @{
        "jepa_weight: 0.0" = "jepa_weight: 0.05"
        "jepa_dist_weight: 0.0" = "jepa_dist_weight: 1.0"
    }
    "ghost" = @{
        "ghost_weight: 0.0" = "ghost_weight: 0.07"
        "ghost_enabled: false" = "ghost_enabled: true"
    }
    "spider" = @{
        "spider_sense_enabled: false" = "spider_sense_enabled: true"
        "spider_calibration_enabled: false" = "spider_calibration_enabled: true"
        "spider_calibration_weight: 0.0" = "spider_calibration_weight: 0.02"
    }
    "heartbeat_ttm" = @{
        "heartbeat_enabled: false" = "heartbeat_enabled: true"
        "ttm_residual_enabled: false" = "ttm_residual_enabled: true"
        "ttm_residual_max_scale: 0.0" = "ttm_residual_max_scale: 0.15"
        "ttm_associative_weight: 0.0" = "ttm_associative_weight: 0.02"
    }
    "dae" = @{
        "dae_enabled: false" = "dae_enabled: true"
    }
    "gaba" = @{
        "gaba_enabled: false" = "gaba_enabled: true"
    }
    "inter_hemispheric" = @{
        "inter_hemispheric_enabled: false" = "inter_hemispheric_enabled: true"
    }
    "sleep" = @{
        "sleep_enabled: false" = "sleep_enabled: true"
    }
    "decision_engine" = @{
        "decision_engine_enabled: false" = "decision_engine_enabled: true"
    }
    "unified_mesh" = @{
        "unified_mesh_enabled: false" = "unified_mesh_enabled: true"
    }
}

foreach ($organ in $organs.Keys) {
    $content = $baseline
    foreach ($find in $organs[$organ].Keys) {
        $replace = $organs[$organ][$find]
        $content = $content.Replace($find, $replace)
    }
    $ckptRoot = "workspace/03_CHECKPOINTS_100M_ABLATION_$($organ.ToUpper())"
    $content = $content.Replace(
        'checkpoint_root: "workspace/03_CHECKPOINTS_100M_BASELINE_PURO"',
        "checkpoint_root: `"$ckptRoot`""
    )
    $content = $content.Replace(
        "blockchain_path: workspace/03_CHECKPOINTS_100M_BASELINE_PURO/blocks.jsonl",
        "blockchain_path: $ckptRoot/blocks.jsonl"
    )
    $header = "# F51 Darwin-X 100M - Ablacao: SO $($organ.ToUpper()) ligado, resto igual ao baseline puro (ver src/configs/darwin_x_100m_baseline_puro.yaml).`n"
    $content = $header + $content
    $outPath = "src\\configs\\darwin_x_100m_ablation_$organ.yaml"
    Set-Content -LiteralPath $outPath -Value $content -Encoding UTF8 -NoNewline
    Write-Host "Gerado: $outPath"
}
