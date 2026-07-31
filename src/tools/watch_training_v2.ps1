param(
    [string]$MetricsJsonl = "workspace\runtime\runs\full_organism_v1\metrics.jsonl",
    [string]$CheckpointRoot = "workspace\03_CHECKPOINTS_100M_FULL_ORGANISM_V1",
    [int]$Interval = 5
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$host.UI.RawUI.WindowTitle = "F51 Darwin-X Monitor"

while ($true) {
    Clear-Host
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  F51 Darwin-X - Full Organism V1" -ForegroundColor Cyan
    Write-Host "  $ts  (Ctrl+C para sair)" -ForegroundColor DarkGray
    Write-Host "========================================" -ForegroundColor Cyan

    # -- GPU --
    Write-Host "`n-- GPU --" -ForegroundColor Yellow
    $gpu = & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        foreach ($line in $gpu) {
            $parts = @($line -split ',\s*')
            if ($parts.Count -ge 6) {
                Write-Host "  GPU$($parts[0]): $($parts[2])% util  $($parts[3]) / $($parts[4]) MiB  $($parts[5])C  [$($parts[1])]"
            }
        }
    }

    # -- Processo --
    Write-Host "`n-- PROCESSO --" -ForegroundColor Yellow
    $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "darwin_organism" })
    $liveRunId = $null
    $liveCkptRoot = $null
    $liveMetricsJsonl = $null
    if ($procs.Count -eq 0) {
        Write-Host "  NENHUM processo darwin_organism rodando!" -ForegroundColor Red
    } else {
        foreach ($p in $procs) {
            $mem = [math]::Round($p.WorkingSetSize / 1MB, 0)
            $cmd = $p.CommandLine
            $config = "?"
            if ($cmd -match '--config\s+(\S+\.yaml)') { $config = $Matches[1] }
            $procRoot = "?"
            if ($cmd -match '--checkpoint-root\s+(\S+)') { $procRoot = $Matches[1] }
            $runId = $null
            if ($cmd -match '--run-id\s+(\S+)') { $runId = $Matches[1] }
            $procMetrics = $null
            if ($cmd -match '--metrics-jsonl\s+(\S+)') { $procMetrics = $Matches[1] }
            # O worker de verdade (nao o launcher wrapper) e o que tem
            # WorkingSet alto -- usa ELE como fonte de verdade pro
            # checkpoint-root/metrics ativos, senao PROGRESSO/HOLDOUT ficam
            # presos nos defaults hardcoded (bug achado 2026-07-26: mostrava
            # dados de um run antigo/morto enquanto o PID/config no topo
            # já mostrava o run novo correto).
            if ($mem -gt 100 -and $runId) {
                $liveRunId = $runId
                $liveCkptRoot = $procRoot
                $liveMetricsJsonl = $procMetrics
            }
            Write-Host "  PID $($p.ProcessId): $mem MB  |  config=$config  root=$procRoot  run_id=$runId"
        }
    }

    # -- Progresso passo-a-passo (ledger causal + dopamine, atualizam a cada step) --
    Write-Host "`n-- PROGRESSO (ao vivo, passo a passo) --" -ForegroundColor Yellow
    $ckptRootPath = if ($liveCkptRoot) { $liveCkptRoot } else { Join-Path $repoRoot $CheckpointRoot }
    $dopaminePath = Join-Path $ckptRootPath "dopamine_status.txt"
    $ledgerPath = Join-Path $ckptRootPath "causal_events.jsonl"

    if (Test-Path $ledgerPath) {
        try {
            $lastOutcome = Get-Content $ledgerPath -Tail 40 -ErrorAction SilentlyContinue |
                Where-Object { $_ -match '"event_type":"STEP_OUTCOME"' } |
                Select-Object -Last 1
            if ($lastOutcome) {
                $o = $lastOutcome | ConvertFrom-Json
                $cycle = $o.step.cycle
                $optStep = $o.step.optimizer_step
                $lm = [math]::Round($o.payload.effective_losses.lm, 4)
                $total = [math]::Round($o.payload.effective_losses.total, 4)
                $gBefore = [math]::Round($o.payload.gradient_norm_before, 3)
                $gAfter = [math]::Round($o.payload.gradient_norm_after, 3)
                $err = $o.payload.error
                $applied = $o.payload.optimizer_step_applied
                $color = if ($err) { "Red" } else { "White" }
                Write-Host "  cycle=$cycle  optimizer_step=$optStep  LM=$lm  total=$total  grad(before/after)=$gBefore/$gAfter  applied=$applied  error=$err" -ForegroundColor $color
            } else {
                Write-Host "  (aguardando primeiro STEP_OUTCOME...)"
            }
        } catch {
            Write-Host "  (falha lendo ultimo evento do ledger causal)" -ForegroundColor DarkYellow
        }
    } else {
        Write-Host "  Ledger causal nao encontrado: $ledgerPath"
    }

    if (Test-Path $dopaminePath) {
        $dop = Get-Content $dopaminePath -Tail 1 -ErrorAction SilentlyContinue
        if ($dop) { Write-Host "  dopamine: $dop" -ForegroundColor DarkCyan }
    }

    # -- Metricas de fim de ciclo / holdout (so aparecem apos 500 passos) --
    Write-Host "`n-- HOLDOUT / FIM DE CICLO --" -ForegroundColor Yellow
    $metricsPath = if ($liveMetricsJsonl) { $liveMetricsJsonl } else { Join-Path $repoRoot $MetricsJsonl }
    if (Test-Path $metricsPath) {
        # metrics.jsonl e um arquivo de nome fixo reaproveitado entre lancamentos --
        # pode conter entradas de um run_id anterior/descartado. Filtra pelo run_id
        # do processo vivo detectado acima pra nao misturar dados de runs diferentes.
        $rawLines = Get-Content $metricsPath -Tail 200 -ErrorAction SilentlyContinue
        $lines = @()
        if ($rawLines) {
            if ($liveRunId) {
                $lines = @($rawLines | Where-Object { $_ -match [regex]::Escape($liveRunId) } | Select-Object -Last 5)
                if ($lines.Count -eq 0) {
                    Write-Host "  (sem entradas ainda pro run_id ativo '$liveRunId' -- aguardando fim do primeiro ciclo)" -ForegroundColor DarkGray
                }
            } else {
                $lines = @($rawLines | Select-Object -Last 5)
            }
        }
        if ($lines.Count -gt 0) {
            foreach ($line in $lines) {
                try {
                    $m = $line | ConvertFrom-Json
                    $step = $m.step
                    $cycle = $m.cycle
                    $phase = $m.phase
                    $lm = [math]::Round($m.metrics.fresh.lm_loss, 4)
                    $total = [math]::Round($m.metrics.fresh.total_loss, 4)
                    $tps = [math]::Round($m.metrics.tokens_per_sec, 0)
                    $tok = $m.metrics.tokens_processed
                    $tokM = [math]::Round($tok / 1000000, 2)
                    $skipped = $m.metrics.skipped_updates
                    $updates = $m.metrics.successful_updates

                    $color = "White"
                    if ($phase -eq "cycle_end") { $color = "Green" }
                    Write-Host "  c$cycle s$step  LM=$lm  total=$total  ${tps}tok/s  ${tokM}M tok  skip=$skipped  upd=$updates  $phase" -ForegroundColor $color

                    if ($m.heldout -and $m.heldout.lm_loss) {
                        $hppl = [math]::Round($m.heldout.ppl, 0)
                        $hlm = [math]::Round($m.heldout.lm_loss, 4)
                        Write-Host "    |_ HOLDOUT: lm=$hlm  ppl=$hppl" -ForegroundColor Magenta
                    }
                } catch {
                    # Ignore json parsing errors on partial lines
                }
            }
        } else {
            Write-Host "  (aguardando fim do primeiro ciclo de 500 passos...)"
        }
    } else {
        Write-Host "  (ainda nao existe -- so e criado no fim do primeiro ciclo de 500 passos)"
    }

    # -- Halted? --
    $halted = Test-Path (Join-Path $repoRoot "TRAINING_HALTED.flag")
    if ($halted) {
        Write-Host "`n  [HALTED] TRAINING_HALTED.flag presente!" -ForegroundColor Red -BackgroundColor Black
    }

    Write-Host "`n----------------------------------------" -ForegroundColor DarkGray
    Start-Sleep -Seconds $Interval
}
