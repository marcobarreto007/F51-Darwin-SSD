param(
    [string]$Hub = "root@70.30.158.46",
    [int]$Port = 56013,
    [int64]$ExpectedCheckpointBytes = 21185819745,
    [switch]$Json
)

$ErrorActionPreference = "Stop"

function Invoke-Hub {
    param([Parameter(Mandatory = $true)][string]$Command)

    $name = "nucleo_ready_" + ([Guid]::NewGuid().ToString("N")) + ".sh"
    $localScript = Join-Path ([System.IO.Path]::GetTempPath()) $name
    $remoteScript = "/tmp/$name"
    Set-Content -Path $localScript -Value $Command -Encoding ascii

    $baseSshArgs = @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", [string]$Port,
        $Hub
    )
    $scpArgs = @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-P", [string]$Port,
        $localScript,
        "${Hub}:$remoteScript"
    )
    $runArgs = $baseSshArgs + @("bash $remoteScript")
    $cleanupArgs = $baseSshArgs + @("rm -f $remoteScript")

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & scp @scpArgs 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            return
        }
        & ssh @runArgs 2>$null
    } finally {
        & ssh @cleanupArgs 2>$null | Out-Null
        Remove-Item -LiteralPath $localScript -Force -ErrorAction SilentlyContinue
        $ErrorActionPreference = $previousPreference
    }
}

function Last-Matching-Line {
    param(
        [string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Pattern
    )
    $matches = @($Lines | Where-Object { $_ -match $Pattern })
    if ($matches.Count -eq 0) {
        return ""
    }
    return $matches[-1].Trim()
}

function Add-Check {
    param(
        [System.Collections.Generic.List[object]]$Checks,
        [string]$Name,
        [bool]$Ok,
        [string]$Detail
    )
    $Checks.Add([PSCustomObject]@{
        name = $Name
        ok = $Ok
        detail = $Detail
    }) | Out-Null
}

$checks = [System.Collections.Generic.List[object]]::new()

$tokenProgressCommand = @'
python3 - <<'PY'
import re
from pathlib import Path
p = Path('/workspace/tokens/tokenize_v2.log')
if p.exists():
    text = p.read_text(errors='replace')
    lines = [line.strip() for line in re.split(r'[\r\n]+', text) if re.search(r'[0-9]+/[0-9]+ files|DONE|done', line)]
    if lines:
        print(lines[-1])
PY
'@
$tokenLines = @(Invoke-Hub $tokenProgressCommand)
$tokenLine = Last-Matching-Line -Lines $tokenLines -Pattern "([0-9]+/[0-9]+ files|DONE|done)"
$tokenDone = $false
if ($tokenLine -match "([0-9]+)/([0-9]+) files") {
    $tokenDone = ([int]$matches[1] -ge [int]$matches[2])
} elseif ($tokenLine -match "DONE|done") {
    $tokenDone = $true
}

$activeTokenizerCommand = @'
ps -eo pid,ppid,cmd | grep '[t]okenize_v2.py' | grep -v 'Wait for tokenization' | wc -l
'@
$activeTokenizerLines = @(Invoke-Hub $activeTokenizerCommand)
$activeTokenizerLine = Last-Matching-Line -Lines $activeTokenizerLines -Pattern "^[0-9]+$"
$activeTokenizerCount = 0
if ($activeTokenizerLine -match "^[0-9]+$") {
    $activeTokenizerCount = [int]$activeTokenizerLine
}

$tokenFileCommand = @'
python3 - <<'PY'
from pathlib import Path
p=Path('/workspace/tokens/tokens_full.bin')
if not p.exists():
    print('missing')
else:
    size=p.stat().st_size
    print(f'size={size} mod4={size % 4} tokens={size // 4}')
PY
'@
$tokenFileLines = @(Invoke-Hub $tokenFileCommand)
$tokenFileLine = Last-Matching-Line -Lines $tokenFileLines -Pattern "^(size=|missing)"
$tokenFileOk = $tokenFileLine -match "^size=([1-9][0-9]*) mod4=0 tokens=([1-9][0-9]*)"
Add-Check -Checks $checks -Name "main_token_file_int32" -Ok $tokenFileOk -Detail $tokenFileLine

if (-not $tokenDone -and $tokenFileOk -and $activeTokenizerCount -eq 0) {
    $tokenDone = $true
    $tokenLine = "artifact_ready; last_log=$tokenLine; active_tokenizers=0; $tokenFileLine"
} else {
    $tokenLine = "last_log=$tokenLine; active_tokenizers=$activeTokenizerCount; $tokenFileLine"
}
Add-Check -Checks $checks -Name "main_tokenization_complete" -Ok $tokenDone -Detail $tokenLine

$ckptLines = @(Invoke-Hub "stat -c%s /workspace/checkpoints/step_0007900_moe.pt 2>/dev/null || true")
$ckptBytesLine = Last-Matching-Line -Lines $ckptLines -Pattern "^[0-9]+$"
$ckptBytes = 0L
if ($ckptBytesLine -match "^[0-9]+$") {
    $ckptBytes = [int64]$ckptBytesLine
}
$ckptGB = [Math]::Round($ckptBytes / 1GB, 2)
$ckptDone = $ckptBytes -ge $ExpectedCheckpointBytes
Add-Check -Checks $checks -Name "checkpoint_uploaded" -Ok $ckptDone -Detail "bytes=$ckptBytes gib=$ckptGB expected_bytes=$ExpectedCheckpointBytes"

$tokenizerLines = @(Invoke-Hub "test -s /workspace/tokenizer/f51_bpe/vocab.json -o -s /workspace/f51_corpus_factory/tokenizer/f51_bpe_80k/vocab.json; echo `$?")
$tokenizerLine = Last-Matching-Line -Lines $tokenizerLines -Pattern "^[0-9]+$"
Add-Check -Checks $checks -Name "tokenizer_present" -Ok ($tokenizerLine -eq "0") -Detail "exit=$tokenizerLine"

$importLines = @(Invoke-Hub "cd /workspace && python3 -c `"import f51_darwin; print('import_ok')`"")
$importLine = Last-Matching-Line -Lines $importLines -Pattern "import_ok"
Add-Check -Checks $checks -Name "workspace_import" -Ok ($importLine -eq "import_ok") -Detail $importLine

$indexCommand = @'
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/workspace/f51_corpus_factory/tokens/80k/index.json')
if not p.exists():
    print('missing')
else:
    data=json.loads(p.read_text())
    print(f"batch_count={data.get('batch_count')} total_tokens={data.get('total_tokens')} total_bytes={data.get('total_bytes')}")
PY
'@
$indexLines = @(Invoke-Hub $indexCommand)
$indexLine = Last-Matching-Line -Lines $indexLines -Pattern "^(batch_count=|missing)"
$indexOk = $indexLine -match "^batch_count=([1-9][0-9]*) total_tokens=([1-9][0-9]*) total_bytes=([1-9][0-9]*)"
Add-Check -Checks $checks -Name "dataset_index_present" -Ok $indexOk -Detail $indexLine

$allOk = -not @($checks | Where-Object { -not $_.ok })
$payload = [PSCustomObject]@{
    ok = $allOk
    utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    state = $(if ($allOk) { "PRONTO" } else { "MONTAGEM" })
    checks = $checks
}

if ($Json) {
    $payload | ConvertTo-Json -Depth 5
} else {
    $payload
}

if (-not $allOk) {
    exit 1
}
