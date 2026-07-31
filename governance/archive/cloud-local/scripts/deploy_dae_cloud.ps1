[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$SshHost,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$Port,
    [ValidateSet('Validate', 'Sync', 'Canary', 'Launch', 'PullCheckpoint')]
    [string]$Mode = 'Validate',
    [string]$KeyPath = ([IO.Path]::Combine(
        [Environment]::GetFolderPath('UserProfile'), '.ssh', 'id_ed25519'
    )),
    [string]$RemoteRoot = '/workspace',
    [string]$LocalCheckpointDir = '',
    [string]$RunId = ('dae-v1-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$BlockSize = 256,
    [int]$AccumSteps = 16,
    [double]$LearningRate = 0.00005,
    [int]$WarmupSteps = 100,
    [double]$MaxHeldoutRegression = 0.02
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$corpusName = '00_CORPUS_PRINCIPAL_tokens_feast_v2.bin'
$corpusPath = "$RemoteRoot/01_TOKENIZADOS/$corpusName"
$corpusBytes = 74195890756
$corpusSha256 = '9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff'
$remoteConfig = "$RemoteRoot/configs/darwin_x_1.6b_dae_cloud.yaml"
$remoteCanaryRoot = "$RemoteRoot/03_CHECKPOINTS/canary_dae_v1_$RunId"
$remoteMetrics = "$RemoteRoot/runs/dae_v1/$RunId.jsonl"
$remoteMetadata = "$RemoteRoot/runs/dae_v1/$RunId.metadata.json"
$remoteDecision = "$RemoteRoot/runs/dae_v1/$RunId.decision.json"
$sourceCommit = (& git -C $repo rev-parse HEAD).Trim().ToLowerInvariant()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}([0-9a-f]{24})?$') {
    throw 'Could not resolve the source Git commit'
}

if (-not (Test-Path -LiteralPath $KeyPath -PathType Leaf)) {
    throw "SSH key does not exist: $KeyPath"
}

function Invoke-Remote {
    param([Parameter(Mandatory = $true)][string]$Command)
    & ssh -i $KeyPath -p $Port -o BatchMode=yes -o ConnectTimeout=15 -- $SshHost $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Copy-ToRemote {
    param(
        [Parameter(Mandatory = $true)][string]$Local,
        [Parameter(Mandatory = $true)][string]$Remote
    )
    & scp -i $KeyPath -P $Port -o BatchMode=yes -o ConnectTimeout=15 -- $Local "${SshHost}:$Remote"
    if ($LASTEXITCODE -ne 0) {
        throw "SCP upload failed with exit code $LASTEXITCODE"
    }
}

function Assert-RemoteReady {
    $template = @'
set -e
f='__CORPUS__'
test "$(stat -c %s "$f")" -eq __BYTES__
got=$(sha256sum "$f" | awk '{print $1}')
test "$got" = '__SHA__'
if pgrep -af '[d]arwin_organism.py.*(run247|cycle)' >/tmp/f51-active.txt; then
  cat /tmp/f51-active.txt
  exit 19
fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
echo REMOTE_READY=$(stat -c %s "$f")'|'$got
'@
    $command = $template.Replace('__CORPUS__', $corpusPath).
        Replace('__BYTES__', [string]$corpusBytes).
        Replace('__SHA__', $corpusSha256)
    Invoke-Remote -Command $command
}

function Sync-Code {
    Push-Location $repo
    try {
        & git diff --quiet --
        if ($LASTEXITCODE -ne 0) { throw 'Tracked worktree is dirty' }
        & git diff --cached --quiet --
        if ($LASTEXITCODE -ne 0) { throw 'Index contains uncommitted changes' }
        $archive = Join-Path ([IO.Path]::GetTempPath()) "$RunId-code.tar"
        $archiveRoots = @('f51_darwin', 'scripts', 'configs')
        & git cat-file -e 'HEAD:tokenizer' 2>$null
        if ($LASTEXITCODE -eq 0) { $archiveRoots += 'tokenizer' }
        & git archive --format=tar --output=$archive HEAD @archiveRoots
        if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }
        try {
            Copy-ToRemote -Local $archive -Remote "$RemoteRoot/$RunId-code.tar.partial"
            $extract = "set -e; cd '$RemoteRoot'; tar -xf '$RunId-code.tar.partial'; rm -f '$RunId-code.tar.partial'; test -d tokenizer/f51_bpe_80k; python3 -m py_compile f51_darwin/darwin_x.py f51_darwin/dae_optimizer.py scripts/darwin_organism.py; echo CODE_SYNC_OK"
            Invoke-Remote -Command $extract
        }
        finally {
            Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        }
    }
    finally {
        Pop-Location
    }
}

function Write-AndUploadMetadata {
    Push-Location $repo
    try {
        $configPath = Join-Path $repo 'configs/darwin_x_1.6b_dae_cloud.yaml'
        $metadata = [ordered]@{
            schema_version = 1
            run_id = $RunId
            git_commit = $sourceCommit
            command = 'darwin_organism.py cycle --canary --fresh-start'
            config_identity = (Get-FileHash -Algorithm SHA256 -LiteralPath $configPath).Hash.ToLowerInvariant()
            base_checkpoint_sha256 = $null
            corpus_sha256 = $corpusSha256
            corpus_bytes = $corpusBytes
            optimizer = 'dae_hybrid'
        }
        $temp = Join-Path ([IO.Path]::GetTempPath()) "$RunId.metadata.json"
        $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temp -Encoding utf8NoBOM
        try {
            Invoke-Remote -Command "mkdir -p '$RemoteRoot/runs/dae_v1'"
            Copy-ToRemote -Local $temp -Remote "$remoteMetadata.partial"
            Invoke-Remote -Command "mv -f '$remoteMetadata.partial' '$remoteMetadata'"
        }
        finally {
            Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
        }
    }
    finally {
        Pop-Location
    }
}

switch ($Mode) {
    'Validate' {
        Assert-RemoteReady
    }
    'Sync' {
        Assert-RemoteReady
        Sync-Code
    }
    'Canary' {
        Assert-RemoteReady
        Sync-Code
        Write-AndUploadMetadata
        $command = "set -euo pipefail; cd '$RemoteRoot'; mkdir -p '$remoteCanaryRoot' '$RemoteRoot/runs/dae_v1'; rm -f '$remoteMetrics' '$remoteDecision'; export F51_DATASET_ROOT='$RemoteRoot'; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python3 -u scripts/darwin_organism.py cycle --canary --fresh-start --config '$remoteConfig' --optimizer dae_hybrid --device cuda --steps 250 --cycles 1 --block-size $BlockSize --batch-size 1 --accum-steps $AccumSteps --lr $LearningRate --warmup-steps $WarmupSteps --eval-every 50 --token-bin '$corpusPath' --checkpoint-root '$remoteCanaryRoot' --metrics-jsonl '$remoteMetrics' --canary-metadata-json '$remoteMetadata' --run-id '$RunId' --source-git-commit '$sourceCommit' --holdout-tokens 262144 --holdout-batches 16 2>&1 | tee '$RemoteRoot/runs/dae_v1/$RunId.log'"
        Invoke-Remote -Command $command
        $pointer = "$remoteCanaryRoot/organism_latest.json"
        $gate = "cd '$RemoteRoot'; python3 scripts/evaluate_dae_cloud_canary.py --metrics '$remoteMetrics' --pointer '$pointer' --decision '$remoteDecision' --run-id '$RunId' --expected-microbatches 250 --accum-steps $AccumSteps --expected-git-commit '$sourceCommit' --expected-corpus-sha256 '$corpusSha256' --max-heldout-regression $MaxHeldoutRegression"
        Invoke-Remote -Command $gate
    }
    'Launch' {
        Assert-RemoteReady
        $preflight = "python3 -c `"import json; d=json.load(open('$remoteDecision')); assert d['passed'] is True and d['status']=='runtime_canary_passed' and d['run_id']=='$RunId' and d['git_commit']=='$sourceCommit' and d['corpus_sha256']=='$corpusSha256'`""
        Invoke-Remote -Command $preflight
        $command = "set -e; cd '$RemoteRoot'; checkpoint=`$(python3 -c `"import json; print(json.load(open('$remoteDecision'))['checkpoint'])`" ); sha=`$(python3 -c `"import json; print(json.load(open('$remoteDecision'))['checkpoint_sha256'])`" ); test -s `"`$checkpoint`"; export F51_DATASET_ROOT='$RemoteRoot'; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; nohup python3 -u scripts/darwin_organism.py run247 --resume `"`$checkpoint`" --base-checkpoint-sha256 `"`$sha`" --config '$remoteConfig' --optimizer dae_hybrid --device cuda --steps 2000 --block-size $BlockSize --batch-size 1 --accum-steps $AccumSteps --lr $LearningRate --warmup-steps $WarmupSteps --eval-every 500 --token-bin '$corpusPath' > '$RemoteRoot/runs/dae_v1/$RunId-run247.log' 2>&1 < /dev/null & echo RUN247_PID=`$!"
        Invoke-Remote -Command $command
    }
    'PullCheckpoint' {
        if (-not $LocalCheckpointDir) {
            throw '-LocalCheckpointDir is required for PullCheckpoint'
        }
        New-Item -ItemType Directory -Force -Path $LocalCheckpointDir | Out-Null
        $pointerTemplate = @'
python3 -c 'import json; p=json.load(open("__POINTER__")); print(p["path"])'
'@
        $pointerCommand = $pointerTemplate.Replace(
            '__POINTER__',
            "$RemoteRoot/03_CHECKPOINTS/organism_latest.json"
        )
        $remoteCheckpoint = (& ssh -i $KeyPath -p $Port -o BatchMode=yes -- $SshHost $pointerCommand).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $remoteCheckpoint) {
            throw 'Could not resolve remote checkpoint pointer'
        }
        & scp -i $KeyPath -P $Port -o BatchMode=yes -- "${SshHost}:$RemoteRoot/03_CHECKPOINTS/$remoteCheckpoint" $LocalCheckpointDir
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint download failed' }
        & scp -i $KeyPath -P $Port -o BatchMode=yes -- "${SshHost}:$RemoteRoot/03_CHECKPOINTS/organism_latest.json" $LocalCheckpointDir
        if ($LASTEXITCODE -ne 0) { throw 'Pointer download failed' }
    }
}
