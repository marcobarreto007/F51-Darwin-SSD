[CmdletBinding()]
param(
    [string]$Root = "",
    [string]$OutputPath = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }
$Root = (Resolve-Path -LiteralPath $Root).Path
$started = (Get-Date).ToUniversalTime()
$previousCuda = [Environment]::GetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "Process")
$previousPycache = [Environment]::GetEnvironmentVariable("PYTHONPYCACHEPREFIX", "Process")
$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")

function Get-RedactedTail {
    param([string]$Text)
    if (-not $Text) { return "" }
    $redacted = $Text
    $patterns = @(
        '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'AKIA[0-9A-Z]{16}',
        'gh[pousr]_[A-Za-z0-9]{30,}',
        'sk-[A-Za-z0-9]{32,}',
        'xox[baprs]-[A-Za-z0-9-]{20,}',
        'AIza[0-9A-Za-z_-]{35}'
    )
    foreach ($pattern in $patterns) { $redacted = $redacted -replace $pattern, '[REDACTED]' }
    if ($redacted.Length -gt 4000) { return $redacted.Substring($redacted.Length - 4000) }
    return $redacted
}

function Publish-ValidatedJsonAtomic {
    param(
        [Parameter(Mandatory)]$Payload,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$SchemaPath,
        [Parameter(Mandatory)][string]$ValidatorPath,
        [Parameter(Mandatory)][string]$ExpectedRoot,
        [Parameter(Mandatory)][string]$ExpectedSourceCommit
    )
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $temporary = "$Path.tmp-$PID-$([guid]::NewGuid().ToString('N'))"
    try {
        $json = $Payload | ConvertTo-Json -Depth 12
        [System.IO.File]::WriteAllText(
            $temporary,
            $json,
            [System.Text.UTF8Encoding]::new($false)
        )
        & $ValidatorPath `
            -ReportPath $temporary `
            -SchemaPath $SchemaPath `
            -ExpectedRoot $ExpectedRoot `
            -ExpectedSourceCommit $ExpectedSourceCommit | Out-Null
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Invoke-ExternalGate {
    param(
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    $gateStart = (Get-Date).ToUniversalTime()
    $exitCode = 99
    $output = ""
    try {
        $output = (& $FilePath @Arguments 2>&1 | Out-String)
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    } catch {
        $output = $_.Exception.Message
        $exitCode = 99
    }
    return [ordered]@{
        id = $Id
        command = ((@($FilePath) + $Arguments) -join ' ')
        started_utc = $gateStart.ToString('o')
        ended_utc = (Get-Date).ToUniversalTime().ToString('o')
        exit_code = $exitCode
        output_tail = Get-RedactedTail -Text $output
    }
}

function New-InternalGate {
    param([string]$Id, [int]$ExitCode, [string]$Command, [string]$Output = "")
    $now = (Get-Date).ToUniversalTime().ToString('o')
    return [ordered]@{
        id = $Id
        command = $Command
        started_utc = $now
        ended_utc = $now
        exit_code = $ExitCode
        output_tail = Get-RedactedTail -Text $Output
    }
}

if (-not $OutputPath) {
    $runId = $started.ToString('yyyyMMddTHHmmssZ')
    $OutputPath = Join-Path $Root "workspace\runtime\evaluations\gold-source-$runId\report.json"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $Root $OutputPath
}

try {
    [Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "-1", "Process")
    [Environment]::SetEnvironmentVariable(
        "PYTHONPYCACHEPREFIX",
        (Join-Path $Root "workspace\runtime\cache\gold-source-$PID"),
        "Process"
    )
    $sourcePath = Join-Path $Root "src"
    $auditPythonPath = if ($previousPythonPath) {
        "$sourcePath$([System.IO.Path]::PathSeparator)$previousPythonPath"
    } else {
        $sourcePath
    }
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $auditPythonPath, "Process")
    Push-Location $Root
    try {
        $sourceCommit = (& git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}$') {
            throw "Unable to resolve a 40-character source commit."
        }

        if (-not $PythonExe) {
            $localPython = Join-Path $Root ".venv_nitro\Scripts\python.exe"
            $PythonExe = if (Test-Path -LiteralPath $localPython) { $localPython } else { (Get-Command python).Source }
        }
        $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
        $commands = [System.Collections.Generic.List[object]]::new()

        $commands.Add((Invoke-ExternalGate -Id "pytest" -FilePath $PythonExe -Arguments @("-m", "pytest", "-q", "-p", "no:cacheprovider")))
        $compileGate = Invoke-ExternalGate -Id "py_compile" -FilePath $PythonExe -Arguments @(
            (Join-Path $Root "src\\tools\\check_python_compilation.py"),
            "--root",
            $Root
        )
        $commands.Add($compileGate)
        try {
            $compileJson = @(
                ([string]$compileGate.output_tail) -split "`r?`n" |
                    Where-Object { $_.Trim().StartsWith("{") } |
                    Select-Object -Last 1
            )
            if ($compileJson.Count -ne 1) { throw "missing compilation JSON line" }
            $compileSummary = $compileJson[0] | ConvertFrom-Json -ErrorAction Stop
            $pythonCompilation = [ordered]@{
                tracked = [int]$compileSummary.tracked
                attempted = [int]$compileSummary.attempted
                compiled = [int]$compileSummary.compiled
                skipped_archive = [int]$compileSummary.skipped_archive
                failed = [int]$compileSummary.failed
            }
        } catch {
            throw "py_compile did not emit its required JSON count summary: $($compileGate.output_tail)"
        }

        $parseErrors = [System.Collections.Generic.List[string]]::new()
        $psFiles = & git ls-files '*.ps1'
        foreach ($file in $psFiles) {
            if ($file.Replace('\', '/').StartsWith('governance/archive/')) { continue }
            $tokens = $null
            $errors = $null
            [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $Root $file), [ref]$tokens, [ref]$errors) | Out-Null
            foreach ($parseError in $errors) { $parseErrors.Add("${file}:$($parseError.Message)") }
        }
        $commands.Add((New-InternalGate -Id "powershell_parse" -ExitCode $(if ($parseErrors.Count) { 2 } else { 0 }) -Command "PowerShell Parser::ParseFile tracked active ps1" -Output ($parseErrors -join "`n")))

        foreach ($checker in @(
            "check_docs_links.py",
            "check_canonical_docs.py",
            "check_duplicates.py",
            "check_dependency_policy.py",
            "check_operational_surface.py",
            "check_physical_hygiene.py",
            "check_architecture_boundaries.py",
            "check_distribution.py"
        )) {
            $id = [System.IO.Path]::GetFileNameWithoutExtension($checker)
            $commands.Add((Invoke-ExternalGate -Id $id -FilePath $PythonExe -Arguments @((Join-Path $Root "src\\tools\\$checker"), "--root", $Root)))
        }
        $commands.Add((Invoke-ExternalGate -Id "secret_scan" -FilePath $PythonExe -Arguments @((Join-Path $Root "src\\tools\\check_source_secrets.py"), "--scope", "all", "--root", $Root)))
        $commands.Add((Invoke-ExternalGate -Id "git_diff_check" -FilePath "git" -Arguments @("diff-tree", "--check", "--root", "-m", "--no-commit-id", "-r", $sourceCommit, "--")))

        $statusOutput = (& git status --porcelain --untracked-files=no | Out-String).Trim()
        $trackedClean = -not $statusOutput
        $commands.Add((New-InternalGate -Id "tracked_worktree" -ExitCode $(if ($trackedClean) { 0 } else { 2 }) -Command "git status --porcelain --untracked-files=no" -Output $statusOutput))

        $observedEndCommit = (& git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $observedEndCommit -notmatch '^[0-9a-f]{40}$') {
            throw "Unable to resolve the 40-character end commit."
        }
        $commitMatches = $observedEndCommit -ceq $sourceCommit
        $commands.Add((New-InternalGate -Id "source_report_commit" -ExitCode $(if ($commitMatches) { 0 } else { 2 }) -Command "compare frozen source commit with observed end HEAD" -Output $(if ($commitMatches) { "match" } else { "mismatch" })))

        $failures = @($commands | Where-Object { $_.exit_code -ne 0 } | ForEach-Object { $_.id })
        $pythonVersion = (& $PythonExe --version 2>&1 | Out-String).Trim()
        $report = [ordered]@{
            schema_version = 2
            audit_kind = "gold_source"
            source_commit = $sourceCommit
            report_source_commit = $sourceCommit
            observed_end_commit = $observedEndCommit
            started_utc = $started.ToString('o')
            ended_utc = (Get-Date).ToUniversalTime().ToString('o')
            root = $Root
            tools = [ordered]@{
                powershell = $PSVersionTable.PSVersion.ToString()
                git = (& git --version | Out-String).Trim()
                python = $pythonVersion
            }
            commands = @($commands)
            counts = [ordered]@{
                total = $commands.Count
                passed = @($commands | Where-Object { $_.exit_code -eq 0 }).Count
                failed = $failures.Count
            }
            python_compilation = $pythonCompilation
            tracked_worktree_clean = $trackedClean
            status = if ($failures.Count) { "fail" } else { "pass" }
            failures = $failures
        }
        Publish-ValidatedJsonAtomic `
            -Payload $report `
            -Path $OutputPath `
            -SchemaPath (Join-Path $Root "governance\\audit\\schemas\gold-source-report.schema.json") `
            -ValidatorPath (Join-Path $Root "governance\\audit\\validators\assert_gold_source_report.ps1") `
            -ExpectedRoot $Root `
            -ExpectedSourceCommit $sourceCommit
        Write-Output $OutputPath
        if ($failures.Count) { exit 2 }
        exit 0
    } finally {
        Pop-Location
    }
} finally {
    [Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", $previousCuda, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONPYCACHEPREFIX", $previousPycache, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $previousPythonPath, "Process")
}
