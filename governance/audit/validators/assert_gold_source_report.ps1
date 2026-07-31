[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ReportPath,
    [Parameter(Mandatory)][string]$SchemaPath,
    [Parameter(Mandatory)][string]$ExpectedRoot,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedSourceCommit
)

$ErrorActionPreference = "Stop"

$expectedGateIds = @(
    "pytest",
    "py_compile",
    "powershell_parse",
    "check_docs_links",
    "check_canonical_docs",
    "check_duplicates",
    "check_dependency_policy",
    "check_operational_surface",
    "check_physical_hygiene",
    "check_architecture_boundaries",
    "check_distribution",
    "secret_scan",
    "git_diff_check",
    "tracked_worktree",
    "source_report_commit"
)

function Assert-ReportInvariant {
    param([Parameter(Mandatory)][bool]$Condition, [Parameter(Mandatory)][string]$Message)
    if (-not $Condition) { throw "gold_source_report_semantic_error:$Message" }
}

function Assert-ExactSequence {
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Actual,
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Expected,
        [Parameter(Mandatory)][string]$Name
    )
    Assert-ReportInvariant ($Actual.Count -eq $Expected.Count) "$Name`:count"
    for ($index = 0; $index -lt $Expected.Count; $index++) {
        Assert-ReportInvariant ([string]$Actual[$index] -ceq [string]$Expected[$index]) "$Name`:index:$index"
    }
}

function Assert-ExactPropertyNames {
    param(
        [Parameter(Mandatory)]$Object,
        [Parameter(Mandatory)][string[]]$Expected,
        [Parameter(Mandatory)][string]$Name
    )
    $actual = @($Object.PSObject.Properties.Name | Sort-Object)
    $sortedExpected = @($Expected | Sort-Object)
    Assert-ExactSequence -Actual $actual -Expected $sortedExpected -Name "$Name`:properties"
}

$ReportPath = (Resolve-Path -LiteralPath $ReportPath).Path
$SchemaPath = (Resolve-Path -LiteralPath $SchemaPath).Path
$ExpectedRoot = (Resolve-Path -LiteralPath $ExpectedRoot).Path

if (Get-Command Test-Json -ErrorAction SilentlyContinue) {
    $schemaValid = Test-Json -LiteralPath $ReportPath -SchemaFile $SchemaPath -ErrorAction Stop
    Assert-ReportInvariant $schemaValid "schema"
}

$report = Get-Content -LiteralPath $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json -ErrorAction Stop
$reportProperties = @(
    "schema_version", "audit_kind", "source_commit", "report_source_commit",
    "observed_end_commit", "started_utc", "ended_utc", "root", "tools",
    "commands", "counts", "python_compilation", "tracked_worktree_clean",
    "status", "failures"
)
Assert-ExactPropertyNames -Object $report -Expected $reportProperties -Name "report"
Assert-ExactPropertyNames -Object $report.tools -Expected @("powershell", "git", "python") -Name "tools"
Assert-ExactPropertyNames -Object $report.counts -Expected @("total", "passed", "failed") -Name "counts"
Assert-ExactPropertyNames -Object $report.python_compilation -Expected @(
    "tracked", "attempted", "compiled", "skipped_archive", "failed"
) -Name "python_compilation"
$commands = @($report.commands)
foreach ($command in $commands) {
    Assert-ExactPropertyNames -Object $command -Expected @(
        "id", "command", "started_utc", "ended_utc", "exit_code", "output_tail"
    ) -Name "gate:$($command.id)"
}
$actualGateIds = @($commands | ForEach-Object { [string]$_.id })
Assert-ExactSequence -Actual $actualGateIds -Expected $expectedGateIds -Name "gate_ids"

Assert-ReportInvariant ([System.IO.Path]::IsPathRooted([string]$report.root)) "root:not_absolute"
$reportedRoot = [System.IO.Path]::GetFullPath([string]$report.root).TrimEnd([char[]]@('\', '/'))
$normalizedExpectedRoot = [System.IO.Path]::GetFullPath($ExpectedRoot).TrimEnd([char[]]@('\', '/'))
Assert-ReportInvariant (
    [System.StringComparer]::OrdinalIgnoreCase.Equals($reportedRoot, $normalizedExpectedRoot)
) "root:mismatch"

Assert-ReportInvariant ([string]$report.source_commit -ceq $ExpectedSourceCommit) "source_commit:unexpected"
Assert-ReportInvariant ([string]$report.report_source_commit -ceq $ExpectedSourceCommit) "report_source_commit:mismatch"

$reportStarted = [DateTimeOffset]::Parse([string]$report.started_utc)
$reportEnded = [DateTimeOffset]::Parse([string]$report.ended_utc)
Assert-ReportInvariant ($reportStarted -le $reportEnded) "timestamps:report_order"
foreach ($command in $commands) {
    $gateStarted = [DateTimeOffset]::Parse([string]$command.started_utc)
    $gateEnded = [DateTimeOffset]::Parse([string]$command.ended_utc)
    Assert-ReportInvariant ($gateStarted -le $gateEnded) "timestamps:$($command.id):order"
    Assert-ReportInvariant ($gateStarted -ge $reportStarted) "timestamps:$($command.id):before_report"
    Assert-ReportInvariant ($gateEnded -le $reportEnded) "timestamps:$($command.id):after_report"
}

$expectedFailures = @(
    $commands |
        Where-Object { [int]$_.exit_code -ne 0 } |
        ForEach-Object { [string]$_.id }
)
$actualFailures = @($report.failures | ForEach-Object { [string]$_ })
Assert-ExactSequence -Actual $actualFailures -Expected $expectedFailures -Name "failures"

$passedCount = @($commands | Where-Object { [int]$_.exit_code -eq 0 }).Count
$failedCount = $expectedFailures.Count
Assert-ReportInvariant ([int]$report.counts.total -eq $commands.Count) "counts:total"
Assert-ReportInvariant ([int]$report.counts.passed -eq $passedCount) "counts:passed"
Assert-ReportInvariant ([int]$report.counts.failed -eq $failedCount) "counts:failed"
Assert-ReportInvariant (
    [int]$report.counts.total -eq ([int]$report.counts.passed + [int]$report.counts.failed)
) "counts:sum"

$expectedStatus = if ($failedCount -eq 0) { "pass" } else { "fail" }
Assert-ReportInvariant ([string]$report.status -ceq $expectedStatus) "status"

$trackedGate = $commands | Where-Object { $_.id -ceq "tracked_worktree" }
$expectedTrackedClean = [int]$trackedGate.exit_code -eq 0
Assert-ReportInvariant (
    [bool]$report.tracked_worktree_clean -eq $expectedTrackedClean
) "tracked_worktree_clean"

$sourceGate = $commands | Where-Object { $_.id -ceq "source_report_commit" }
$endMatchesSource = [string]$report.observed_end_commit -ceq $ExpectedSourceCommit
Assert-ReportInvariant (
    ([int]$sourceGate.exit_code -eq 0) -eq $endMatchesSource
) "observed_end_commit:gate"

$compilation = $report.python_compilation
$trackedPython = [int]$compilation.tracked
$attemptedPython = [int]$compilation.attempted
$compiledPython = [int]$compilation.compiled
$skippedPython = [int]$compilation.skipped_archive
$failedPython = [int]$compilation.failed
Assert-ReportInvariant ($trackedPython -eq ($attemptedPython + $skippedPython)) "python_compilation:tracked"
Assert-ReportInvariant ($attemptedPython -eq ($compiledPython + $failedPython)) "python_compilation:attempted"
$compileGate = $commands | Where-Object { $_.id -ceq "py_compile" }
Assert-ReportInvariant (
    ([int]$compileGate.exit_code -eq 0) -eq ($failedPython -eq 0)
) "python_compilation:gate"

Write-Output "valid"
