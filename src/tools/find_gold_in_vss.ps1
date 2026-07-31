#Requires -RunAsAdministrator
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$expectedRelative = 'Users\marco\Desktop\F51-Darwin-SSD\workspace\03_CHECKPOINTS\organism_cycle_071.pt'
$expectedBytes = 10876850383
$expectedSha256 = '239FDCF175AC35D9402D664E2A2B40252EC9C73AEA8EE458250D426ADDE9241B'

$results = foreach ($shadow in Get-CimInstance Win32_ShadowCopy) {
    $candidate = Join-Path $shadow.DeviceObject $expectedRelative
    if (-not (Test-Path -LiteralPath $candidate)) {
        continue
    }
    $item = Get-Item -LiteralPath $candidate
    $hash = $null
    if ($item.Length -eq $expectedBytes) {
        $hash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash
    }
    [pscustomobject]@{
        ShadowId = $shadow.ID
        InstallDate = $shadow.InstallDate
        Candidate = $candidate
        Bytes = $item.Length
        Sha256 = $hash
        ExactMatch = ($item.Length -eq $expectedBytes -and $hash -eq $expectedSha256)
    }
}

$results | ConvertTo-Json -Depth 4
