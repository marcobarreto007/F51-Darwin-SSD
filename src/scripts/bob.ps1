# ── F51 Bob Agent (GLM 5.2 via z.ai) ─────────────────────────────────────────
# Uso: powershell -ExecutionPolicy Bypass -File src\\scripts\\bob.ps1
#      ou no terminal: .\src\\scripts\\bob.ps1

# 1. Exigir a credencial no ambiente e limpar apenas a configuração herdada.
$anthropicApiKey = [System.Environment]::GetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    "Process"
)
if ([string]::IsNullOrWhiteSpace($anthropicApiKey)) {
    throw "ANTHROPIC_API_KEY must be set in the process environment."
}

$envVarsToClear = @(
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CONFIG_DIR", "CLAUDE_HOME",
    "API_TIMEOUT_MS"
)
foreach ($v in $envVarsToClear) {
    [System.Environment]::SetEnvironmentVariable($v, $null, "Process")
}

# 2. Setar GLM 5.2 via z.ai sem persistir a credencial no código.
# NOTA: Usa ANTHROPIC_API_KEY (não AUTH_TOKEN) — Claude Code exige API_KEY
#       para considerar autenticado. z.ai aceita via ambos os headers.
$env:ANTHROPIC_API_KEY                 = $anthropicApiKey
$env:ANTHROPIC_BASE_URL                = "https://api.z.ai/api/anthropic"
$env:ANTHROPIC_MODEL                   = "glm-5.2"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL      = "glm-5.2"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL    = "glm-5.2"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL     = "glm-5-turbo"
$env:ANTHROPIC_SMALL_FAST_MODEL        = "glm-4.5-air"
$env:CLAUDE_CODE_SUBAGENT_MODEL        = "glm-5.2"
$env:API_TIMEOUT_MS                    = "120000"

# 3. Chamar claude
& claude --dangerously-skip-permissions @args
