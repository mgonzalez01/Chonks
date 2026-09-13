<#
.SYNOPSIS
  Run Chonks as a shared host on this Windows machine: backend (loopback) +
  MCP proxy in HTTP mode (LAN), so other machines connect with ONE command and
  install nothing.

.DESCRIPTION
  What the other machines run afterwards (nothing else):
    claude mcp add --transport http chonks http://<this-host>:11439/mcp

  This script:
    1. builds mcp-server\dist if missing (needs node >= 18)
    2. starts the MCP proxy on 0.0.0.0:<Port>; the proxy spawns the Python
       backend on 127.0.0.1:11438 itself (managed mode) and restarts it if it dies.

  Indexing stays a separate, native step (uv run chonks index ...) — this
  script only serves an existing DB.

  Default -BindHost is 0.0.0.0 with no authentication: trusted networks only.

.EXAMPLE
  .\scripts\host.ps1 -Db C:\src\.db\chonks.db -Config .\config.json
.EXAMPLE
  # also start the embedder (CUDA build of llama-server)
  .\scripts\host.ps1 -Db C:\src\.db\chonks.db -Config .\config.json `
    -LlamaServer C:\llama\llama-server.exe -EmbedModel jinaai/jina-code-embeddings-0.5b-GGUF
#>
param(
  [string]$Db     = ".db\chonks.db",
  [string]$Config = "config.json",
  [int]   $Port   = 11439,
  [string]$BindHost = "0.0.0.0",
  # Embedder (llama-server). Semantic search needs it at query time, not just at
  # index time. Either start it here (-LlamaServer + -EmbedModel) or run it
  # yourself and this script just checks it answers on -EmbedPort.
  [string]$LlamaServer = "",          # path to llama-server.exe; empty = don't start one
  [string]$EmbedModel  = "",          # --hf-repo value (e.g. jinaai/jina-code-embeddings-0.5b-GGUF) or a local .gguf path
  [int]   $EmbedPort   = 11437,
  [string]$LlamaArgs   = "--parallel 16 --ctx-size 65536 -ngl 99"   # extra llama-server flags
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

foreach ($tool in @("node", "uv")) {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    throw "$tool not found on PATH — install it first (node >= 18, uv)."
  }
}
if (-not (Test-Path $Db)) { throw "DB not found: $Db — index first (uv run chonks index ...)." }
$Db = (Resolve-Path $Db).Path
$ConfigAbs = if (Test-Path $Config) { (Resolve-Path $Config).Path } else { "" }

if (-not (Test-Path "mcp-server\dist\index.js")) {
  Write-Host "[host] building mcp-server ..."
  Push-Location mcp-server
  npm install --no-audit --no-fund | Out-Null
  npm run build | Out-Null
  Pop-Location
}

$llama = $null
function Test-Embedder { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$EmbedPort/health" | Out-Null; $true } catch { $false } }
if ($LlamaServer) {
  if (-not $EmbedModel) { throw "-EmbedModel is required with -LlamaServer" }
  $modelArg = if (Test-Path $EmbedModel) { "-m `"$EmbedModel`"" } else { "--hf-repo $EmbedModel" }
  $largs = "$modelArg --embedding --pooling last --port $EmbedPort --host 127.0.0.1 $LlamaArgs"
  Write-Host "[host] starting embedder: $LlamaServer $largs"
  $llama = Start-Process -FilePath $LlamaServer -ArgumentList $largs -PassThru -NoNewWindow
  $deadline = (Get-Date).AddMinutes(5)   # first run downloads the model
  while (-not (Test-Embedder)) {
    if ($llama.HasExited) { throw "llama-server exited (code $($llama.ExitCode))" }
    if ((Get-Date) -gt $deadline) { throw "embedder did not answer on :$EmbedPort within 5 min" }
    Start-Sleep -Seconds 2
  }
  Write-Host "[host] embedder ready on :$EmbedPort"
} elseif (-not (Test-Embedder)) {
  Write-Warning "no embedder answering on http://127.0.0.1:$EmbedPort — semantic/hybrid queries will fail (fts/regex/graph tools still work). Start llama-server or pass -LlamaServer/-EmbedModel."
}

$env:CHONKS_MCP_HTTP_PORT = "$Port"
$env:CHONKS_MCP_HOST      = $BindHost
# Host-header allowlist for the adapter. Default to this machine's hostname, which is
# the name the printed `claude mcp add` line uses; set CHONKS_MCP_ALLOWED_HOSTS yourself
# to add an IP or a second name.
if (-not $env:CHONKS_MCP_ALLOWED_HOSTS) { $env:CHONKS_MCP_ALLOWED_HOSTS = [System.Net.Dns]::GetHostName() }
$env:CHONKS_URL           = "http://127.0.0.1:11438"
$env:CHONKS_SERVER_PY     = (Join-Path $Root "chonks\server.py")
$env:CHONKS_DB            = $Db
$env:CHONKS_PY_CMD        = "uv run python"
if ($ConfigAbs) { $env:CHONKS_CONFIG = $ConfigAbs }

$hostname = [System.Net.Dns]::GetHostName()
Write-Host ""
Write-Host "[host] Chonks MCP (HTTP) on http://${hostname}:${Port}/mcp"
Write-Host "[host] other machines run:"
Write-Host "  claude mcp add --transport http chonks http://${hostname}:${Port}/mcp"
Write-Host "[host] no authentication: anyone who can reach port $Port can query the index. Trusted network only."
Write-Host ""
Write-Host "[host] if the port is blocked, once (admin): New-NetFirewallRule -DisplayName 'Chonks MCP' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow"
Write-Host ""

try {
  node "mcp-server\dist\index.js"
} finally {
  if ($llama -and -not $llama.HasExited) { Write-Host "[host] stopping embedder"; Stop-Process -Id $llama.Id -Force }
}
