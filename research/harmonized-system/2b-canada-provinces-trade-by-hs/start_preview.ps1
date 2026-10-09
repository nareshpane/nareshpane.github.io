# Start the dependency-free local preview in a hidden process.
$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$previewLogs = Join-Path $PSScriptRoot '.qa'
New-Item -ItemType Directory -Path $previewLogs -Force | Out-Null
$previewProcess = Start-Process -FilePath python -ArgumentList @('-m', 'http.server', '8000', '--bind', '127.0.0.1', '--directory', $repositoryRoot) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $previewLogs 'server-output.log') -RedirectStandardError (Join-Path $previewLogs 'server-error.log')
Set-Content -LiteralPath (Join-Path $previewLogs 'server.pid') -Value $previewProcess.Id
Write-Output "Preview PID: $($previewProcess.Id)"
Write-Output 'http://localhost:8000/research/harmonized-system/canada-and-provinces-trade-by-hs.html'
