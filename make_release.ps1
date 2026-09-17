$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "======================================"
Write-Host " AutoDocumentScanner - Full Release"
Write-Host "======================================"

Write-Host "`n[1/2] Membuat ZIP portable"
& (Join-Path $Root "release_windows.ps1")

Write-Host "`n[2/2] Membuat installer Windows"
& (Join-Path $Root "build_installer.ps1")

Write-Host "`n======================================"
Write-Host " FULL RELEASE BERHASIL"
Write-Host "======================================"
Write-Host "Semua artefak tersedia di folder release\."
