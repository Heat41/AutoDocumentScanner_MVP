$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "======================================"
Write-Host " AutoDocumentScanner - Windows Build"
Write-Host "======================================"

Write-Host "`n[1/6] Python"
python --version

Write-Host "`n[2/6] Install runtime dependencies"
python -m pip install -r requirements.txt

Write-Host "`n[3/6] Install build dependency"
python -m pip install -r requirements-build.txt

Write-Host "`n[4/6] Run regression tests"
python -m unittest discover -s tests -v

Write-Host "`n[5/6] Clean old build output"
if (Test-Path build) {
    Remove-Item build -Recurse -Force
}
if (Test-Path dist) {
    Remove-Item dist -Recurse -Force
}

Write-Host "`n[6/6] Build Windows application"
python -m PyInstaller --noconfirm --clean AutoDocumentScanner.spec

$PackageDir = Join-Path $Root "dist\AutoDocumentScanner"
$ExePath = Join-Path $PackageDir "AutoDocumentScanner.exe"

if (-not (Test-Path $ExePath)) {
    throw "Build selesai tetapi AutoDocumentScanner.exe tidak ditemukan."
}

New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "output") | Out-Null

Write-Host "`n======================================"
Write-Host " BUILD BERHASIL"
Write-Host "======================================"
Write-Host "Folder : $PackageDir"
Write-Host "EXE    : $ExePath"
Write-Host "`nJalankan AutoDocumentScanner.exe dari folder tersebut untuk UAT packaging."
