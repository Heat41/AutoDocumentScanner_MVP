$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VersionFile = Join-Path $Root "VERSION.txt"
$TemplateFile = Join-Path $Root "installer\AutoDocumentScanner.iss.in"
$ReleaseRoot = Join-Path $Root "release"

if (-not (Test-Path $VersionFile)) {
    throw "VERSION.txt tidak ditemukan."
}

$Version = (Get-Content $VersionFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($Version)) {
    throw "VERSION.txt kosong."
}

if (-not (Test-Path $TemplateFile)) {
    throw "Template installer tidak ditemukan: $TemplateFile"
}

Write-Host "Menyiapkan logo aplikasi..."
python -m autodocscanner.support.branding
$LogoIco = Join-Path $Root "assets\logo.ico"
if (-not (Test-Path -LiteralPath $LogoIco -PathType Leaf)) {
    throw "Logo installer tidak berhasil dibuat: $LogoIco"
}

$PackageName = "AutoDocumentScanner-v$Version-windows-x64"
$PackageDir = Join-Path $ReleaseRoot $PackageName
$PackageExe = Join-Path $PackageDir "AutoDocumentScanner.exe"

if (-not (Test-Path $PackageExe)) {
    Write-Host "Paket release bersih belum tersedia. Membuat release ZIP terlebih dahulu..."
    & (Join-Path $Root "release_windows.ps1")
}

if (-not (Test-Path $PackageExe)) {
    throw "Paket release tidak ditemukan setelah release_windows.ps1 dijalankan."
}

# Installer harus dibangun hanya dari release yang sudah disanitasi.
foreach ($RuntimeName in @("input", "output")) {
    $RuntimeDir = Join-Path $PackageDir $RuntimeName
    if (Test-Path $RuntimeDir) {
        $RuntimeItems = @(Get-ChildItem $RuntimeDir -Force)
        if ($RuntimeItems.Count -gt 0) {
            throw "Folder $RuntimeName pada release tidak kosong. Jalankan release_windows.ps1 ulang."
        }
    }
}

$Compiler = $null

# 1) Lokasi custom dari environment variable.
if (
    $env:INNO_SETUP_COMPILER -and
    (Test-Path -LiteralPath $env:INNO_SETUP_COMPILER -PathType Leaf)
) {
    $Compiler = (Resolve-Path -LiteralPath $env:INNO_SETUP_COMPILER).Path
}

# 2) Lokasi instalasi standar Inno Setup 6.
# Gunakan foreach, bukan indexing hasil pipeline, karena satu hasil pipeline
# PowerShell dapat menjadi scalar string dan $value[0] akan menghasilkan 'C'.
if (-not $Compiler) {
    $CandidatePaths = @()

    if (${env:ProgramFiles(x86)}) {
        $CandidatePaths += Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"
    }

    if ($env:ProgramFiles) {
        $CandidatePaths += Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"
    }

    foreach ($Candidate in $CandidatePaths) {
        if (
            $Candidate -and
            (Test-Path -LiteralPath $Candidate -PathType Leaf)
        ) {
            $Compiler = (Resolve-Path -LiteralPath $Candidate).Path
            break
        }
    }
}

# 3) Fallback bila ISCC.exe sudah ada di PATH.
if (-not $Compiler) {
    $Command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($Command) {
        $Compiler = $Command.Source
    }
}

if (-not $Compiler) {
    throw @"
Inno Setup 6 tidak ditemukan.
Install Inno Setup 6 terlebih dahulu, lalu jalankan script ini lagi.
Jika ISCC.exe berada di lokasi custom, set environment variable:
  `$env:INNO_SETUP_COMPILER = 'C:\path\to\ISCC.exe'
"@
}

if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw "Compiler Inno Setup tidak valid: $Compiler"
}

$BuildDir = Join-Path $Root "build\installer"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$GeneratedIss = Join-Path $BuildDir "AutoDocumentScanner.iss"
$InstallerIcon = Join-Path $BuildDir "logo.ico"
Copy-Item -LiteralPath $LogoIco -Destination $InstallerIcon -Force

$Template = Get-Content $TemplateFile -Raw
$Generated = $Template.Replace("@@VERSION@@", $Version)
$Generated = $Generated.Replace("@@SOURCE_DIR@@", $PackageDir)
$Generated = $Generated.Replace("@@OUTPUT_DIR@@", $ReleaseRoot)
Set-Content -Path $GeneratedIss -Value $Generated -Encoding UTF8

$InstallerName = "AutoDocumentScanner-v$Version-Setup.exe"
$InstallerPath = Join-Path $ReleaseRoot $InstallerName
$InstallerHashPath = "$InstallerPath.sha256.txt"

if (Test-Path $InstallerPath) {
    Remove-Item $InstallerPath -Force
}
if (Test-Path $InstallerHashPath) {
    Remove-Item $InstallerHashPath -Force
}

Write-Host "======================================"
Write-Host " AutoDocumentScanner - Installer Build"
Write-Host "======================================"
Write-Host "Version : $Version"
Write-Host "Compiler: $Compiler"
Write-Host "Source  : $PackageDir"
Write-Host "Logo    : $InstallerIcon"

& $Compiler $GeneratedIss
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup gagal membangun installer. Exit code: $LASTEXITCODE"
}

if (-not (Test-Path $InstallerPath)) {
    throw "Compiler selesai tetapi installer tidak ditemukan: $InstallerPath"
}

$InstallerHash = Get-FileHash $InstallerPath -Algorithm SHA256
Set-Content -Path $InstallerHashPath -Value "$($InstallerHash.Hash.ToLower())  $InstallerName" -Encoding UTF8

Write-Host "`n======================================"
Write-Host " INSTALLER BERHASIL"
Write-Host "======================================"
Write-Host "Installer: $InstallerPath"
Write-Host "SHA256   : $InstallerHashPath"
Write-Host "`nInstaller dibuat dari folder release yang input/output-nya sudah kosong."
