$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VersionFile = Join-Path $Root "VERSION.txt"
$SourceDir = Join-Path $Root "dist\AutoDocumentScanner"
$SourceExe = Join-Path $SourceDir "AutoDocumentScanner.exe"
$ReleaseRoot = Join-Path $Root "release"

if (-not (Test-Path $VersionFile)) {
    throw "VERSION.txt tidak ditemukan."
}

$Version = (Get-Content $VersionFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($Version)) {
    throw "VERSION.txt kosong."
}

if (-not (Test-Path $SourceExe)) {
    throw "Build belum tersedia. Jalankan build_windows.ps1 terlebih dahulu."
}

$PackageName = "AutoDocumentScanner-v$Version-windows-x64"
$PackageDir = Join-Path $ReleaseRoot $PackageName
$ZipPath = Join-Path $ReleaseRoot "$PackageName.zip"
$ZipHashPath = "$ZipPath.sha256.txt"

Write-Host "======================================"
Write-Host " AutoDocumentScanner - Final Release"
Write-Host "======================================"
Write-Host "Version : $Version"

if (Test-Path $PackageDir) {
    Remove-Item $PackageDir -Recurse -Force
}
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
if (Test-Path $ZipHashPath) {
    Remove-Item $ZipHashPath -Force
}

New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null

Write-Host "`n[1/5] Copy runtime hasil build"
Copy-Item $SourceDir $PackageDir -Recurse -Force

Write-Host "`n[2/5] Bersihkan data runtime/UAT"
foreach ($RuntimeName in @("input", "output")) {
    $RuntimeDir = Join-Path $PackageDir $RuntimeName

    if (Test-Path $RuntimeDir) {
        Get-ChildItem $RuntimeDir -Force | Remove-Item -Recurse -Force
    }
    else {
        New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    }
}

# Release metadata only. No source code, tests, local input, or local output are copied.
Copy-Item (Join-Path $Root "VERSION.txt") (Join-Path $PackageDir "VERSION.txt") -Force
Copy-Item (Join-Path $Root "DEPLOYMENT.md") (Join-Path $PackageDir "DEPLOYMENT.md") -Force

Write-Host "`n[3/5] Buat checksum isi paket"
$ChecksumFile = Join-Path $PackageDir "SHA256SUMS.txt"
$Lines = @()

Get-ChildItem $PackageDir -Recurse -File |
    Where-Object { $_.FullName -ne $ChecksumFile } |
    Sort-Object FullName |
    ForEach-Object {
        $Hash = Get-FileHash $_.FullName -Algorithm SHA256
        $Relative = [System.IO.Path]::GetRelativePath($PackageDir, $_.FullName)
        $Lines += "$($Hash.Hash.ToLower())  $Relative"
    }

Set-Content -Path $ChecksumFile -Value $Lines -Encoding UTF8

Write-Host "`n[4/5] Buat ZIP final"
Compress-Archive -Path $PackageDir -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host "`n[5/5] Buat checksum ZIP"
$ZipHash = Get-FileHash $ZipPath -Algorithm SHA256
Set-Content -Path $ZipHashPath -Value "$($ZipHash.Hash.ToLower())  $([System.IO.Path]::GetFileName($ZipPath))" -Encoding UTF8

Write-Host "`n======================================"
Write-Host " RELEASE BERHASIL"
Write-Host "======================================"
Write-Host "Folder : $PackageDir"
Write-Host "ZIP    : $ZipPath"
Write-Host "SHA256 : $ZipHashPath"
Write-Host "`nFolder input/output pada release sudah dikosongkan."
Write-Host "Pindahkan ZIP final ke PC induk, bukan seluruh repository."
