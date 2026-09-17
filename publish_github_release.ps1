$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Repo = "Heat41/AutoDocumentScanner_MVP"
$VersionFile = Join-Path $Root "VERSION.txt"

if (-not (Test-Path $VersionFile)) {
    throw "VERSION.txt tidak ditemukan."
}

$Version = (Get-Content $VersionFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($Version)) {
    throw "VERSION.txt kosong."
}

$Tag = "v$Version"
$Title = "AutoDocumentScanner v$Version"
$ReleaseDir = Join-Path $Root "release"
$NotesFile = Join-Path $Root "RELEASE_NOTES_v$Version.md"

$Assets = @(
    (Join-Path $ReleaseDir "AutoDocumentScanner-v$Version-windows-x64.zip"),
    (Join-Path $ReleaseDir "AutoDocumentScanner-v$Version-windows-x64.zip.sha256.txt"),
    (Join-Path $ReleaseDir "AutoDocumentScanner-v$Version-Setup.exe"),
    (Join-Path $ReleaseDir "AutoDocumentScanner-v$Version-Setup.exe.sha256.txt")
)

if (-not (Test-Path $NotesFile)) {
    throw "Release notes tidak ditemukan: $NotesFile"
}

foreach ($Asset in $Assets) {
    if (-not (Test-Path -LiteralPath $Asset -PathType Leaf)) {
        throw "Asset release tidak ditemukan: $Asset`nJalankan make_release.ps1 terlebih dahulu."
    }
}

$GhCommand = Get-Command gh.exe -ErrorAction SilentlyContinue
if (-not $GhCommand) {
    $GhCommand = Get-Command gh -ErrorAction SilentlyContinue
}

if (-not $GhCommand) {
    throw @"
GitHub CLI (gh) belum ditemukan.
Install dengan:
  winget install --id GitHub.cli
Lalu buka PowerShell baru dan jalankan:
  gh auth login
"@
}

$Gh = $GhCommand.Source

function Test-GitHubReleaseExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReleaseTag
    )

    # gh menulis "release not found" ke STDERR dan mengembalikan exit code 1.
    # Dengan $ErrorActionPreference = Stop, Windows PowerShell dapat mengubah
    # STDERR native command menjadi terminating error sebelum exit code sempat
    # diperiksa. Turunkan ErrorActionPreference hanya selama probe ini.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Gh release view $ReleaseTag --repo $Repo 1>$null 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}

Write-Host "======================================"
Write-Host " AutoDocumentScanner - GitHub Release"
Write-Host "======================================"
Write-Host "Repository : $Repo"
Write-Host "Tag        : $Tag"
Write-Host "Version    : $Version"

Write-Host "`n[1/3] Memeriksa autentikasi GitHub CLI"
& $Gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI belum login. Jalankan: gh auth login"
}

Write-Host "`n[2/3] Membuat / memperbarui GitHub Release"
$ReleaseExists = Test-GitHubReleaseExists -ReleaseTag $Tag

if ($ReleaseExists) {
    Write-Host "Release $Tag sudah ada. Memperbarui notes dan asset..."

    & $Gh release edit $Tag `
        --repo $Repo `
        --title $Title `
        --notes-file $NotesFile

    if ($LASTEXITCODE -ne 0) {
        throw "Gagal memperbarui metadata release $Tag."
    }

    & $Gh release upload $Tag @Assets `
        --repo $Repo `
        --clobber

    if ($LASTEXITCODE -ne 0) {
        throw "Gagal mengunggah asset ke release $Tag."
    }
}
else {
    Write-Host "Release $Tag belum ada. Membuat release baru..."

    & $Gh release create $Tag @Assets `
        --repo $Repo `
        --target main `
        --title $Title `
        --notes-file $NotesFile

    if ($LASTEXITCODE -ne 0) {
        throw "Gagal membuat GitHub Release $Tag."
    }
}

Write-Host "`n[3/3] Verifikasi release"
$ReleaseUrl = & $Gh release view $Tag --repo $Repo --json url --jq '.url'
if ($LASTEXITCODE -ne 0) {
    throw "Release dibuat tetapi URL tidak dapat diverifikasi."
}

Write-Host "`n======================================"
Write-Host " GITHUB RELEASE BERHASIL"
Write-Host "======================================"
Write-Host "Release : $ReleaseUrl"
Write-Host "Assets  : 4 file distribusi"
