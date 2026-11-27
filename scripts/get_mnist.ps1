# get_mnist.ps1 - the Windows version of get_mnist.sh: download MNIST into
# data\ (gitignored), verify the same checksums, decompress to raw IDX.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\get_mnist.ps1 [DIR]
# (or: cmake --build build --target data). Works in Windows PowerShell 5.1
# and PowerShell 7; only built-in cmdlets and .NET classes are used.
param([string]$Dir = "data")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue" # the progress bar makes Invoke-WebRequest very slow in 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Mirrors = @("https://storage.googleapis.com/cvdf-datasets/mnist", "https://ossci-datasets.s3.amazonaws.com/mnist")
# MD5 of the .gz (as published by torchvision) and SHA-256 of the decompressed file
$Files = @(
    @("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db"),
    @("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5"),
    @("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7"),
    @("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2")
)

$Dir = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Dir))
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

function Hash-Of([string]$Path, [string]$Algo) { (Get-FileHash -Algorithm $Algo -LiteralPath $Path).Hash.ToLower() }

foreach ($f in $Files) {
    $name, $md5, $sha = $f
    $raw = Join-Path $Dir ($name -replace '\.gz$', '')
    if (Test-Path -LiteralPath $raw) { Write-Output "ok      $raw (already present)"; continue }
    $gz = Join-Path $Dir $name
    if (-not (Test-Path -LiteralPath $gz) -or (Hash-Of $gz MD5) -ne $md5) {
        Remove-Item -LiteralPath $gz -ErrorAction SilentlyContinue
        foreach ($m in $Mirrors) {
            Write-Output "fetch   $m/$name"
            try { Invoke-WebRequest -UseBasicParsing -Uri "$m/$name" -OutFile "$gz.part"; Move-Item -Force "$gz.part" $gz; break }
            catch { Remove-Item -LiteralPath "$gz.part" -ErrorAction SilentlyContinue }
        }
    }
    if (-not (Test-Path -LiteralPath $gz)) { throw "could not download $name from any mirror" }
    $got = Hash-Of $gz MD5
    if ($got -ne $md5) { Remove-Item -LiteralPath $gz; throw "checksum mismatch for ${name}: got $got, expected $md5" }
    Write-Output "verify  $name md5 $got"
    $in = [IO.File]::OpenRead($gz)
    $zip = New-Object IO.Compression.GZipStream($in, [IO.Compression.CompressionMode]::Decompress)
    $out = [IO.File]::Create("$raw.part")
    try { $zip.CopyTo($out) } finally { $out.Close(); $zip.Close(); $in.Close() }
    $got = Hash-Of "$raw.part" SHA256
    if ($got -ne $sha) { Remove-Item -LiteralPath "$raw.part"; throw "decompressed $name has sha256 $got, expected $sha" }
    Move-Item -Force "$raw.part" $raw
    Remove-Item -LiteralPath $gz
    Write-Output "ok      $raw"
}
Write-Output "MNIST is ready in $Dir"
