# get_text.ps1 - the Windows version of get_text.sh: download the two texts
# into data\ (gitignored), verify every download, and check that the result
# is byte-identical to what get_text.sh produces (same SHA-256).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\get_text.ps1 [DIR]
# (or: cmake --build build --target data-text). See get_text.sh for the
# sources and licences. ASCII-only on purpose: Windows PowerShell 5.1 reads
# a script without a BOM in the ANSI code page, so the Arabic words of the
# wiki markup are written as \uXXXX regex escapes.
param([string]$Dir = "data")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Dir = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Dir))
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Utf8 = New-Object Text.UTF8Encoding($false, $true) # no BOM; throw on invalid bytes
$Agent = "neuromz-get-text/1.0"

# .NET, not Get-FileHash (a script function of Windows PowerShell 5.1 that
# does not load when 5.1 is started from PowerShell 7)
function Sha-Of([string]$Path) { Sha-OfBytes ([IO.File]::ReadAllBytes($Path)) }
function Sha-OfBytes([byte[]]$Bytes) {
    $h = [Security.Cryptography.SHA256]::Create()
    ([BitConverter]::ToString($h.ComputeHash($Bytes)) -replace '-', '').ToLower()
}
function Fetch([string]$Url, [string]$Out) {
    Invoke-WebRequest -UseBasicParsing -UserAgent $Agent -Uri $Url -OutFile $Out
}
# write lines joined by "\n" with a final "\n" (like awk), after checking the hash
function Write-Checked([string]$Out, [string[]]$Lines, [string]$Want) {
    $bytes = $Utf8.GetBytes((($Lines -join "`n") + "`n"))
    $got = Sha-OfBytes $bytes
    if ($got -ne $Want) { throw "$Out would have sha256 $got, expected $Want (the same file as get_text.sh)" }
    [IO.File]::WriteAllBytes($Out, $bytes)
    Write-Output "ok      $Out ($($bytes.Length) bytes, sha256 matches get_text.sh)"
}

function Get-Alice {
    $Name = "11-0.txt"
    $Sha = "a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754"
    $OutSha = "2c9eddc277e75b19ce8e5e884e1ee57f2341fa798be0e30057ca82d32366e4af"
    $Mirrors = @("https://gutenberg.pglaf.org/1/11", "https://mirrors.xmission.com/gutenberg/1/11")
    $Out = Join-Path $Dir "alice.txt"
    if (Test-Path -LiteralPath $Out) { Write-Output "ok      $Out (already present)"; return }
    $raw = Join-Path $Dir "$Name.part"
    foreach ($m in $Mirrors) {
        Write-Output "fetch   $m/$Name"
        try { Fetch "$m/$Name" $raw; break } catch { Remove-Item -LiteralPath $raw -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path -LiteralPath $raw)) { throw "could not download $Name from any mirror" }
    $got = Sha-Of $raw
    if ($got -ne $Sha) { Remove-Item -LiteralPath $raw; throw "checksum mismatch for ${Name}: got $got, expected $Sha" }
    Write-Output "verify  $Name sha256 $got"
    # the lines strictly between "*** START OF" and "*** END OF" (each keeps its "\r")
    $text = $Utf8.GetString([IO.File]::ReadAllBytes($raw))
    $lines = $text.Split("`n")
    if ($text.EndsWith("`n")) { $lines = $lines[0..($lines.Length - 2)] }
    $keep = New-Object Collections.Generic.List[string]
    $on = $false
    foreach ($l in $lines) {
        if ($l.StartsWith("*** START OF")) { $on = $true; continue }
        if ($l.StartsWith("*** END OF")) { $on = $false }
        if ($on) { $keep.Add($l) }
    }
    Write-Checked $Out $keep.ToArray() $OutSha
    Remove-Item -LiteralPath $raw
}

# Kalila wa-Dimna: page revision and SHA-256 of its raw wikitext, in the
# book's order (the same list as get_text.sh)
$KalilaRevs = @(
    "488570 e6c20d524aeb5a28e77dc7547664ded0809c1b283387cedb8fae6141c0b61dbc",
    "245550 7ec8e336e98828ed2fb1d1c1258fa22fa17fecf3dfb2daf2128e45c0d00901ad",
    "245551 c0439c96172f72eb44c86e067555ba44943dedb44bcf63bb534cd2f73911d22b",
    "141243 3c1b1f9009952a7cc63e9799958ae6f5059b1c0f73d8bd8d301482e885b37394",
    "272368 1bb1ab2f940e799d3bc6315d190f61fbae6f0120047833d1b062e37feb4dd8dc",
    "290965 107dc47453f5c0138842c2561239580bb84a3948923e07d0c17a9523264c80ee",
    "245541 45f5d526726e83adb150f02d652826f35fd6716e0accf00b0fc09bc9134bfd86",
    "285510 1fa9b2daf0e4a1f7b670a93fc6b4026fd1b4ab49be128b60cc8024e60d66b882",
    "231360 7e91090670ce60ea93db7e2e67078b8f77a642a12e272c6f463173ea57f6ee89",
    "251292 0e8e4373dbaa3f121abc25fbbc8621740f90c19877998811e1b306948b617230",
    "231362 20846ca4f50efd4114ccbc1229799d10de61064a845a6bb9f6a097141c6498fe",
    "231363 b7a5aa9ac135254b406ceaea168fe0e44775ef2c9e48b031ec9a160874cc31cc",
    "302554 210617ce591afcf6f766301250addd178d2ba8f95e515507c684f19cdb3275d2",
    "231372 201a4ffa9f68905f7a4df9e05f3f4ec5e954b71be39df804b8340c7e27a630bc",
    "374244 ab8688bd587efb310d54c0f084a90ef1f10581811077a2641e2c12d7e88ea43a",
    "231371 2f8506a25a6211259a8758d44f001a839dd65497746451f62e1214e03600ecff",
    "381514 a91887cd095491c699bb0bbfcf495458d8f0cf5545adac74ff62b9c642235460",
    "231368 39e5c7eb6650429cc7ed7424bfa03ac812a11da4e44ad990d173610d32b97497",
    "231369 c0479eb75d3b07b4fe2fb9782e84d622d0d4f26b58b230179125f037449318fb"
)

function Get-Kalila {
    $OutSha = "709e5b8fc92a2e3393d57382a046bba51713d4d511bddf9e62cfbb8def0ae90a"
    $Out = Join-Path $Dir "kalila.txt"
    if (Test-Path -LiteralPath $Out) { Write-Output "ok      $Out (already present)"; return }
    $tmp = Join-Path $Dir "kalila.part.d"
    if (Test-Path -LiteralPath $tmp) { Remove-Item -Recurse -Force -LiteralPath $tmp }
    New-Item -ItemType Directory -Path $tmp | Out-Null
    Write-Output "fetch   19 chapters of Kalila wa-Dimna from ar.wikisource.org (fixed revisions)"
    $all = New-Object IO.MemoryStream
    $n = 0
    foreach ($entry in $KalilaRevs) {
        $rev, $sum = $entry.Split(" ")
        $n++
        $f = Join-Path $tmp ("{0:D2}.txt" -f $n)
        Fetch "https://ar.wikisource.org/w/index.php?oldid=$rev&action=raw" $f
        $got = Sha-Of $f
        if ($got -ne $sum) { throw "checksum mismatch for revision ${rev}: got $got, expected $sum" }
        $b = [IO.File]::ReadAllBytes($f)
        $all.Write($b, 0, $b.Length)
    }
    Write-Output "verify  19 revisions sha256 ok"
    # strip_wikitext.awk, step by step (see that file)
    $t = $Utf8.GetString($all.ToArray())
    if (-not $t.EndsWith("`n")) { $t += "`n" }
    do { $before = $t; $t = [regex]::Replace($t, '\{\{[^{}]*\}\}', '') } while ($t -ne $before)
    $t = [regex]::Replace($t, '\[\[(\u0645\u0644\u0641|\u0635\u0648\u0631\u0629|\u062A\u0635\u0646\u064A\u0641|File|Image|Category):[^\]\n]*\]\]', '')
    $t = [regex]::Replace($t, '\[\[[^\]|\n]*\|', '')
    $t = [regex]::Replace($t, '\[\[|\]\]', '')
    $t = [regex]::Replace($t, "'''|''", '')
    $t = [regex]::Replace($t, '<[^>\n]*>', '')
    $t = $t.Replace('&nbsp;', ' ')
    $keep = New-Object Collections.Generic.List[string]
    foreach ($s in $t.Split("`n")) {
        $s = [regex]::Replace($s, '[\t\r]', ' ')
        $s = [regex]::Replace($s, '^[=*#:;| ]+', '')
        $s = [regex]::Replace($s, '[= ]+$', '')
        $s = [regex]::Replace($s, '  +', ' ')
        if ($s -ne '') { $keep.Add($s) }
    }
    Write-Checked $Out $keep.ToArray() $OutSha
    Remove-Item -Recurse -Force -LiteralPath $tmp
}

$rc = 0
try { Get-Alice } catch { Write-Error "error: $_" -ErrorAction Continue; $rc = 1 }
try { Get-Kalila } catch { Write-Error "error: $_" -ErrorAction Continue; $rc = 1 }
exit $rc
