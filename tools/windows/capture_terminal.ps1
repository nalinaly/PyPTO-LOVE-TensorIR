param(
    [Parameter(Mandatory = $true)]
    [string]$Title,

    [Parameter(Mandatory = $true)]
    [string]$LinuxCommand,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$Distro = "Ubuntu",
    [Parameter(Mandatory = $true)]
    [string]$Workspace,
    [int]$TimeoutSeconds = 7200,
    [int]$HoldSeconds = 45
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class PyptoWindowCapture {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr extraData);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int command);

    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr hWnd, uint message, IntPtr wParam, IntPtr lParam);
}
"@

function Find-WindowByExactTitle([string]$ExpectedTitle) {
    $script:MatchedWindow = [IntPtr]::Zero
    $callback = [PyptoWindowCapture+EnumWindowsProc]{
        param([IntPtr]$Handle, [IntPtr]$Unused)
        if (-not [PyptoWindowCapture]::IsWindowVisible($Handle)) {
            return $true
        }
        $text = New-Object System.Text.StringBuilder 1024
        [void][PyptoWindowCapture]::GetWindowText($Handle, $text, $text.Capacity)
        if ($text.ToString() -eq $ExpectedTitle) {
            $script:MatchedWindow = $Handle
            return $false
        }
        return $true
    }
    [void][PyptoWindowCapture]::EnumWindows($callback, [IntPtr]::Zero)
    return $script:MatchedWindow
}

if ($TimeoutSeconds -le 0 -or $HoldSeconds -lt 10) {
    throw "TimeoutSeconds must be positive and HoldSeconds must be at least 10"
}
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    throw "OutputPath must be an absolute Windows path"
}

$uniqueTitle = "PyPTO release - $Title - $([Guid]::NewGuid().ToString('N').Substring(0, 8))"
$nonce = [Guid]::NewGuid().ToString('N')
$marker = "/tmp/pypto-terminal-capture-$nonce.done"
$runner = "/tmp/pypto-terminal-capture-$nonce.sh"
$uncRoot = "\\wsl.localhost\$Distro"
$markerWindows = "$uncRoot\tmp\pypto-terminal-capture-$nonce.done"
$runnerWindows = "$uncRoot\tmp\pypto-terminal-capture-$nonce.sh"
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($LinuxCommand))
$workspaceEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Workspace))
$titleQuoted = $uniqueTitle.Replace("'", "")
$wrapper = @"
set -o pipefail
printf '\033]0;$titleQuoted\007'
cd "`$(printf '%s' '$workspaceEncoded' | base64 -d)"
printf '\nPyPTO release evidence: $titleQuoted\n'
printf 'workspace=%s\n' "`$PWD"
printf 'started_utc=%s\n\n' "`$(date -u +%FT%TZ)"
printf '%s' '$encoded' | base64 -d | bash --noprofile --norc
status=`$?
printf '\nfinished_utc=%s\nexit_code=%s\n' "`$(date -u +%FT%TZ)" "`$status"
printf '%s\n' "`$status" > '$marker'
sleep $HoldSeconds
exit `$status
"@

$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runnerWindows, $wrapper, $utf8)
$safeDistro = $Distro.Replace('"', '')
$safeTitle = $uniqueTitle.Replace('"', '')
$argumentLine = "-w new nt -p `"$safeDistro`" --title `"$safeTitle`" " +
    "wsl.exe -d `"$safeDistro`" -- bash --noprofile --norc $runner"
Start-Process -FilePath "wt.exe" -ArgumentList $argumentLine | Out-Null

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
$window = [IntPtr]::Zero
do {
    Start-Sleep -Milliseconds 250
    $window = Find-WindowByExactTitle $uniqueTitle
    $finished = Test-Path -LiteralPath $markerWindows
    if ([DateTime]::UtcNow -ge $deadline) {
        throw "Timed out waiting for terminal evidence command"
    }
} while ($window -eq [IntPtr]::Zero -or -not $finished)

[void][PyptoWindowCapture]::ShowWindow($window, 3)
[void][PyptoWindowCapture]::SetForegroundWindow($window)
Start-Sleep -Milliseconds 750

$rect = New-Object PyptoWindowCapture+RECT
if (-not [PyptoWindowCapture]::GetWindowRect($window, [ref]$rect)) {
    throw "GetWindowRect failed"
}
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -lt 320 -or $height -lt 200) {
    throw "Terminal window is unexpectedly small: ${width}x${height}"
}

$directory = [System.IO.Path]::GetDirectoryName($OutputPath)
[System.IO.Directory]::CreateDirectory($directory) | Out-Null
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}

$statusText = [System.IO.File]::ReadAllText($markerWindows).Trim()
Remove-Item -LiteralPath $markerWindows, $runnerWindows -Force -ErrorAction SilentlyContinue
[void][PyptoWindowCapture]::PostMessage($window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)

if ($statusText -ne "0") {
    throw "Evidence command failed with exit code $statusText; screenshot retained at $OutputPath"
}

$file = Get-Item -LiteralPath $OutputPath
if ($file.Length -lt 4096) {
    throw "Captured PNG is unexpectedly small: $($file.Length) bytes"
}
Write-Output $file.FullName
