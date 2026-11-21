#requires -Version 5.1
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') {
    throw 'Run this helper on Windows. On Linux, use bash scripts/setup.sh.'
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this helper in a regular PowerShell window, not as Administrator.'
}

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($env:SIGN_TRANSLATOR_VENV) {
    if (![IO.Path]::IsPathRooted($env:SIGN_TRANSLATOR_VENV)) {
        throw 'SIGN_TRANSLATOR_VENV must be an absolute Windows path.'
    }
    $venv = [IO.Path]::GetFullPath($env:SIGN_TRANSLATOR_VENV)
} else {
    $venv = Join-Path $root '.venv'
}
$python = Join-Path $venv 'Scripts\python.exe'
$check = @'
import platform, sys
minor = sys.argv[1]
supported = (sys.implementation.name == 'cpython' and sys.platform == 'win32'
             and platform.machine().lower() in ('amd64', 'x86_64')
             and sys.maxsize > 2**32 and sys.version_info[:2] in ((3, 11), (3, 12)))
matches = minor == 'any' or f'3.{sys.version_info.minor}' == minor
is_venv = sys.argv[2] != 'venv' or sys.prefix != sys.base_prefix
raise SystemExit(0 if supported and matches and is_venv else 1)
'@

function Test-Interpreter($Exe, [string[]] $LauncherArgs, $Minor, $Kind) {
    try {
        & $Exe @LauncherArgs -c $check $Minor $Kind 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Invoke-Checked($Exe, [string[]] $Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE." }
}

if (Test-Path -LiteralPath $venv) {
    if (!(Test-Path -LiteralPath (Join-Path $venv 'pyvenv.cfg')) -or
        !(Test-Path -LiteralPath $python) -or !(Test-Interpreter $python @() 'any' 'venv')) {
        throw "Existing $venv is not a compatible Windows x64 CPython 3.11/3.12 venv. It was not changed."
    }
} else {
    $launcher = Get-Command -Name py.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $direct = Get-Command -Name python.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $interpreter = $null
    $launcherArgs = @()
    foreach ($minor in @('3.12', '3.11')) {
        if ($launcher -and (Test-Interpreter $launcher.Source @("-$minor") $minor 'base')) {
            $interpreter = $launcher.Source
            $launcherArgs = @("-$minor")
            break
        }
        if ($direct -and $direct.Source -notlike '*\Microsoft\WindowsApps\*' -and
            (Test-Interpreter $direct.Source @() $minor 'base')) {
            $interpreter = $direct.Source
            break
        }
    }
    if (!$interpreter) { throw 'Install Windows x64 CPython 3.12 (preferred) or 3.11 from python.org.' }
    Invoke-Checked $interpreter ($launcherArgs + @('-m', 'venv', $venv))
}

Push-Location -LiteralPath $root
try {
    Invoke-Checked $python @('-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes', '-r', 'requirements.txt')
    Invoke-Checked $python @('-m', 'pip', 'check')
    Invoke-Checked $python @('-m', 'utils.model_assets', 'install')
    Invoke-Checked $python @('-m', 'utils.model_assets', 'verify')
    Write-Host ('Launch with: & "{0}" "{1}"' -f $python, (Join-Path $root 'app.py'))
} finally {
    Pop-Location
}
