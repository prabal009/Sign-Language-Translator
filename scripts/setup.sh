#!/usr/bin/env bash
set -euo pipefail

fail() { printf '%s\n' "$*" >&2; exit 1; }
(( EUID != 0 )) || fail 'Run this setup as your normal user, not with sudo.'

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
venv=${SIGN_TRANSLATOR_VENV:-"$root/.venv"}
[[ $venv = /* ]] || fail 'SIGN_TRANSLATOR_VENV must be an absolute Linux path.'
python="$venv/bin/python"
check='import platform, sys
minor = sys.argv[1]
supported = (sys.implementation.name == "cpython" and sys.platform == "linux"
             and platform.machine() == "x86_64" and sys.version_info[:2] in ((3, 11), (3, 12)))
matches = minor == "any" or f"3.{sys.version_info.minor}" == minor
is_venv = sys.argv[2] != "venv" or sys.prefix != sys.base_prefix
raise SystemExit(0 if supported and matches and is_venv else 1)'

if [[ -e $venv || -L $venv ]]; then
    [[ -f $venv/pyvenv.cfg && -x $python ]] && "$python" -c "$check" any venv ||
        fail "Existing $venv is not a compatible Linux x86-64 CPython 3.11/3.12 venv. It was not changed."
else
    interpreter=''
    for minor in 3.12 3.11; do
        for candidate in "python$minor" python3; do
            if command -v "$candidate" >/dev/null && "$candidate" -c "$check" "$minor" base 2>/dev/null; then
                interpreter=$candidate
                break 2
            fi
        done
    done
    [[ -n $interpreter ]] || fail 'Install Linux x86-64 CPython 3.12 (preferred) or 3.11 and make it available on PATH.'
    "$interpreter" -m venv "$venv" ||
        fail "Cannot create $venv. Check venv support and use a symlink-capable filesystem (SIGN_TRANSLATOR_VENV=/absolute/path)."
fi

cd -- "$root"
"$python" -m pip install --only-binary=:all: --require-hashes -r requirements.txt
"$python" -m pip check
"$python" -m utils.model_assets install
"$python" -m utils.model_assets verify
printf 'Launch with: %q %q\n' "$python" "$root/app.py"
