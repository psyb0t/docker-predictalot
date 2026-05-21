#!/usr/bin/env bash
# Compile deps.in into hash-locked requirements files for both image
# variants. Run via `make deps-lock` after touching deps.in.
#
# Outputs:
#   requirements-cpu.txt    — torch from pytorch.org/whl/cpu
#   requirements-cuda.txt   — torch from pytorch.org/whl/cu126
#
# Both Dockerfiles install with `uv pip install --require-hashes -r <file>`
# so a hijacked wheel uploaded post-publish fails the hash check.
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="scripts/deps.in"
[ -f "$INPUT" ] || { echo "missing $INPUT" >&2; exit 1; }

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

compile_one() {
    local backend="$1"
    local output="$2"
    echo ">> compiling $output (torch-backend=$backend)"
    uv pip compile \
        --quiet \
        --generate-hashes \
        --no-strip-markers \
        --emit-index-url \
        --python-version "$PYTHON_VERSION" \
        --torch-backend "$backend" \
        --output-file "$output" \
        "$INPUT"
}

compile_one cpu   requirements-cpu.txt
# torch 2.4.1 was only built for cu118/cu121/cu124 — cu126 wheels start at
# torch 2.5. CUDA runtimes are forward-compatible at minor-version level, so
# cu124 wheels run fine on the 12.6.2 runtime base image.
compile_one cu124 requirements-cuda.txt

echo ">> done"
echo "   wc -l requirements-*.txt:"
wc -l requirements-cpu.txt requirements-cuda.txt
