#!/usr/bin/env bash
set -euo pipefail

buildType="${1:-Release}"
scriptDir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repoRoot="$(cd -- "${scriptDir}/.." && pwd)"
buildDir="${repoRoot}/build/${buildType,,}"

generator="Ninja"
if ! command -v ninja >/dev/null 2>&1; then
    generator="Unix Makefiles"
fi

cmake -S "${repoRoot}" -B "${buildDir}" -G "${generator}" -DCMAKE_BUILD_TYPE="${buildType}"
cmake --build "${buildDir}" --parallel "$(nproc)"
