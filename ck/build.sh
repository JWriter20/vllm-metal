#!/bin/bash
# Build this checkout into a fresh, self-contained environment:
#   envs/vllm-metal-<version>/  with every dependency pinned by ck/requirements.lock,
#   vLLM's CPU wheel (sha256-checked), and vllm-metal built from THIS source,
#   Metal kernels and the C++ extension included.
# Run on the Apple Silicon host that will serve, with Xcode's Metal toolchain.
# Activating the result is a separate, deliberate step (ck/activate.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${CK_INFERENCE_ROOT:-$HOME/ck-inference}
PYTHON=${PYTHON:-/opt/homebrew/bin/python3.12}
VLLM_WHEEL_URL=https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl
VLLM_WHEEL_SHA256=21378516a51ff15029f71168e827b94660e21b39484eb52277166f01ac7dd686

[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "refusing to build a dirty tree" >&2; exit 1; }
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml)
env=$ROOT/envs/vllm-metal-$version
[ ! -e "$env" ] || { echo "$env exists; bump the version or remove it" >&2; exit 1; }
echo "== building vllm-metal $version ($(git rev-parse --short HEAD)) into $env"

"$PYTHON" -m venv "$env"
"$env/bin/pip" install -q -r ck/requirements.lock

wheels=$(mktemp -d); trap 'rm -rf "$wheels"' EXIT
curl -sSfL -o "$wheels/vllm-0.31.0+cpu-cp312-cp312-macosx_11_0_arm64.whl" "$VLLM_WHEEL_URL"
echo "$VLLM_WHEEL_SHA256  $wheels/vllm-0.31.0+cpu-cp312-cp312-macosx_11_0_arm64.whl" | shasum -a 256 -c -
"$env/bin/pip" install -q --no-deps "$wheels"/vllm-*.whl

# Native artifacts are compiled against the mlx just installed, from this source.
"$env/bin/python" -c "from vllm_metal.metal.build import require_nax_sdk; require_nax_sdk()"
"$env/bin/python" -m vllm_metal.metal.build
"$env/bin/pip" wheel -q --no-deps --no-build-isolation -w "$wheels" .
wheel=$(ls "$wheels"/vllm_metal-*.whl)
"$env/bin/python" - "$wheel" <<'PY'
import sys, zipfile
from vllm_metal.metal.build import METALLIB_NAMES, NAX_METALLIB_NAME, metallib_path, mlx_version_path, output_path
names = {n.rsplit("/", 1)[-1] for n in zipfile.ZipFile(sys.argv[1]).namelist()}
want = [output_path().name, mlx_version_path().name] + [metallib_path(n).name for n in (*METALLIB_NAMES, NAX_METALLIB_NAME)]
missing = [w for w in want if w not in names]
sys.exit(f"wheel is missing native artifacts: {missing}" if missing else 0)
PY
"$env/bin/pip" install -q --no-deps "$wheel"
cp "$wheel" "$env/"

# Upstream's Homebrew smoke test, run from outside the source tree so the
# installed package is what imports.
(cd / && "$env/bin/python" - "$version" <<'PY'
import importlib.metadata, sys
import mlx.core as mx
from vllm.platforms import current_platform
from vllm_metal.metal import get_ops
from vllm_metal.platform import MetalPlatform
assert importlib.metadata.version("vllm-metal") == sys.argv[1]
assert isinstance(current_platform, MetalPlatform), current_platform
key = mx.ones((1, 1, 64), dtype=mx.float16)
value = key * 2
key_cache = mx.zeros((1, 16, 1, 64), dtype=mx.float16)
value_cache = mx.zeros((1, 16, 1, 64), dtype=mx.float16)
slots = mx.array([3], dtype=mx.int64)
keys, values = get_ops().reshape_and_cache(key, value, key_cache, value_cache, slots)
assert mx.sum(keys).item() == 64 and mx.sum(values).item() == 128
print("smoke test ok:", importlib.metadata.version("vllm-metal"))
PY
)
git rev-parse HEAD > "$env/SOURCE_COMMIT"
echo "== built $env; activate with ck/activate.sh $version"
