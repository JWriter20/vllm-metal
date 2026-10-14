#!/bin/bash
# Point the serving services at a built environment and restart them.
# Rollback is the same command with the previous version.
set -euo pipefail
ROOT=${CK_INFERENCE_ROOT:-$HOME/ck-inference}
version=${1:?usage: ck/activate.sh <version>}
env=$ROOT/envs/vllm-metal-$version
[ -x "$env/bin/python" ] && [ -f "$env/SOURCE_COMMIT" ] || { echo "$env is not a finished build" >&2; exit 1; }
ln -sfn "$env" "$ROOT/vllm-env"
for s in vllm-9b vllm-27b; do launchctl kickstart -k "gui/$(id -u)/com.captchakraken.$s"; done
echo "active: $(readlink "$ROOT/vllm-env") @ $(cat "$env/SOURCE_COMMIT")"
