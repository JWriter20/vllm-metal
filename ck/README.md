# ck/main

Our own build of vllm-metal: an upstream release plus our fixes, built from
source and served from a pinned environment, independent of upstream merging
them and of Homebrew.

- Base: upstream `v0.31.0`.
- Fixes on top: mixed decode+prefill LoRA routing, `--enable-tower-connector-lora`,
  Qwen3-VL video encoding (upstream PRs #1106, #1107, #1113).
- Performance: batched M-RoPE for packed batches, LoRA decode grouped by adapter
  (no per-row weight gathers). Serving sets `VLLM_METAL_NATIVE_SAMPLING=1`.
- ck3: LoRA slot writes are evaluated eagerly. Before this, text-only adapter churn
  leaked Metal buffers in the idle vision encoder until `[metal::malloc] Resource
  limit (499000) exceeded`. Also: M-RoPE goes through `rotary_emb.apply_rotary`
  (mlx-vlm's fused kernel), and a warning when the tower LoRA flag has no effect.
- `ck/requirements.lock`: every dependency, frozen from the upstream v0.31.0
  Homebrew environment that served correctly before this build existed.
  `mlx-lm` is the git commit vllm-metal's pyproject pins: PyPI's `mlx-lm 0.32.0`
  is a different build that needs a newer `mlx`.
- `ck/build.sh`: fresh venv from the lock, vLLM's CPU wheel (sha256-checked),
  Metal kernels and C++ extension compiled from this source, wheel built,
  checked and installed, upstream's smoke test run. Needs Xcode's Metal toolchain.
- `ck/activate.sh <version>`: points the services at a build and restarts them.

To move to a new upstream release: rebase our fixes onto the new tag, refresh
the lock from that release's environment, bump the `+ckN` version, build,
validate, activate.
