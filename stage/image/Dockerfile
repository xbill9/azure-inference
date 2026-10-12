# vLLM 0.30.0 with one file changed, for the NVIDIA T4 (Turing, SM 7.5) on Container Apps.
#
# Gemma 4's full-attention layers are 512 wide, which makes vLLM pick its Triton
# attention kernel. That kernel's tile needs 98,304 bytes of shared memory per block
# and Turing allows 65,536. patch_triton_turing.py halves the tiles on pre-Ampere
# GPUs only (a no-op on the A10). Same patch as ~/sagemaker-gemma/turing.
FROM vllm/vllm-openai:v0.30.0
COPY patch_triton_turing.py /opt/turing/patch_triton_turing.py
RUN set -eu; \
    target="$(python3 -c 'import importlib.util, os; print(os.path.join(importlib.util.find_spec("vllm").submodule_search_locations[0], "v1/attention/ops/triton_unified_attention.py"))')"; \
    python3 /opt/turing/patch_triton_turing.py "$target"; \
    python3 /opt/turing/patch_triton_turing.py --check "$target"; \
    python3 -c 'import torch, sys; a = torch._C._cuda_getArchFlags(); print("torch arch:", a); sys.exit(0 if "sm_75" in a else 1)'
