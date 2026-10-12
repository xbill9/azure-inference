# One Strands agent, every Azure inference path this subscription can run

The Azure third of a three-cloud survey of LLM inference ([aws-inference-strands](https://github.com/xbill9/aws-inference-strands), [gcp-inference](https://github.com/xbill9/gcp-inference), and the comparison in [cross-cloud-inference](https://github.com/xbill9/cross-cloud-inference)). Same demos, same grader, same prompts, same Gemma 4 repack on the self-hosted path.

| Backend | What serves the model | Model |
|---|---|---|
| `foundry` | Azure AI Foundry, global deployment, per token | `gpt-4.1-nano` |
| `phi` | Azure AI Foundry, global deployment, per token | `Phi-4-mini-instruct` |
| `vm` | vLLM 0.30.0 on a `Standard_NV36ads_A10_v5` (one A10, GRID driver), eastus | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` |
| `aca` | vLLM on Container Apps serverless GPU (T4), westus3 | the same repack; **blocked: GPU quota 0** |

| Script | What it does |
|---|---|
| `demo1_counting.py` | The same tool-using agent and question on each backend, graded on the answer **and** on the filter the model sent |
| `demo2_orchestrator.py` | A Foundry agent that calls every backend as a tool and quotes a scoreboard computed in code |
| `bench_chat.py` | The same 100-token prompt on every backend, timed in code |
| `cost.py` | USD per hour and per million output tokens, from Azure Retail Prices API list prices and measured throughput |

The three files above are the ones in the AWS and GCP repos; only `backends.py` changes per cloud.

## Bring the backends up

```bash
stage/up.sh vm 2>&1 | tee stage/up.log    # NV36ads_A10_v5 in eastus; starts billing (USD 3.20/hr list)
stage/forward.sh && source stage/demo.env  # SSH tunnel to the VM, Foundry key, writes the variables
python3 stage/check.py --wait 1800         # one question and one tool call per backend; READY
python3 stage/ask.py vm                    # one backend, one question, timed
stage/down.sh                              # delete the VM and the app; --all also deletes the RG and the Foundry deployments
```

The VM boots Ubuntu 24.04, installs NVIDIA's GRID driver (`stage/vm-setup.sh`; the NVadsA10 v5 sizes are vGPU-backed and the datacenter driver does not bind), the container toolkit and vLLM. vLLM listens on the VM's localhost with an API key; the only inbound rule is SSH from the address that ran `up.sh`.

`stage/image/` is vLLM 0.30.0 plus the Turing shared-memory clamp from `~/sagemaker-gemma/turing`, built with `az acr build` for the Container Apps T4 profile.

## What this subscription allowed (2026-10-12)

| Path | Quota | Result |
|---|---|---|
| VM, NVadsA10 v5 family | 36 vCPUs in eastus, 6 in southcentralus | one NV36ads_A10_v5 |
| VM, NCasT4_v3 and NDasrA100_v4 families | 0 | not run |
| Container Apps serverless GPU, T4 and A100 | 0 for both (`ManagedEnvironmentConsumptionT4Gpus`, `...NCA100Gpus`) | create refused: `ContainerAppInvalidScaleExceedsQuota` |
| Azure ML managed online endpoints | GPU quota only in the retired NC and NV families | not run |
| Foundry per-token | gpt-4.1-nano at 50K TPM; Phi-4-mini capped at 20K TPM | both deployed |

Evidence is in `results/2026-10-12/`.
