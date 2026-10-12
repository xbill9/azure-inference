"""The inference backends the two demos talk to, and how to reach each one.

Every backend is configured from the environment, so the same code runs against
whatever is up on the day. A backend whose variable is unset is skipped, not
faked: the demos print which ones they could reach.

    name     what it is                                        how it is reached
    foundry  gpt-4.1-nano, Azure AI Foundry, per token          Foundry OpenAI v1 endpoint, account key
    phi      Phi-4-mini-instruct, Azure AI Foundry, per token   the same endpoint and key
    aca      vLLM on Container Apps serverless GPU (T4)         public HTTPS, vLLM --api-key
    vm       vLLM on an NV36ads_A10_v5 VM (A10)                 SSH tunnel to localhost, vLLM --api-key

All of them speak the OpenAI chat completions API, so Strands drives every one
with strands.models.openai.OpenAIModel. Every self-hosted backend runs vLLM with
--enable-auto-tool-choice, so all of them can drive an agent.
"""

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class Backend:
    name: str
    kind: str
    label: str
    model: str
    url: str
    key: str = "unused"

    tool_capable = True


def configured() -> list[Backend]:
    """Every backend whose settings are present in the environment."""
    env = os.environ.get
    found = []
    if env("FOUNDRY_URL"):
        found.append(Backend("foundry", "openai", "Azure AI Foundry (gpt-4.1-nano)",
                             env("FOUNDRY_MODEL", "cci-gpt41-nano"), url=env("FOUNDRY_URL"),
                             key=env("FOUNDRY_KEY", "unused")))
        if env("PHI_MODEL"):
            found.append(Backend("phi", "openai", "Azure AI Foundry (Phi-4-mini)",
                                 env("PHI_MODEL"), url=env("FOUNDRY_URL"), key=env("FOUNDRY_KEY", "unused")))
    if env("ACA_URL"):
        found.append(Backend("aca", "openai", "vLLM on Container Apps (T4)",
                             env("ACA_MODEL", "gemma"), url=env("ACA_URL"), key=env("VLLM_KEY", "unused")))
    if env("VM_URL"):
        found.append(Backend("vm", "openai", "vLLM on NV36ads_A10_v5 (A10)",
                             env("VM_MODEL", "gemma"), url=env("VM_URL"), key=env("VLLM_KEY", "unused")))
    only = env("BACKENDS")
    if only:
        keep = {b.strip() for b in only.split(",")}
        found = [b for b in found if b.name in keep]
    return found


def strands_model(b: Backend, temperature: float = 0.0, max_tokens: int = 512):
    """The Strands model object for a backend."""
    from strands.models.openai import OpenAIModel
    return OpenAIModel(client_args={"base_url": b.url, "api_key": b.key}, model_id=b.model,
                       params={"temperature": temperature, "max_tokens": max_tokens})


def chat_once(b: Backend, prompt: str, max_tokens: int = 64) -> dict:
    """One plain chat request, timed here. Returns text, tokens and seconds.

    Tokens per second is computed from the backend's own completion token count
    and this function's wall clock, so it includes network and prefill.
    """
    body = {"model": b.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    headers = {"content-type": "application/json", "authorization": f"Bearer {b.key}"}
    req = urllib.request.Request(b.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    seconds = time.perf_counter() - t0
    text = out["choices"][0]["message"]["content"] or ""
    completion = out["usage"]["completion_tokens"]
    return {"backend": b.name, "label": b.label, "text": text.strip(), "completion_tokens": completion,
            "seconds": round(seconds, 3), "tokens_per_second": round(completion / seconds, 1) if seconds else 0.0}
