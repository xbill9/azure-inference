#!/usr/bin/env bash
# Launch the self-hosted demo backends. Each one starts billing; stage/down.sh stops all of them.
#
#   stage/up.sh             # both: vm aca
#   stage/up.sh vm          # only some
#
# The Foundry models (gpt-4.1-nano, Phi-4-mini) are deployments on an existing
# account and bill per token; they need no launch.
#
# Everything this script creates is in resource group $RG, tagged ManagedBy=cci-demo-run,
# except the Foundry deployments. down.sh deletes the VM and scales the app to zero.
set -uo pipefail

RG=${RG:-cci-demo-rg}
VM_REGIONS=${VM_REGIONS:-eastus southcentralus}       # where NVadsA10 v5 quota exists (36 and 6 vCPUs)
VM_SIZE=${VM_SIZE:-Standard_NV36ads_A10_v5}           # one full A10, 24 GB
VM_IMAGE=${VM_IMAGE:-Canonical:ubuntu-24_04-lts:server:latest}
ACA_ENV=${ACA_ENV:-cci-demo-env}                      # westus3, workload profile gpu-t4 (Consumption-GPU-NC8as-T4)
ACA_APP=${ACA_APP:-cci-demo-vllm}
ACR=${ACR:-ccidemoacr}
REPACK=${REPACK:-xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}
here=$(cd "$(dirname "$0")" && pwd)
keyfile=$here/.vllm-key
[ -f "$keyfile" ] || { python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$keyfile"; chmod 600 "$keyfile"; }
VLLM_KEY=$(cat "$keyfile")
MYIP=$(curl -s -m 10 https://api.ipify.org)

vm() {
  local st region nsg ci
  st=$(az vm show -g "$RG" -n cci-demo-vm --query provisioningState -o tsv 2>/dev/null)
  if [ -n "$st" ]; then echo "vm: cci-demo-vm already exists ($st)"; return 0; fi
  ci=$(mktemp)
  { echo '#cloud-config'
    echo 'write_files:'
    echo '  - path: /etc/cci.env'
    echo "    permissions: '0600'"
    echo '    content: |'
    echo "      REPACK=$REPACK"
    echo "      VLLM_KEY=$VLLM_KEY"
    echo "      VLLM_IMAGE=$VLLM_IMAGE"
    echo '  - path: /opt/cci/setup.sh'
    echo "    permissions: '0755'"
    echo '    encoding: b64'
    echo "    content: $(base64 -w0 "$here/vm-setup.sh")"
    echo '  - path: /etc/systemd/system/cci-setup.service'
    echo '    content: |'
    echo '      [Unit]'
    echo '      After=network-online.target'
    echo '      Wants=network-online.target'
    echo '      [Service]'
    echo '      Type=oneshot'
    echo '      ExecStart=/opt/cci/setup.sh'
    echo '      [Install]'
    echo '      WantedBy=multi-user.target'
    echo 'runcmd:'
    echo '  - [systemctl, daemon-reload]'
    echo '  - [systemctl, enable, cci-setup.service]'
    echo '  - [systemctl, start, --no-block, cci-setup.service]'
  } > "$ci"
  for region in $VM_REGIONS; do
    echo "vm: creating $VM_SIZE in $region"
    out=$(az vm create -g "$RG" -n cci-demo-vm -l "$region" --size "$VM_SIZE" --image "$VM_IMAGE" \
      --admin-username azureuser --ssh-key-values ~/.ssh/id_ed25519.pub \
      --os-disk-size-gb 128 --security-type TrustedLaunch --enable-secure-boot false \
      --nsg-rule NONE --public-ip-sku Standard --custom-data "$ci" \
      --tags ManagedBy=cci-demo-run -o tsv --query publicIpAddress 2>&1)
    # az prints warnings before the IP, so look for the IP line rather than matching the whole output.
    ip=$(echo "$out" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
    if [ -n "$ip" ]; then echo "vm: launched cci-demo-vm $region $ip"; break; fi
    echo "vm: $region failed:"; echo "$out" | grep -v '^WARNING' | tail -4 | sed 's/^/    /'
    az vm delete -g "$RG" -n cci-demo-vm --yes >/dev/null 2>&1
  done
  rm -f "$ci"
  nsg=$(az network nsg list -g "$RG" --query "[?starts_with(name,'cci-demo-vm')].name | [0]" -o tsv)
  [ -n "$nsg" ] && az network nsg rule create -g "$RG" --nsg-name "$nsg" -n ssh-from-here --priority 100 \
      --source-address-prefixes "$MYIP/32" --destination-port-ranges 22 --protocol Tcp --access Allow -o none \
    && echo "vm: SSH open to $MYIP only; vLLM listens on the VM's localhost"
}

aca() {
  local fqdn pw
  fqdn=$(az containerapp show -g "$RG" -n "$ACA_APP" --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null)
  if [ -n "$fqdn" ]; then
    echo "aca: $ACA_APP exists ($fqdn); scaling to 1 replica"
    az containerapp update -g "$RG" -n "$ACA_APP" --min-replicas 1 --max-replicas 1 -o none
    return 0
  fi
  pw=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)
  envid=$(az containerapp env show -g "$RG" -n "$ACA_ENV" --query id -o tsv)
  loc=$(az containerapp env show -g "$RG" -n "$ACA_ENV" --query location -o tsv)
  # A PUT to Resource Manager: the preview containerapp extension's --yaml path fails to parse.
  spec=$(mktemp --suffix .json)
  ACR=$ACR PW=$pw KEY=$VLLM_KEY ENVID=$envid LOC=$loc REPACK=$REPACK python3 - > "$spec" <<'PY'
import json, os
e = os.environ
print(json.dumps({
  "location": e["LOC"], "tags": {"ManagedBy": "cci-demo-run"},
  "properties": {
    "environmentId": e["ENVID"], "workloadProfileName": "gpu-t4",
    "configuration": {
      "ingress": {"external": True, "targetPort": 8000, "transport": "http"},
      "registries": [{"server": f"{e['ACR']}.azurecr.io", "username": e["ACR"], "passwordSecretRef": "acr-pw"}],
      "secrets": [{"name": "acr-pw", "value": e["PW"]}, {"name": "vllm-key", "value": e["KEY"]}]},
    "template": {
      "scale": {"minReplicas": 1, "maxReplicas": 1},
      "containers": [{
        "name": "vllm", "image": f"{e['ACR']}.azurecr.io/vllm-turing:v0.30.0",
        "resources": {"cpu": 8, "memory": "56Gi"},
        "env": [{"name": "VLLM_API_KEY", "secretRef": "vllm-key"}],
        "args": ["--model", e["REPACK"], "--served-model-name", e["REPACK"], "--max-model-len", "8192",
                 "--gpu-memory-utilization", "0.90", "--enable-auto-tool-choice", "--tool-call-parser", "gemma4",
                 "--host", "0.0.0.0", "--port", "8000"]}]}}}))
PY
  echo "aca: creating $ACA_APP on the T4 profile in $ACA_ENV ($loc)"
  az rest --method put --url "https://management.azure.com${envid%/providers/*}/providers/Microsoft.App/containerApps/$ACA_APP?api-version=2024-03-01" \
    --body "@$spec" -o tsv --query 'properties.[provisioningState,configuration.ingress.fqdn]' 2>&1 | tail -3
  rm -f "$spec"
}

want=${*:-vm aca}
echo "$(date -u +%FT%TZ) up.sh start: $want"
pids=()
for b in $want; do
  case $b in
    vm)  vm  2>&1 | sed 's/^/  /' & ;;
    aca) aca 2>&1 | sed 's/^/  /' & ;;
    *) echo "unknown backend: $b (use vm aca)"; continue ;;
  esac
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date -u +%FT%TZ) up.sh done"
echo "Next: stage/forward.sh && source stage/demo.env && python3 stage/check.py --wait 1800"
exit $rc
