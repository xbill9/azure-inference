#!/usr/bin/env bash
# Open an SSH tunnel to the A10 VM, look up every other backend, and write stage/demo.env.
#
#   stage/forward.sh && source stage/demo.env
#
# localhost:8011 -> cci-demo-vm:8000 (vLLM listens on the VM's localhost only)
# Re-running it closes the old tunnel first.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pidfile=$here/.forwards.pid
RG=${RG:-cci-demo-rg}
ACA_APP=${ACA_APP:-cci-demo-vllm}
FOUNDRY_RG=${FOUNDRY_RG:-research-mesh-rg}
FOUNDRY=${FOUNDRY:-research-mesh-foundry}
REPACK=${REPACK:-xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4}

if [ -f "$pidfile" ]; then
  while read -r p; do kill "$p" 2>/dev/null; done < "$pidfile"
  rm -f "$pidfile"
fi

key=$(az cognitiveservices account keys list -g "$FOUNDRY_RG" -n "$FOUNDRY" --query key1 -o tsv)
env_lines=("export FOUNDRY_URL=https://$FOUNDRY.services.ai.azure.com/openai/v1"
           "export FOUNDRY_KEY=$key FOUNDRY_MODEL=cci-gpt41-nano PHI_MODEL=cci-phi4-mini"
           "export VLLM_KEY=$(cat "$here/.vllm-key")")
echo "foundry: cci-gpt41-nano, cci-phi4-mini"

ip=$(az vm show -d -g "$RG" -n cci-demo-vm --query publicIps -o tsv 2>/dev/null)
if [ -n "$ip" ]; then
  ssh -N -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
      -L 8011:localhost:8000 "azureuser@$ip" > "$here/.fwd-vm.log" 2>&1 &
  echo $! >> "$pidfile"
  env_lines+=("export VM_URL=http://localhost:8011/v1 VM_MODEL=$REPACK")
  echo "vm: localhost:8011 -> $ip:8000"
else
  echo "vm: not running, skipped"
fi

fqdn=$(az containerapp show -g "$RG" -n "$ACA_APP" --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null)
if [ -n "$fqdn" ]; then
  env_lines+=("export ACA_URL=https://$fqdn/v1 ACA_MODEL=$REPACK")
  echo "aca: https://$fqdn"
else
  echo "aca: $ACA_APP not found, skipped"
fi

printf '%s\n' "${env_lines[@]}" > "$here/demo.env"
chmod 600 "$here/demo.env"
sleep 3
echo; echo "Wrote stage/demo.env. Next: source stage/demo.env && python3 stage/check.py"
