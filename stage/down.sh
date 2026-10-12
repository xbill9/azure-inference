#!/usr/bin/env bash
# Stop everything the demo bills for by the hour, then list what is left.
#
#   stage/down.sh          # delete the VM (and its disk, NIC, IP), delete the Container App
#   stage/down.sh --all    # also delete resource group cci-demo-rg and the two Foundry deployments
#
# Without --all it keeps the Container Apps environment and the registry (no hourly GPU
# charge; the Basic registry bills a small daily fee) so the next run skips the image build.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
RG=${RG:-cci-demo-rg}
ACA_APP=${ACA_APP:-cci-demo-vllm}
FOUNDRY_RG=${FOUNDRY_RG:-research-mesh-rg}
FOUNDRY=${FOUNDRY:-research-mesh-foundry}

if [ -f "$here/.forwards.pid" ]; then
  while read -r p; do kill "$p" 2>/dev/null; done < "$here/.forwards.pid"; rm -f "$here/.forwards.pid"
fi

if [ "${1:-}" = --all ]; then
  for d in cci-gpt41-nano cci-phi4-mini; do
    az cognitiveservices account deployment delete -g "$FOUNDRY_RG" -n "$FOUNDRY" --deployment-name "$d" \
      && echo "foundry: deleted $d"
  done
  echo "rg: deleting $RG (VM, Container Apps, registry, logs)"
  az group delete -n "$RG" --yes
else
  if az vm show -g "$RG" -n cci-demo-vm >/dev/null 2>&1; then
    echo "vm: deleting cci-demo-vm with its disk, NIC and public IP"
    az vm delete -g "$RG" -n cci-demo-vm --yes --force-deletion none
    for t in Microsoft.Network/networkInterfaces Microsoft.Network/publicIPAddresses Microsoft.Compute/disks; do
      for id in $(az resource list -g "$RG" --resource-type "$t" --query "[?starts_with(name,'cci-demo-vm')].id" -o tsv); do
        az resource delete --ids "$id" >/dev/null && echo "  deleted ${id##*/}"
      done
    done
  fi
  if az containerapp show -g "$RG" -n "$ACA_APP" >/dev/null 2>&1; then
    echo "aca: deleting $ACA_APP"
    az containerapp delete -g "$RG" -n "$ACA_APP" --yes -o none
  fi
fi

echo; echo "Sweep (VMs, container apps, Foundry deployments still present):"
az vm list -o tsv --query '[].[name,location,hardwareProfile.vmSize]' | sed 's/^/  vm /'
az containerapp list -o tsv --query '[].[name,location,properties.template.scale.minReplicas]' | sed 's/^/  aca /'
az cognitiveservices account deployment list -g "$FOUNDRY_RG" -n "$FOUNDRY" -o tsv --query '[].name' | sed 's/^/  foundry /'
echo "(sweep done)"
