#!/usr/bin/env bash
# Runs as root at every boot of the A10 VM (cloud-init installs it as a systemd unit).
# Idempotent: each step is skipped once done. The NVadsA10 v5 sizes are vGPU-backed,
# so they need NVIDIA's GRID driver, which Azure publishes; the datacenter CUDA
# driver does not bind to them.
#
# Boot 1: blacklist nouveau and reboot.  Boot 2: GRID driver, container toolkit, vLLM.
set -euxo pipefail
exec >>/var/log/cci-setup.log 2>&1
echo "=== $(date -u +%FT%TZ) cci-setup boot"
. /etc/cci.env   # REPACK, VLLM_KEY, VLLM_IMAGE

if [ ! -f /etc/modprobe.d/blacklist-nouveau.conf ]; then
  printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/blacklist-nouveau.conf
  update-initramfs -u
  echo "$(date -u +%FT%TZ) reboot for nouveau"; reboot; exit 0
fi

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential dkms "linux-headers-$(uname -r)" docker.io curl gpg
  curl -fsSL -o /tmp/grid.run 'https://go.microsoft.com/fwlink/?linkid=874272'
  sh /tmp/grid.run --silent --dkms
  cp /etc/nvidia/gridd.conf.template /etc/nvidia/gridd.conf
  printf 'IgnoreSP=FALSE\nEnableUI=FALSE\n' >> /etc/nvidia/gridd.conf
  sed -i '/^FeatureType=0/d' /etc/nvidia/gridd.conf
  echo "$(date -u +%FT%TZ) GRID driver installed"
fi
nvidia-smi

if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

if ! docker inspect vllm >/dev/null 2>&1; then
  docker run -d --name vllm --restart unless-stopped --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
    "$VLLM_IMAGE" \
    --model "$REPACK" --served-model-name "$REPACK" \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --api-key "$VLLM_KEY"
  echo "$(date -u +%FT%TZ) vllm container started"
fi
