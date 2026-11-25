#!/bin/bash
# Step 1 on a fresh Ubuntu 22.04 GPU server (what was run on the UChicago RTX 6000, 2026-11-25).
#   bash scripts/setup_server.sh            # everything except the Kaldi build
#   BUILD_KALDI=1 bash scripts/setup_server.sh
# Layout: venv ~/venvs/asr, tools ~/persian_asr_local/tools, data ~/persian_asr_data.
# Needs sudo for apt. Python 3.12 via uv (the MDC SDK needs >= 3.11; Ubuntu 22.04 has 3.10).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
TOOLS=$HOME/persian_asr_local/tools
DATA=${PERSIAN_ASR_DATA:-$HOME/persian_asr_data}
VENV=$HOME/venvs/asr
CAT_COMMIT=$(python3 -c "import re;print(re.search(r'CAT_COMMIT = \"(\w+)\"', open('$REPO/src/persian_asr/config.py').read()).group(1))")
mkdir -p "$TOOLS" "$DATA"/{logs,pretrained}

sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake libopenblas-dev liblapack-dev liblapacke-dev \
  ffmpeg sox libsox-dev git-lfs python3-pip python3-dev zlib1g-dev libbz2-dev liblzma-dev libboost-all-dev \
  libeigen3-dev automake autoconf libtool unzip wget gfortran pigz rsync bc
command -v rclone >/dev/null || { curl -sLo /tmp/rclone.deb https://downloads.rclone.org/rclone-current-linux-amd64.deb && sudo dpkg -i /tmp/rclone.deb; }

python3 -m pip install -q --user uv==0.9.5
export PATH=$HOME/.local/bin:$PATH
[ -x "$VENV/bin/python" ] || { uv python install 3.12 && uv venv -q -p 3.12 "$VENV"; }
export VIRTUAL_ENV=$VENV
# torch 2.14.1 cu126 runs on driver 545 (CUDA 12.3) and includes sm_75 (Quadro RTX 6000).
uv pip install -q torch==2.14.1 torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -q -r "$REPO/requirements-dev.txt" datacollective==0.8.1
[ -d "$TOOLS/CAT/.git" ] || git clone -q https://github.com/thu-spmi/CAT.git "$TOOLS/CAT"
git -C "$TOOLS/CAT" checkout -q "$CAT_COMMIT"
"$VENV/bin/python" "$REPO/scripts/patch_cat.py" "$TOOLS/CAT"
uv pip install -q "$REPO/third_party/cat_ext_stubs"
uv pip install -q --no-deps -e "$TOOLS/CAT"
uv pip install -q "git+https://github.com/webdataset/webdataset.git@d7334016f44a03c4a385971aa835c4f460d3f30a" \
  https://github.com/kpu/kenlm/archive/master.zip
uv pip install -q -e "$REPO"

"$VENV/bin/python" - <<EOF
from pathlib import Path
from persian_asr import config, install, whistle
p = config.Paths(drive_root=Path("$DATA"), local_root=Path("$HOME/persian_asr_local")).ensure()
print("kenlm bins:", install.ensure_kenlm_bins(p))
print("whistle-small:", whistle.download_checkpoint("whistle-small", Path("$DATA/pretrained")))
EOF
if [ "${BUILD_KALDI:-0}" = 1 ]; then
  bash "$REPO/scripts/build_kaldi_min.sh" "$TOOLS/kaldi" "$TOOLS/CAT" "$(nproc)"
fi
cd "$REPO" && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -m pytest -q
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
