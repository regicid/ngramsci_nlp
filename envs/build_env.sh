#!/bin/bash
# Build one environment kind with uv and install this repo into it (editable, so `git pull` updates the code).
#   envs/build_env.sh <kind> <prefix> [cuda]      kind: propp | vllm | quotes | rubriques | assemble | portrayal
#   cuda: cu121 (V100: torch 2.4.1, the newest wheels with sm_70) or cu128/cu130 (A100/H100/L40S/Ada). Default cu128.
# Pinned versions for the propp env (the one validated bit-for-bit against propp): envs/propp-cu121.lock.txt.
# On network filesystems, pack the result afterwards: envs/pack_squashfs.sh <prefix>.
set -euo pipefail
KIND=$1; P=$2; CU=${3:-cu128}; HERE=$(cd "$(dirname "$0")/.." && pwd)
export UV_CACHE_DIR=${UV_CACHE_DIR:-$(dirname $P)/.uvcache} UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$(dirname $P)/.uvpython}
uv venv --python 3.11 --python-preference only-managed "$P"      # managed python: also present on compute nodes
PIP="uv pip install --python $P/bin/python"
case $KIND in
  propp)
    if [ "$CU" = cu121 ]; then $PIP -r "$HERE/envs/propp-cu121.lock.txt" --index-strategy unsafe-best-match \
         --extra-index-url https://download.pytorch.org/whl/cu121 "setuptools<80"
    else $PIP torch --index-url https://download.pytorch.org/whl/$CU; $PIP "cupy-cuda12x" "setuptools<80"; fi
    $PIP -e "$HERE[ner,assemble]"
    $PIP "fr_dep_news_trf @ https://github.com/explosion/spacy-models/releases/download/fr_dep_news_trf-3.8.0/fr_dep_news_trf-3.8.0-py3-none-any.whl" ;;
  vllm)      $PIP -e "$HERE[qwen]" ;;
  quotes)    $PIP torch --index-url https://download.pytorch.org/whl/$CU; $PIP -e "$HERE[quotes]" ;;
  rubriques|portrayal|assemble)
             $PIP torch --index-url https://download.pytorch.org/whl/$CU; $PIP -e "$HERE[$KIND]" ;;
  *) echo "unknown kind $KIND"; exit 1 ;;
esac
$P/bin/python -c "import ngramsci; print('ngramsci ok in $P')"
