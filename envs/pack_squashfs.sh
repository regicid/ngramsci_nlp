#!/bin/bash
# Pack a venv (and the uv-managed python it points to) into squashfs images, for fast imports on network filesystems.
#   envs/pack_squashfs.sh <venv> [<out_dir>]   -> <out_dir>/<name>.sqfs, <out_dir>/<name>_python.sqfs
# Run it as a CPU job: reading the venv's files once from the shared disk is the slow part.
# Use in the sites file:  envs: {vllm: {squashfs: <out_dir>/<name>.sqfs, python_squashfs: <out_dir>/<name>_python.sqfs}}
# Re-pack after changing the env. The repo itself is installed editable, so code updates need no re-pack.
set -euo pipefail
V=$(cd "$1" && pwd); OUT=${2:-$(dirname $V)}; N=$(basename $V)
PYDIR=$(dirname $(dirname $(readlink -f $V/bin/python)))
mksquashfs "$V" "$OUT/$N.sqfs.tmp" -comp lz4 -no-progress -quiet -noappend && mv "$OUT/$N.sqfs.tmp" "$OUT/$N.sqfs"
mksquashfs "$(dirname $PYDIR)" "$OUT/${N}_python.sqfs.tmp" -comp lz4 -no-progress -quiet -noappend && mv "$OUT/${N}_python.sqfs.tmp" "$OUT/${N}_python.sqfs"
ls -la "$OUT/$N.sqfs" "$OUT/${N}_python.sqfs"
