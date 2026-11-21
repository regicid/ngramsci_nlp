# Activate a Python env packed as squashfs images (source this inside a job).
#   source activate_sqfs.sh <env.sqfs> [<python.sqfs>]
# Network filesystems (Ceph, Lustre, GPFS) are slow at opening many small files, and importing torch + vLLM opens
# tens of thousands: from a squashfs image mounted on the node they are read from one big file instead.
# <env.sqfs>: the venv dir packed with `mksquashfs <venv> env.sqfs -comp lz4`; <python.sqfs>: the uv-managed
# interpreter dir the venv points to (optional: without it the stdlib is read from its original place).
_E=$1; _P=${2:-}
_L=${TMPDIR:-/tmp}/ngsq_$$; mkdir -p $_L/env $_L/venv/bin
squashfuse "$_E" $_L/env || { echo "squashfuse failed for $_E" >&2; return 1; }
_PYV=$(ls $_L/env/lib | grep -m1 '^python3')
if [ -n "$_P" ]; then
  mkdir -p $_L/py; squashfuse "$_P" $_L/py || { echo "squashfuse failed for $_P" >&2; return 1; }
  _PYBIN=$(dirname $(ls $_L/py/*/bin/$_PYV | head -1))
else
  _PYBIN=$(dirname $(readlink -f $_L/env/bin/python))
fi
# a tiny venv on local disk whose site-packages is the mounted image
mkdir -p $_L/venv/lib/$_PYV
printf "home = $_PYBIN\ninclude-system-site-packages = false\n" > $_L/venv/pyvenv.cfg
ln -sf $_PYBIN/$_PYV $_L/venv/bin/python; ln -sf python $_L/venv/bin/python3
ln -sfn $_L/env/lib/$_PYV/site-packages $_L/venv/lib/$_PYV/site-packages
for b in $_L/env/bin/*; do n=$(basename $b); [ -e $_L/venv/bin/$n ] || ln -s $b $_L/venv/bin/$n; done
export VIRTUAL_ENV=$_L/venv PATH=$_L/venv/bin:$PATH
trap "fusermount -u $_L/env 2>/dev/null; [ -n '$_P' ] && fusermount -u $_L/py 2>/dev/null" EXIT
