#!/bin/bash
# Minimal, static, CPU-only Kaldi build: just what CAT's WFST decoding needs
# (OpenFST + fstbin/latbin/bin tools + the static libs that CAT's
# src/fst-decoder/latgen-faster links against). Also builds CAT's latgen-faster.
#
# Usage: build_kaldi_min.sh <kaldi_dir> <cat_dir> [nj]
#   <kaldi_dir>  where to clone/build Kaldi (e.g. /content/kaldi)
#   <cat_dir>    an existing CAT checkout (e.g. /content/CAT)
#
# Afterwards:  <kaldi_dir>/src/{fstbin,latbin,bin}/...  and  <cat_dir>/src/bin/latgen-faster
# Build is static so the tree can be tarred, moved to Drive and extracted again
# at the same path without LD_LIBRARY_PATH fiddling.
set -euo pipefail

KALDI_DIR=$1
CAT_DIR=$2
NJ=${3:-$(nproc)}
KALDI_COMMIT=${KALDI_COMMIT:-e02e35f0254bb033fab73d1df99fc34123e31d56}

log() { echo "[build_kaldi_min $(date +%H:%M:%S)] $*"; }

if [ ! -d "$KALDI_DIR/.git" ]; then
    log "cloning kaldi @ $KALDI_COMMIT"
    git clone -q https://github.com/kaldi-asr/kaldi.git "$KALDI_DIR"
fi
cd "$KALDI_DIR"
git checkout -q "$KALDI_COMMIT"

log "tools: openfst"
cd "$KALDI_DIR/tools"
make -j "$NJ" openfst >"$KALDI_DIR/build_openfst.log" 2>&1 || { tail -40 "$KALDI_DIR/build_openfst.log"; exit 1; }

# Math library: Ubuntu's packaged OpenBLAS (apt libopenblas-dev), linked statically.
# Kaldi's own tools/extras/install_openblas.sh builds OpenBLAS 0.3.13, which fails to
# compile with gcc 13 on AVX-512 CPUs (verified 2026-11-24: "inlining failed in call to
# always_inline _mm512_add_ps"); the system package avoids that and saves ~10 min.
log "tools: system OpenBLAS shim"
OB_ROOT="$KALDI_DIR/tools/openblas_sys"
OB_INC=/usr/include/x86_64-linux-gnu/openblas-pthread
OB_LIB=/usr/lib/x86_64-linux-gnu/openblas-pthread
[ -f "$OB_INC/cblas.h" ] && [ -f "$OB_LIB/libopenblas.a" ] && [ -f /usr/include/lapacke.h ] || {
    echo "system OpenBLAS not found: apt-get install libopenblas-dev liblapacke-dev"; exit 1; }
rm -rf "$OB_ROOT" && mkdir -p "$OB_ROOT/include"
ln -sfn "$OB_LIB" "$OB_ROOT/lib"
for h in "$OB_INC"/*.h; do ln -sfn "$h" "$OB_ROOT/include/$(basename "$h")"; done
# System lapack.h (LAPACK >= 3.9.1) unconditionally defines LAPACK_FORTRAN_STRLEN_END,
# which adds hidden size_t string-length args that Kaldi's cblas-wrappers.h does not pass
# ("too few arguments to function stptri_"). Use private copies of the LAPACKE headers
# without that define (lapacke.h includes "lapack.h" relative to itself, so copy all).
for h in lapack.h lapacke.h lapacke_config.h lapacke_mangling.h lapacke_utils.h; do
    [ -f "/usr/include/$h" ] && cp "/usr/include/$h" "$OB_ROOT/include/$h"
done
sed -i 's|^#define LAPACK_FORTRAN_STRLEN_END|/* #define LAPACK_FORTRAN_STRLEN_END  (disabled for Kaldi) */|' "$OB_ROOT/include/lapack.h"

log "src: configure"
cd "$KALDI_DIR/src"
./configure --static --static-math=yes --static-fst=yes --use-cuda=no \
    --mathlib=OPENBLAS --openblas-root="$OB_ROOT" \
    >"$KALDI_DIR/build_configure.log" 2>&1 || { tail -40 "$KALDI_DIR/build_configure.log"; exit 1; }

log "src: make depend + bin fstbin latbin lmbin (NJ=$NJ)"
{ make -j "$NJ" depend && make -j "$NJ" bin fstbin latbin lmbin; } >"$KALDI_DIR/build_src.log" 2>&1 \
    || { grep -n "error" "$KALDI_DIR/build_src.log" | head -30; tail -40 "$KALDI_DIR/build_src.log"; exit 1; }

log "CAT fst-decoder (latgen-faster)"
cd "$CAT_DIR/src/fst-decoder"
KALDI_ROOT="$KALDI_DIR" make >"$KALDI_DIR/build_cat_fst_decoder.log" 2>&1 \
    || { tail -40 "$KALDI_DIR/build_cat_fst_decoder.log"; exit 1; }
mkdir -p "$CAT_DIR/src/bin"
cp -f latgen-faster "$CAT_DIR/src/bin/latgen-faster"
# keep a copy inside the kaldi tree so the Drive tarball carries it
cp -f latgen-faster "$KALDI_DIR/src/bin/cat-latgen-faster"

log "shrinking tree (keep only what CAT's graph build + FST decoding call)"
# Unshrunk tree is ~12 GB (static binaries with debug info, ~48 MB each incl. OpenBLAS;
# 2.4 GB openfst sources). Whitelist = tools used by CAT
#   egs/cv-lang10/utils/tool/build_decoding_graph.sh : arpa2fst fstaddselfloops fstisstochastic
#       fsttablecompose fstdeterminizestar fstminimizeencoded (+ openfst fstcompile fstarcsort)
#   cat/ctc/fst_decode.sh : latgen-faster(CAT) lattice-scale lattice-add-penalty
#       lattice-best-path lattice-to-nbest nbest-to-linear
KEEP="bin/cat-latgen-faster lmbin/arpa2fst fstbin/fstaddselfloops fstbin/fstisstochastic
fstbin/fsttablecompose fstbin/fstdeterminizestar fstbin/fstminimizeencoded latbin/lattice-scale
latbin/lattice-add-penalty latbin/lattice-best-path latbin/lattice-to-nbest latbin/nbest-to-linear"
TMPKEEP="$KALDI_DIR/.keep_bins"; rm -rf "$TMPKEEP"; mkdir -p "$TMPKEEP"
for b in $KEEP; do mkdir -p "$TMPKEEP/$(dirname $b)"; cp "$KALDI_DIR/src/$b" "$TMPKEEP/$b"; done
rm -rf "$KALDI_DIR/.git" "$KALDI_DIR/tools/openfst-"*/src
for d in "$KALDI_DIR"/src/*/; do rm -rf "$d"; done          # all of src/ (sources, libs, binaries) ...
find "$KALDI_DIR/src" -maxdepth 1 -type f -delete
cp -r "$TMPKEEP"/* "$KALDI_DIR/src/" && rm -rf "$TMPKEEP"    # ... then the whitelisted binaries back
find "$KALDI_DIR/tools" \( -name '*.o' -o -name '*.a' -o -name '*.la' \) -delete
for f in "$KALDI_DIR"/src/*/* "$KALDI_DIR"/tools/openfst/bin/* "$KALDI_DIR"/tools/openfst/lib/*.so*; do
    [ -f "$f" ] && [ ! -L "$f" ] && strip --strip-unneeded "$f" 2>/dev/null || true
done

for b in $KEEP; do
    [ -x "$KALDI_DIR/src/$b" ] || { echo "MISSING $b"; exit 1; }
done
[ -x "$KALDI_DIR/tools/openfst/bin/fstcompile" ] || { echo "MISSING openfst fstcompile"; exit 1; }
log "OK: kaldi minimal build complete"
