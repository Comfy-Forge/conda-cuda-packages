# One build script, three modes. Templated verbatim into every generated
# recipe, so a shard job and its link job run byte-identical compiles — which
# is what makes the ccache handoff replay instead of recompile.
#
#   CUW_MODE=full   compile + install (unsharded packages)
#   CUW_MODE=shard  compile only this shard's slice, then exit 0
#   CUW_MODE=link   full build, every compile expected to be a cache hit
#
# The environment this relies on is set in the recipe's build.script.env,
# NOT by the workflow: rattler-build hands the script a clean environment and
# the outer shell's exports do not reach it (measured, step-0 probe).
set -euo pipefail

CUW_DIR="$(dirname "${CUW_LEDGER:-/tmp/cuw/ledger.txt}")"
mkdir -p "$CUW_DIR" "$CCACHE_DIR"
: > "$CUW_LEDGER"

CCACHE_BIN="$(command -v "${CUW_CCACHE_BIN:-ccache}" || echo "${CUW_CCACHE_BIN:-ccache}")"
if ! "$CCACHE_BIN" --version >/dev/null 2>&1; then
  echo "::error::ccache not usable at '$CCACHE_BIN' -- the shard handoff and the compile ledger both run through it" >&2
  exit 1
fi
# ccache 3.x has no "cu" entry in its source-language table, so every TU the
# build compiles as `-x cu` is passed through UNCACHED. That is not a slow
# cache, it is no cache, and the shard lane would silently void itself.
CC_MAJOR="$("$CCACHE_BIN" --version | head -1 | grep -oE '[0-9]+' | head -1)"
if [ "${CC_MAJOR:-0}" -lt 4 ]; then
  echo "::error::ccache $CC_MAJOR.x found; >= 4 required (3.x cannot cache '-x cu')" >&2
  exit 1
fi

# ---- the wrapper goes in the nvcc SEAT, not on PATH ---------------------
# torch's cpp_extension invokes "$CUDA_HOME/bin/nvcc" by absolute path, so a
# PATH-based shim is simply never consulted. Move the real binary aside and
# occupy its filename.
if [ -f "$PREFIX/bin/nvcc" ] && [ ! -f "$PREFIX/bin/nvcc.real" ]; then
  mv "$PREFIX/bin/nvcc" "$PREFIX/bin/nvcc.real"
  install -m 0755 "$RECIPE_DIR/nvcc-wrap.sh" "$PREFIX/bin/nvcc"
fi
export CUW_REAL_NVCC="$PREFIX/bin/nvcc.real"
export CUW_CCACHE_BIN="$CCACHE_BIN"

export CUDA_HOME="$PREFIX"
export PYTORCH_NVCC="$PREFIX/bin/nvcc"   # torch's ninja writer honours this
export CUDACXX="$PREFIX/bin/nvcc"        # cmake honours this
# Trailing, so it beats any --threads a setup.py hardcodes earlier in the line.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} --threads ${CUW_NVCC_THREADS:-1}"

echo "=== cuw build ==================================================="
echo "  mode        : ${CUW_MODE}"
echo "  shard       : ${CUW_SHARD_INDEX}/${CUW_SHARD_COUNT}"
echo "  ccache      : $("$CCACHE_BIN" --version | head -1)  dir=$CCACHE_DIR"
echo "  arch list   : ${TORCH_CUDA_ARCH_LIST:-unset}"
echo "  MAX_JOBS    : ${MAX_JOBS:-unset}  nvcc --threads ${CUW_NVCC_THREADS:-1}"
echo "  SRC_DIR     : $SRC_DIR"
echo "  PREFIX      : $PREFIX"
echo "================================================================="

# In shard mode the wrapper stubs out every TU that is not this slice's, so
# the "build" completes in seconds having compiled only its share. In link
# mode it is a pure ccache pass-through and every compile should hit.
if [ "$CUW_MODE" = "shard" ]; then
  export CUW_PARTITION=1
fi

"$CCACHE_BIN" -z >/dev/null 2>&1 || true

BUILD_RC=0
$PYTHON -m pip install . --no-deps --no-build-isolation -vv || BUILD_RC=$?

STATS="$("$CCACHE_BIN" --print-stats 2>/dev/null || true)"
HITS=$(printf '%s\n' "$STATS" | awk -F'\t' '$1 ~ /^(direct_cache_hit|preprocessed_cache_hit)$/ {n+=$2} END{print n+0}')
MISSES=$(printf '%s\n' "$STATS" | awk -F'\t' '$1 == "cache_miss" {n+=$2} END{print n+0}')
COMPILED=$( [ -s "$CUW_LEDGER" ] && wc -l < "$CUW_LEDGER" || echo 0 )
echo "=== ccache: $HITS hit(s) / $MISSES miss(es); ledger records $COMPILED compile(s)"

case "$CUW_MODE" in
  shard)
    # A shard that compiled nothing is not automatically wrong (its slice can
    # be empty), but a shard whose wrapper never ran at all is a real defect:
    # it would ship an empty cache and the link job would recompile the world.
    if [ "$((HITS + MISSES))" -eq 0 ]; then
      echo "::error::shard saw zero ccache lookups -- the wrapper never occupied the nvcc seat" >&2
      exit 1
    fi
    echo "shard $CUW_SHARD_INDEX done; cache populated. Exiting before install."
    exit 0
    ;;
  link)
    if [ "$BUILD_RC" -ne 0 ]; then exit "$BUILD_RC"; fi
    # ZERO tolerance, not a ratio. One miss is a whole TU recompiled, and a
    # percentage cannot tell "one nondeterministic TU" from "four shards built
    # for the wrong architecture".
    if [ "$((HITS + MISSES))" -eq 0 ]; then
      echo "::error::link job saw zero ccache lookups -- no shard cache was restored" >&2
      exit 1
    fi
    if [ "$MISSES" -gt 0 ]; then
      echo "::error::link job had $MISSES ccache miss(es) of $((HITS+MISSES)). Shard caches did not transfer cleanly: flag mismatch, path mismatch, or wrong-architecture artifacts." >&2
      printf '%s\n' "$STATS" | grep -iE 'miss|hit' || true
      exit 1
    fi
    ;;
  full)
    if [ "$BUILD_RC" -ne 0 ]; then exit "$BUILD_RC"; fi
    ;;
esac

# ---- L3: the compile ledger --------------------------------------------
# Every extension module we are about to ship must correspond to at least one
# translation unit this run actually compiled. Catches what the network denial
# cannot see: a binary vendored into the source tree, or a build system that
# copies a prebuilt .so into place.
SITE="$(: | $PYTHON -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
MODULES=$(find "$SITE" -name '*.so' -newermt '-1 day' 2>/dev/null | wc -l)
if [ "$MODULES" -gt 0 ] && [ "$COMPILED" -eq 0 ] && [ "$HITS" -eq 0 ]; then
  echo "::error::installed $MODULES extension module(s) but the compile ledger is empty and nothing was replayed from cache -- this build did not compile what it is shipping" >&2
  exit 1
fi
echo "ledger: $COMPILED compile(s) recorded for $MODULES installed module(s)"
if [ -s "$CUW_RSS_LOG" ]; then
  echo "=== nvcc peak RSS per TU (top 10) ==="
  sort -t= -k2 -nr "$CUW_RSS_LOG" | head -10
fi
