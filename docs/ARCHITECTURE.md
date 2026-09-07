# Architecture

Reconciled from two independent design passes (build pipeline; packaging and
distribution), each of which read `cuda-wheels`, the earlier
`PozzettiAndrea/conda-cuda-packages` experiment, and the `conda-torch`
channel before writing. Decisions the owner has taken are marked **DECIDED**;
open questions are listed at the end rather than silently resolved.

## What this repo is for

`cuda-wheels` builds 42 CUDA extensions as PyPI wheels. That index works, but
it works by opting out of dependency resolution: the combo is encoded in a
local version tag (`+cu128torch2.8`), **every** `Requires-Dist` is stripped
from **every** wheel (`strip_all_requires_dist`, and re-declaring one is a
hard error in the loader), and the consumer installs by direct URL, which is
`--no-deps` by construction. It is an index that is *selected from*, not
*resolved*.

A conda channel is the opposite: the artifact declares what it needs and a
solver is trusted to satisfy it. So the central work of this repo is not
writing recipes — it is **authoring true dependency metadata for 42 packages
that today declare none**. Recipes are the vehicle.

## Compiled from source, enforced in four layers

The old experiment shipped `flash_attn` conda packages that were not compiled
here: `FLASH_ATTENTION_FORCE_BUILD: "0"` let `pip install .` fetch
Dao-AILab's prebuilt wheel. That is the defect this repo exists to make
impossible. Note also that upstream's check is `os.getenv(...) == "TRUE"` —
the experiment's own comment said "set to 1 to force-build", and `"1"` would
have done nothing.

| layer | mechanism |
|---|---|
| **L1 — no network** | The compile runs with network access permanently removed, so `pip` cannot reach an upstream wheel because it cannot reach anything; source is fetched and patched *outside*, so no legitimate build needs it. **Corrected twice, both by measurement.** (a) This is not rattler-build's default: `--sandbox` is opt-in and a build script without it was verified to `curl` pypi.org. (b) `--sandbox` *also cannot deliver it*: it needs a separate `rattler-sandbox` binary plus unprivileged namespace creation, and fails with `sandboxing failure: Operation not permitted` both in this project's container and on GitHub-hosted runners (run 34095721769) — it installs and then does nothing. The mechanism is now `scripts/build_snippets/nonet.py`: a seccomp filter denying `AF_INET`/`AF_INET6` socket creation, installed by the build immediately before the pip step. Unprivileged, inherited by every child (setup.py, pip, cmake, nvcc), irreversible (`PR_SET_NO_NEW_PRIVS`), narrow enough that `AF_UNIX` still works, and it exits non-zero rather than running unprotected if the filter cannot be installed. It wraps only the compile — not the whole rattler-build invocation, which legitimately needs the network to solve and download environments. Enforced by `scripts/lint_guarantee.py`, proven per-run by the canary. |
| **L2 — declared force-source flags** | `package.yml` carries `force_source_build:` for any upstream whose build can download a binary. The loader hard-errors when it is missing or set to a permissive value — the same "too important to leave unstated" pattern as the mandatory `jobs`/`nvcc_threads`. |
| **L3 — compile ledger** | The compiler wrapper records every translation unit it compiles. A post-build gate refuses any artifact holding an extension module with no matching compile record — catching a binary vendored into the source tree, which L1 cannot see. |
| **L4 — canary** | `recipes/_canary_prebuilt/` proves the guarantee positively — `AF_INET` refused, a real `pip download` refused, `AF_UNIX` intact — and prints a proof line CI greps for; the build must SUCCEED. The first version inferred "network denied" from the build FAILING, which passes when the mechanism is broken, and did: it reported the guarantee in force during the very run where the sandbox never started. |

L1 is the guarantee; L2–L4 exist because one mechanism will not stay true
across 42 packages and a year of changes.

## Pipeline

```
[fetch]  git clone --recursive @pinned-rev → apply patches → source tarball
             (outside rattler-build: submodules, patches, and zero build-time network)
[shard]  N × rattler-build (MODE=shard, slice i) → ccache.tar.gz          # sharded packages only
[link]   1 × rattler-build (MODE=link, merged cache) → .conda
[gate]   zero-miss assert · compile ledger · verify_conda.py
[publish] fragment → repodata → Pages + release assets
```

**The sharding medium is a content-addressed ccache, not object files.**
`cuda-wheels` uses `.o` handoff on Windows but ccache on Linux
(`CCACHE_BASEDIR`, `CCACHE_NOHASHDIR=1`, mtime sloppiness), with the link job
re-running the full build and asserting **zero** misses — zero rather than a
ratio, because one miss is a whole TU recompile and a percentage cannot
distinguish a flaky TU from four shards built for the wrong architecture.
A path-independent cache is also the only handoff medium that can plausibly
survive `rattler-build` relocating the build, which is why it matters here.

**Measured 2026-09-07 — it survives.** A cache populated by one
`rattler-build` run replayed at zero new misses in a second run in a
different work dir *and* a different prefix (`cache_miss` 1 → 1,
`direct_cache_hit` 0 → 1), despite conda activation injecting genuinely
path-dependent flags (`-fdebug-prefix-map=$SRC_DIR=…`, `-isystem
$PREFIX/include`); `CCACHE_BASEDIR` + `CCACHE_NOHASHDIR=1` absorb them. No
fallback needed. Caveat kept honest: the probe used a C translation unit,
and `-x cu` is precisely where older ccache silently fails, so the result
deserves re-measuring on real `.cu` TUs at scale rather than inferring.

Two consequences found while proving it: rattler-build hands the build
script a **clean environment**, so every `CCACHE_*` setting must live in
`build.script.env` (exporting from the workflow reaches nothing); and CUDA
`-dev` packages must **not** be pinned to an exact minor — `cuda-cudart-dev
12.8.*` forces `cuda-version >=12.8,<12.9`, which is UNSAT against a pytorch
whose triton pin is cuda129-only, the same conflict conda-torch hit.

## Packaging

Recipe shape for a torch-linked extension:

```yaml
build:
  string: cuda${cuda_short}_torch${torch_nodot}_py${py_nodot}_h${hash}_${build_number}
requirements:
  host:
    - pytorch ${torch_minor}.* cuda${cuda_short}_*   # the exact torch it compiles against
    - cuda-cudart-dev
    - <only the CUDA -dev packages this source #includes>
  run:
    - python
    - pytorch ${torch_minor}.* cuda${cuda_short}_*   # flavour lock, see below
```

- **Never pin an exact torch build.** `run_exports` from conda-torch's
  `libtorch` yields a *minor range*, and torch's C++ ABI is minor-stable. An
  exact build pin would have been invalidated by conda-torch's 300-build
  republish wave in Sept 2026.
- **A minor range alone is not enough.** `pytorch >=2.11,<2.12` matches
  cu126, cu128, cu129 *and* cu130 builds, so torch flavour and extension
  flavour would be two independent solver choices. `cuda-version` does not
  help: it constrains the CUDA *major* only. Hence an explicit build-glob
  dependency, which `run_exports` structurally cannot generate (it derives
  from the host package's version, not its build string).
- **No selector metapackages here.** conda-torch needed them because a bare
  `pytorch = "*"` is a natural user spec. Here the natural spec is
  `flash-attn = "*"`, and the build glob propagates the flavour choice up to
  torch by itself. One mechanism, not two.
- **`purls` on every artifact**, so a pixi solve's PyPI half recognises the
  package and does not install a second copy from PyPI over it — the failure
  conda-torch proved with torch. Conda names are the hyphenated PyPI names
  (`flash-attn`, not `flash_attn`); the underscore import name is recorded
  separately for the import test.
- **No CPU stubs.** The experiment's `track_features` stub cannot fire in
  this stack — conda-torch's `libtorch` hard-depends on `__cuda`, so a
  GPU-less host has no pytorch from us at all — and it contradicts
  comfy-env's precise-degradation design, which wants a named reason rather
  than a `ModuleNotFoundError` deep inside a node.

## Name overlap with conda-forge — **DECIDED: carry complete**

Under strict channel priority, carrying *any* build of a name hides
conda-forge's builds of that name entirely. Nine of our 42 exist there:

`detectron2, flash_attn, gsplat, mmcv, pyg_lib, pytorch3d, torch_scatter,
torchao, torchsparse`

(Anaconda's `defaults`/`main` has two of them, `flash_attn` and `torchao`,
but pixi never adds that channel and conda-torch's README already tells users
to drop it.)

**Decision: carry each of the nine completely** — publish every
(torch × cuda × python) cell a consumer might want, and own that obligation.
The alternative, publishing a partial flavour set, removes conda-forge's
coverage for anyone listing us first: exactly the trap where one partial
triton side-repack hid conda-forge's whole triton line and produced a
64-entry UNSAT in conda-torch.

## Matrix

Driven by **conda-torch's repodata**, not a scraped PyPI matrix: if a torch
build is not in the channel, an extension for it is unbuildable by
construction and should never be emitted. A torch republish wave becomes
visible automatically, and a torch bump costs zero file changes here.

## Recipes: generated, committed, drift-checked

One Jinja template → `recipes/<name>/recipe.yaml`, committed so they are
reviewable and runnable by hand, with CI asserting regeneration is a no-op.
The experiment's three hand-written recipes were ~90% identical text and had
already drifted (`FORCE_BUILD` set in one and not the others, two divergent
Windows blocks); 42 of them would be worse. Escape hatches, in order:
`arch_override.yml` → `build_script_extra:` → `patches/*.py` → a hand-written
recipe with a README explaining why.

## Memory and runners

The physics is unchanged from the wheel farm — peak ≈ `jobs × nvcc_threads ×
one cicc`, and CUTLASS translation units put one cicc near a 16 GB ceiling —
so the per-package tuning ports verbatim and stays mandatory. `--threads` is
appended *trailing* on `NVCC_APPEND_FLAGS` so it outranks `setup.py`
hardcodes. The wrapper is installed at `$PREFIX/bin/nvcc` with the real one
moved aside, because torch's `cpp_extension` invokes `$CUDA_HOME/bin/nvcc`
directly and bypasses `PATH`.

## Verification

- **`tools/verify_conda.py`** — per-artifact publish gate: build string
  matches the cell it claims; `paths.json` matches payload; no `RECORD`;
  `INSTALLER=conda`; `$ORIGIN`-relative RPATHs with no absolute or empty
  entries; computed (not pasted) glibc/libstdc++ floors; SASS architectures
  match the cell's arch list; no vendored `libtorch`; `purls` present; `run:`
  non-empty; the torch build-glob present; compile ledger covers every
  module; import test.

  Two of those are worth pinning down, because they are properties of
  rattler-build rather than of any recipe. `direct_url.json` and `RECORD` are
  removed by the shared build script after `pip install .`, since a path
  install records this runner's `$SRC_DIR` and would make `pip freeze` print
  a `file://` URL. `INSTALLER` is **not** written there and cannot be: measured
  on rattler-build 0.75.0, a build script that leaves exactly `conda` (5 bytes,
  checked with `od` at the end of the script) still yields `conda\n` (6 bytes)
  in the packaged artifact, because rattler-build normalises the file during
  packaging and there is no recipe knob for it. A byte-exact `== b"conda"`
  assertion is therefore unsatisfiable; the check has to compare stripped.
- **`tools/sweep_solve.py`** — one live solve per cell from the published
  channel, asserting resolution from our release URL *and* that the resolved
  `pytorch` build string's flavour equals the extension's.
- **GPU smoke** — import plus the package's own minimal op on real hardware;
  a mis-arch'd kernel shows up as `cudaErrorNoKernelImageForDevice` and
  nowhere else.

## Scope

v1 is **linux-64 only** — prove the cache handoff where the compiler story is
simplest. `target_platform` stays in the build string and the template keeps
an unwired `win` branch; linux-aarch64 is platform #2.

## Staging

| step | lands | proves |
|---|---|---|
| 0 | ccache-across-rattler-build experiment on `cc_torch` | the sharding architecture, or forces a named fallback |
| 1 | loader, policy, repodata-driven matrix, one generated recipe, one full build | end to end, from source, against conda torch |
| 2 | `verify_conda.py`, canary, ledger gates | the from-source guarantee is mechanical |
| 3 | shard/link on `flash_attn` with force-source on | the heavy path, and the first honestly-compiled flash-attn |
| 4 | the remaining 40 packages | scale |
| 5 | linux-aarch64, then win-64 | breadth |

## Open questions

1. **conda-torch publishes no `<subdir>/run_exports.json`** (404 today). The
   data is inside each artifact, but a builder resolving host dependencies
   reads the channel index. Needs fixing in conda-torch before host-dep
   run_exports resolution can be relied on here.
2. **How many packages genuinely need network during build** (a
   `FetchContent`, a version probe). Unknown until measured; the fix is
   vendoring into the source-fetch phase, and the count should be measured by
   running them, not guessed.
3. **`links_torch: false` packages** (`cumm`, `spconv`) need no torch axis at
   all: their build string drops `torchNNN` and their cell count collapses by
   roughly 6×.
