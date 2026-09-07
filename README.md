# conda-cuda-packages

Conda packages for the CUDA Python extensions ComfyUI node packs depend on —
`flash-attn`, `natten`, `spconv`, `gsplat`, `pytorch3d` and ~40 more —
**compiled from source** against the
[conda-torch](https://github.com/Comfy-Forge/conda-torch) channel.

The sibling of [cuda-wheels](https://github.com/Comfy-Forge/cuda-wheels),
which builds the same packages as PyPI wheels. That index exists because
PyPI has no CUDA/torch variant axis: the combo is smuggled into a local
version tag (`+cu128torch2.8`), dependency metadata is stripped wholesale,
and the installer is told `--no-deps`. Here the combo lives in the conda
build string, dependencies are declared and solved, and the extension is
built against the exact `libtorch` it will run against.

## Using the channel

```toml
[workspace]
channels = [
  "https://comfy-forge.github.io/conda-cuda-packages",
  "https://comfy-forge.github.io/conda-torch",
  "conda-forge",
]
```

Channel order is load-bearing (see conda-torch's README for why). Ask for
the package; the flavour follows from whichever `pytorch` you resolved,
because every extension carries a build-glob dependency on its torch
flavour:

```
flash-attn 2.8.3 cuda128_torch211_py312_h<hash>_0
  depends: pytorch 2.11.* cuda128_*
```

## Compiled from source, provably

No build in this repo may ship a binary it did not compile. That is not a
convention here, it is a property of how the build runs:

- **The build script has no network.** `rattler-build` denies it by
  default and this repo never passes `--allow-network`; source is fetched
  and patched *before* the sandbox. `pip` cannot reach an upstream wheel
  because it cannot reach anything.
- **Force-source flags are declared, not remembered.** Packages whose
  upstream build downloads a prebuilt wheel unless told otherwise (flash-attn's
  `FLASH_ATTENTION_FORCE_BUILD`, and note its check is the string `"TRUE"` —
  `"1"` silently does nothing) declare that flag in `package.yml`; the loader
  hard-errors if it is missing or permissive.
- **A compile ledger.** The compiler wrapper records every translation unit
  it compiles; a post-build gate refuses any artifact containing an
  extension module with no matching compile record.
- **A canary.** `recipes/_canary_prebuilt/` tries to fetch a wheel. CI fails
  if it ever succeeds.

Each artifact records `built_from_source`, the source commit, and the exact
`pytorch` build it compiled against in `about.extra`.

## Layout

| path | purpose |
|---|---|
| `packages/<name>/package.yml` | the per-package facts: source, arch list, parallelism, dependencies |
| `recipes/<name>/recipe.yaml` | generated from `package.yml`, committed, CI asserts no drift |
| `templates/recipe.yaml.j2` | the single recipe template |
| `scripts/` | loader, matrix generation, recipe generation, source fetch/patch |
| `tools/` | artifact verification and channel assembly |
| `docs/ARCHITECTURE.md` | the design and the reasoning behind it |
