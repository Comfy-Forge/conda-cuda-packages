#!/usr/bin/env python3
"""Build matrix, derived from the conda-torch channel's own repodata.

cuda-wheels derives its cells from a scraped download.pytorch.org matrix.
Here the source of truth is the channel we build against: if a torch build is
not in conda-torch, an extension for it is unbuildable by construction and
must never be emitted. Two consequences worth stating:

  * a conda-torch republish wave (build numbers _2 -> _3) is visible here
    automatically, and every emitted cell names the EXACT torch build string
    it will compile against;
  * a new torch version appearing in the channel costs zero edits in this
    repo.

Emits matrix.json: a list of job objects (one per shard for sharded
packages), consumed by .github/workflows/build.yml.

Usage:
  generate_matrix.py --package flash-attn [--cuda all] [--pytorch all]
                     [--python all] [--platform all] [-o matrix.json]
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_loader as pl  # noqa: E402

TORCH_CHANNEL = "https://comfy-forge.github.io/conda-torch"
# pytorch build strings look like cuda128_repack_py312_h<hash>_2 or, for the
# mirrored conda-forge builds, cuda129_mkl_py312_h<hash>_302.
_BUILD_RE = re.compile(r"^cuda(\d+)_[a-z]+_py(\d+)_h[0-9a-f]+_(\d+)$")


def fetch_torch_repodata(subdir: str, cache: Path) -> dict:
    """conda-torch's repodata for a subdir, cached on disk."""
    local = cache / f"conda-torch-{subdir}-repodata.json"
    if not local.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{TORCH_CHANNEL}/{subdir}/repodata.json"
        print(f"fetching {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "conda-cuda-packages"})
        local.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return json.loads(local.read_text())


def torch_cells(subdir: str, cache: Path) -> dict:
    """{(cuda, torch_version, python): best pytorch build string}.

    "Best" = highest build number, which is what an unpinned solve would take
    and what a fresh build should compile against. Both our repacks and the
    mirrored conda-forge builds are eligible: an extension linking either is
    equally valid, since the ABI is torch's, not the packaging's.
    """
    d = fetch_torch_repodata(subdir, cache)
    out = {}
    for filename, e in d.get("packages.conda", {}).items():
        if e.get("name") != "pytorch":
            continue
        m = _BUILD_RE.match(e.get("build", ""))
        if not m:
            continue
        cuda_digits, py_digits, build_number = m.groups()
        cuda = f"{cuda_digits[:2]}.{cuda_digits[2:]}"
        python = f"{py_digits[0]}.{py_digits[1:]}"
        key = (cuda, e["version"], python)
        prev = out.get(key)
        if prev is None or int(build_number) > prev[1]:
            out[key] = (e["build"], int(build_number))
    return {k: v[0] for k, v in out.items()}


_PRERELEASE = re.compile(r"[abc]|rc|dev", re.I)


def stable_pythons(subdir: str, cache: Path) -> set:
    """Python minors conda-forge ships a STABLE build of, for this subdir.

    conda-torch carries py3.15 artifacts, but conda-forge's python 3.15 is
    alpha-only (3.15.0a2/a3 today), and a solve will not take a prerelease --
    which is why conda-torch's README says its py3.15 records cannot resolve
    yet. Emitting those cells would queue ~27 jobs per package that cannot
    build a host env. Checked against the live channel rather than pinned to
    a hardcoded ceiling, so this self-heals the day 3.15.0 final lands.
    """
    local = cache / f"cf-pythons-{subdir}.json"
    if not local.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        url = "https://api.anaconda.org/package/conda-forge/python"
        req = urllib.request.Request(url, headers={"User-Agent": "conda-cuda-packages"})
        local.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    files = json.loads(local.read_text()).get("files", [])
    out = set()
    for f in files:
        if (f.get("attrs") or {}).get("subdir") != subdir:
            continue
        v = f.get("version", "")
        if _PRERELEASE.search(v.split(".", 2)[-1] if v.count(".") >= 2 else v):
            continue
        parts = v.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out.add(f"{parts[0]}.{parts[1]}")
    return out


def arch_list_for(cfg: dict, arch_policy: dict, cuda: str, torch_version: str,
                  subdir: str) -> str:
    """Arch list for a cell: per-package override, then exception, then policy."""
    by_cuda = cfg.get("arch_list_by_cuda") or {}
    if cuda in by_cuda:
        return str(by_cuda[cuda])
    if subdir == "linux-aarch64":
        if cfg.get("arch_list_aarch64"):
            return str(cfg["arch_list_aarch64"])
        return str(arch_policy.get("arch_policy_aarch64", {}).get(cuda, "")).replace(";", " ")
    if cfg.get("arch_list"):
        return str(cfg["arch_list"])
    minor = ".".join(torch_version.split(".")[:2])
    exc = arch_policy.get("arch_exceptions", {}).get(f"{cuda}/{minor}")
    if exc:
        return exc.replace(";", " ")
    return str(arch_policy.get("arch_policy", {}).get(cuda, "")).replace(";", " ")


def build_string(cfg: dict, cuda: str, torch_version: str, python: str,
                 build_number: int = 0) -> str:
    """cuda128_torch211_py312_<n>  (the hash is added by the recipe).

    The flavour axis lives here, in the build string, where conda can order
    and match it -- not in a local version tag like the wheel index's
    `+cu128torch2.8`, which conda sorts BELOW the plain version.
    """
    cu = cuda.replace(".", "")
    py = python.replace(".", "")
    if not cfg["links_torch"]:
        return f"cuda{cu}_py{py}_{build_number}"
    tv = ".".join(torch_version.split(".")[:2]).replace(".", "")
    return f"cuda{cu}_torch{tv}_py{py}_{build_number}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--cuda", default="all")
    ap.add_argument("--pytorch", default="all", help="torch MINOR, e.g. 2.11")
    ap.add_argument("--python", default="all")
    ap.add_argument("--platform", default="all")
    ap.add_argument("--build-number", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/conda-cuda-matrix"))
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()

    policy = pl.load_policy()
    arch_policy = pl.load_arch_policy()
    pkg_dir = pl.PACKAGES_DIR / args.package
    if not pkg_dir.is_dir():
        sys.exit(f"unknown package {args.package!r} (packages/{args.package} not found)")
    cfg = pl.load_package(pkg_dir)

    platforms = policy["platforms"] if args.platform == "all" else [args.platform]
    py_min = tuple(int(x) for x in str(policy["python_min"]).split("."))
    jobs = []

    for subdir in platforms:
        cells = torch_cells(policy["torch_subdir"][subdir], args.cache_dir)
        buildable_py = stable_pythons(subdir, args.cache_dir)
        skipped_py = set()
        # links_torch: false -> one build per (cuda, python); collapse the
        # torch axis by keeping one representative torch per (cuda, python).
        seen_no_torch = set()
        for (cuda, torch_version, python), torch_build in sorted(cells.items()):
            if cuda not in policy["supported_cudas"]:
                continue
            if args.cuda != "all" and cuda != args.cuda:
                continue
            minor = ".".join(torch_version.split(".")[:2])
            if args.pytorch != "all" and minor != args.pytorch:
                continue
            if args.python != "all" and python != args.python:
                continue
            if tuple(int(x) for x in python.split(".")) < py_min:
                continue
            if python not in buildable_py:
                skipped_py.add(python)   # conda-forge has no stable build yet
                continue
            if cfg.get("min_pytorch") and _ver(minor) < _ver(str(cfg["min_pytorch"])):
                continue
            if not cfg["links_torch"]:
                if (cuda, python) in seen_no_torch:
                    continue
                seen_no_torch.add((cuda, python))

            arch = arch_list_for(cfg, arch_policy, cuda, torch_version, subdir)
            if not arch:
                continue  # a CUDA line absent from the arch table is not built
            shards = int(cfg.get("sharding") or 1)
            for shard_index in range(1, shards + 1):
                jobs.append({
                    "package": cfg["name"],
                    "folder": args.package,
                    "version": cfg["version"],
                    "source_repo": cfg["source_repo"],
                    "source_rev": cfg["source_rev"],
                    "cuda": cuda,
                    "cuda_short": cuda.replace(".", ""),
                    "pytorch": minor,
                    "pytorch_full": torch_version,
                    "torch_build": torch_build,
                    "python": python,
                    "platform": subdir,
                    "arch_list": arch,
                    "jobs": cfg["jobs"],
                    "nvcc_threads": cfg["nvcc_threads"],
                    "sharding": shards,
                    "shard_index": shard_index,
                    "shard_count": shards,
                    "patch_script": cfg.get("patch_script", ""),
                    "force_source_build": cfg.get("force_source_build") or {},
                    "links_torch": cfg["links_torch"],
                    "clone_recursive": bool(cfg.get("clone_recursive", False)),
                    "free_disk_space": bool(cfg.get("free_disk_space", True)),
                    "nvcc_flags": cfg.get("nvcc_flags", ""),
                    "build_subdir": cfg.get("build_subdir", ""),
                    "runner": policy["runners"][subdir],
                    "build_string": build_string(cfg, cuda, torch_version, python,
                                                 args.build_number),
                })
        if skipped_py:
            print(f"  {subdir}: skipped python {sorted(skipped_py)} -- conda-torch "
                  f"has builds but conda-forge ships no stable python there yet",
                  file=sys.stderr)

    payload = json.dumps(jobs, indent=1)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)
    cells_n = len({(j["cuda"], j["pytorch_full"], j["python"], j["platform"]) for j in jobs})
    print(f"{cfg['name']}: {cells_n} cells, {len(jobs)} jobs "
          f"(sharding {cfg.get('sharding') or 1})", file=sys.stderr)
    return 0


def _ver(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v))


if __name__ == "__main__":
    sys.exit(main())
