"""Run commands from this repo on Modal. Invoke from the repo root.

Workflow: get a command working locally first (tiny config, --device cpu or mps), then run the
same command here with --device cuda and the real config. Modal bills per second while a
container is up, so every run has a timeout (default 1h), the default GPU is a cheap L4, and
any run whose worst-case cost (running until the timeout) exceeds $1 asks for confirmation.

    uv run modal run modal_run.py::download_data          # one-time: fetch datasets into the data volume
    uv run modal run modal_run.py --cmd "nvidia-smi"      # sanity check on the default L4
    uv run modal run modal_run.py --gpu H100 --timeout 5400 --cmd "python <script> --device cuda ..."
    uv run modal run modal_run.py --gpu none --cpu 16 --cmd "python <script> ..."  # CPU-only job
    uv run modal run --detach modal_run.py ...            # keep running after you close the terminal

Inside the container the repo is the working directory (/root), the data volume is mounted at
/root/data and the runs volume at /root/runs, so relative paths like data/owt_train.txt resolve
the same locally and remotely. Write checkpoints/logs under runs/ to keep them.

    uv run modal volume ls cs336-a1-runs
    uv run modal volume get cs336-a1-runs <remote_path> <local_path>
    uv run modal volume put cs336-a1-data <local_file> <remote_path>
"""

import gzip
import shutil
import subprocess
import urllib.request
from pathlib import Path

import modal

REMOTE_ROOT = "/root"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
CONFIRM_ABOVE_USD = 1.0

app = modal.App("cs336-a1")
data_volume = modal.Volume.from_name("cs336-a1-data", create_if_missing=True, version=2)
runs_volume = modal.Volume.from_name("cs336-a1-runs", create_if_missing=True, version=2)
VOLUMES = {f"{REMOTE_ROOT}/data": data_volume, f"{REMOTE_ROOT}/runs": runs_volume}

# Not shipped with the code: data/ and runs/ come from volumes; the rest is local-only or large.
IGNORE = [
    ".venv",
    ".git",
    "data",
    "runs",
    "wandb",
    "**/__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "*.pdf",
    "*.zip",
    ".DS_Store",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .env({"PYTHONPATH": REMOTE_ROOT, "PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(".", REMOTE_ROOT, ignore=IGNORE)
)

# Same sources as README.md; .gz files are decompressed while streaming.
DATASETS = {
    "TinyStoriesV2-GPT4-train.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "owt_train.txt": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
    "owt_valid.txt": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
}


@app.function(image=image, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT_SECONDS)
def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=REMOTE_ROOT)
    finally:
        # Persist whatever was written, even if the command failed or was interrupted.
        data_volume.commit()
        runs_volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=DEFAULT_TIMEOUT_SECONDS)
def download_data() -> None:
    data_dir = Path(REMOTE_ROOT) / "data"
    for name, url in DATASETS.items():
        dest = data_dir / name
        if dest.exists():
            print(f"skip {name} (already in volume)", flush=True)
            continue
        print(f"downloading {name} ...", flush=True)
        part = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url) as resp, open(part, "wb") as f:
            src = gzip.GzipFile(fileobj=resp) if url.endswith(".gz") else resp
            shutil.copyfileobj(src, f, length=16 << 20)
        part.rename(dest)
        data_volume.commit()
        print(f"  {name}: {dest.stat().st_size / 1e9:.2f} GB", flush=True)


def worst_case_cost(gpu_spec: str | None, cpu: float, hours: float) -> float | None:
    """Cost if the run lasts until its timeout (GPU + CPU; memory is extra but small). None if the GPU rate is unknown."""
    rates = dict(modal.Workspace.from_context().billing.rates())
    per_hour = float(rates["cpu_hour_cost"]) * max(cpu, 1.0)
    if gpu_spec:
        name, _, count = gpu_spec.partition(":")
        name = name.lower().replace("-", "_")
        key = f"gpu_hour_cost_{'a100_40gb' if name == 'a100' else name}"
        if key not in rates:
            return None
        per_hour += float(rates[key]) * int(count or 1)
    return per_hour * hours


@app.local_entrypoint()
def main(
    cmd: str, gpu: str = "L4", timeout: int = DEFAULT_TIMEOUT_SECONDS, cpu: float = 0.0, yes: bool = False
) -> None:
    gpu_spec = None if gpu.lower() in ("", "none", "cpu") else gpu
    cost = worst_case_cost(gpu_spec, cpu, timeout / 3600)
    month = float(modal.Workspace.from_context().billing.summary().billed_cost)
    print(
        f"Modal run: gpu={gpu_spec or 'none'} cpu={cpu or 'default'} timeout={timeout}s | "
        f"worst case {'unknown' if cost is None else f'~${cost:.2f}'} | billed this month ${month:.2f}",
        flush=True,
    )
    # Anything that could cost more than a dollar needs an explicit yes (or --yes).
    if (cost is None or cost > CONFIRM_ABOVE_USD) and not yes:
        try:
            answer = input("Launch? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            raise SystemExit("Aborted, nothing was launched.")
    run.with_options(gpu=gpu_spec, cpu=cpu or None, timeout=timeout).remote(cmd)
