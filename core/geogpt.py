import os
from pathlib import Path
import pandas as pd
import requests

# Hugging Face dataset file (raw download)
HF_DATASET_URL = (
    "https://huggingface.co/datasets/GeoGPT-Research-Project/GeoGPT-CoT-QA/resolve/main/geogpt-cot-qa.csv"
)

# If you want to pin a specific revision later, you can add ?download=1 or a commit hash.
# The resolve/main path is usually fine for most workflows.


def _ensure_parent_dir(filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest_path: str, timeout: int = 120) -> None:
    _ensure_parent_dir(dest_path)

    # Stream download so it works for larger files without loading into memory
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    os.replace(tmp_path, dest_path)


def load_geogpt_csv(
    local_path: str = "data/geogpt_cot_qa.csv",
    hf_url: str = HF_DATASET_URL,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load GeoGPT-CoT-QA CSV.

    Behavior:
    - If local file exists and force_download=False, read it.
    - Otherwise download from Hugging Face to local_path, then read it.

    This keeps the repo clean (no giant CSV committed) while still enabling
    Streamlit Cloud deployment and offline caching after first run.
    """
    if (not force_download) and os.path.exists(local_path):
        return pd.read_csv(local_path)

    # Download and cache
    _download_file(hf_url, local_path)

    # Load
    return pd.read_csv(local_path)
