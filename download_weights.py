"""
Download the entire ``results/`` folder from Google Drive.

Retrieves all training runs (weights and configs) into the local
``results/`` directory. Re-running is safe — existing files are skipped.

Usage::

    python download_weights.py

Requires::

    pip install gdown
"""

import sys
from pathlib import Path

RESULTS_FOLDER_ID = "1pgce5w-3qzyzj97sQmXuHC3qosv5lCLj"
RESULTS_DIR = Path("results")


def _check_gdown() -> None:
    """Raise a clear error if gdown is not installed."""
    try:
        import gdown 
    except ImportError:
        print(
            "ERROR: 'gdown' is not installed.\n"
            "Run:  pip install gdown\n"
            "Then re-run this script."
        )
        sys.exit(1)




def download_results() -> None:
    """Download the full results folder from Google Drive."""
    import gdown

    _check_gdown()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Downloading results folder (ID: {RESULTS_FOLDER_ID}) ...")
    url = f"https://drive.google.com/drive/folders/{RESULTS_FOLDER_ID}"

    gdown.download_folder(
        url=url,
        output=str(RESULTS_DIR),
        quiet=False,
        use_cookies=False,
    )

    runs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    print(f"\nDone — {len(runs)} run(s) available in '{RESULTS_DIR}/':")
    for run in sorted(runs):
        weights = run / "model_weights.pth"
        status = "OK" if weights.exists() else "missing weights"
        print(f"  {run.name}  [{status}]")


if __name__ == "__main__":
    download_results()
