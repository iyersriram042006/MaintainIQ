"""Script to download and validate the UCI AI4I 2020 Predictive Maintenance Dataset.

Dataset Source:
UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset
URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
"""

import argparse
import io
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DATA_DIR, RAW_DATA_FILE, UCI_AI4I_URL
from src.data_loader import load_raw_data


def download_dataset(
    url: str = UCI_AI4I_URL,
    target_file: Path = RAW_DATA_FILE,
    force: bool = False,
    timeout: int = 30,
) -> Path:
    """Download the official UCI AI4I 2020 dataset archive and extract ai4i2020.csv.

    Args:
        url: Direct URL to the UCI dataset zip archive.
        target_file: Target CSV filepath (default: data/raw/ai4i2020.csv).
        force: If True, redownload even if the file already exists.
        timeout: Socket timeout in seconds.

    Returns:
        Path to the downloaded and validated raw CSV file.

    Raises:
        RuntimeError: If download, extraction, or validation fails. Do NOT
            generate synthetic replacement data on failure.
    """
    target_path = Path(target_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        print(f"[MaintainIQ] Dataset already exists at '{target_path}'. Validating...")
        try:
            df = load_raw_data(target_path, validate=True)
            print(f"[MaintainIQ] Existing dataset is valid ({len(df):,} records, {df.shape[1]} columns).")
            return target_path
        except Exception as e:
            print(f"[MaintainIQ] Existing file failed validation ({e}). Re-downloading...")

    print(f"[MaintainIQ] Downloading UCI AI4I 2020 dataset from:\n  {url}")

    headers = {"User-Agent": "MaintainIQ-PredictiveMaintenance/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP request failed with status code {response.status}: {response.reason}"
                )
            content = response.read()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download dataset from '{url}'. Error: {e.reason}. "
            "Please check your internet connection or place 'ai4i2020.csv' manually "
            f"into '{target_path.parent}'."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while downloading from '{url}': {e}. "
            "Download aborted. No synthetic data will be generated."
        ) from e

    print(f"[MaintainIQ] Downloaded {len(content):,} bytes. Extracting archive...")

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            namelist = zf.namelist()
            csv_candidates = [n for n in namelist if n.endswith(".csv")]
            if not csv_candidates:
                raise RuntimeError(
                    f"No CSV file found in downloaded zip archive. Archive contents: {namelist}"
                )
            
            # Prefer 'ai4i2020.csv' or the first CSV file found
            chosen_csv = "ai4i2020.csv" if "ai4i2020.csv" in csv_candidates else csv_candidates[0]
            csv_bytes = zf.read(chosen_csv)

            with open(target_path, "wb") as f_out:
                f_out.write(csv_bytes)

    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"Downloaded content from '{url}' is not a valid zip archive: {e}"
        ) from e

    print(f"[MaintainIQ] Saved raw dataset to '{target_path}'. Validating schema...")
    df = load_raw_data(target_path, validate=True)
    print(f"[MaintainIQ] Successfully verified dataset with {len(df):,} rows and {df.shape[1]} columns.")
    return target_path


def main() -> None:
    """CLI entrypoint for dataset download."""
    parser = argparse.ArgumentParser(
        description="Download and validate the UCI AI4I 2020 Predictive Maintenance dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if ai4i2020.csv already exists.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=UCI_AI4I_URL,
        help="Direct URL of the UCI dataset zip archive.",
    )
    args = parser.parse_args()

    try:
        download_dataset(url=args.url, force=args.force)
    except Exception as e:
        print(f"[MaintainIQ ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
