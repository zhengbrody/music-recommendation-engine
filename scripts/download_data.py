"""
Script to download Last.fm dataset.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
import requests
from tqdm import tqdm
import tarfile


def download_file(url: str, output_path: str) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def main():
    """Download Last.fm dataset."""
    config = Config()

    print("="*70)
    print("LAST.FM DATASET DOWNLOAD")
    print("="*70)

    # Last.fm 360K dataset URL
    url = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"

    print(f"\nDataset: Last.fm 360K")
    print(f"Source: {url}")
    print(f"Destination: {config.RAW_DATA_DIR}")

    # Create directory if it doesn't exist
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    # Download
    archive_path = os.path.join(config.DATA_DIR, "lastfm-dataset-360K.tar.gz")

    if os.path.exists(config.RAW_DATA_FILE):
        print(f"\n✓ Dataset already exists at: {config.RAW_DATA_FILE}")
        response = input("Download again? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return

    print("\n[1/3] Downloading dataset...")
    print("This may take several minutes (~200 MB)...")

    try:
        download_file(url, archive_path)
        print("✓ Download completed")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nAlternative: Manual download")
        print(f"1. Download from: {url}")
        print(f"2. Extract to: {config.RAW_DATA_DIR}")
        return

    print("\n[2/3] Extracting archive...")
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(config.DATA_DIR)
        print("✓ Extraction completed")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return

    print("\n[3/3] Organizing files...")
    # The dataset extracts to a folder, we need to move the TSV file
    extracted_folder = os.path.join(config.DATA_DIR, "lastfm-dataset-360K")
    tsv_file = os.path.join(extracted_folder, "usersha1-artmbid-artname-plays.tsv")

    if os.path.exists(tsv_file):
        import shutil
        shutil.move(tsv_file, config.RAW_DATA_FILE)
        print(f"✓ Data file moved to: {config.RAW_DATA_FILE}")

        # Clean up
        shutil.rmtree(extracted_folder, ignore_errors=True)
        os.remove(archive_path)
        print("✓ Cleanup completed")
    else:
        print(f"⚠ Expected file not found in archive")
        print(f"Please manually move the TSV file to: {config.RAW_DATA_FILE}")

    print("\n" + "="*70)
    print("DOWNLOAD COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nDataset location: {config.RAW_DATA_FILE}")
    print("\nNext step:")
    print("  Train models: python scripts/train_models.py")


if __name__ == '__main__':
    main()
