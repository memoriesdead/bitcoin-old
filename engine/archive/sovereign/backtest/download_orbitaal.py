#!/usr/bin/env python3
"""
ORBITAAL Dataset Downloader

Downloads the complete Bitcoin transaction dataset (2009-2021) from Zenodo.
156GB compressed, contains ALL Bitcoin transactions.

Usage:
    python -m engine.sovereign.backtest.download_orbitaal
"""
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
import json
import time

# Zenodo record ID
ZENODO_RECORD = "12581515"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Files we need (prioritized for trading analysis)
PRIORITY_FILES = [
    "orbitaal-stream_graph.tar.gz",      # All transactions by year
    "orbitaal-nodetable.tar.gz",          # Entity/wallet mapping
]

# Optional files
OPTIONAL_FILES = [
    "orbitaal-snapshot-all.tar.gz",       # Full aggregated graph
    "orbitaal-snapshot-day.tar.gz",       # Daily snapshots
]


def get_file_list():
    """Get list of files from Zenodo API."""
    try:
        with urlopen(ZENODO_API, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get('files', [])
    except Exception as e:
        print(f"[!] Error fetching file list: {e}")
        return []


def download_with_progress(url: str, dest: Path, total_size: int = 0):
    """Download file with progress indicator."""

    def progress_hook(count, block_size, total):
        if total > 0:
            percent = min(100, count * block_size * 100 / total)
            downloaded = count * block_size / (1024**3)  # GB
            total_gb = total / (1024**3)
            sys.stdout.write(f"\r  [{percent:5.1f}%] {downloaded:.2f} / {total_gb:.2f} GB")
            sys.stdout.flush()

    try:
        urlretrieve(url, dest, reporthook=progress_hook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n[!] Download error: {e}")
        return False


def main():
    output_dir = Path("data/orbitaal")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ORBITAAL DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Source: Zenodo Record {ZENODO_RECORD}")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 60)

    # Get file list from Zenodo
    print("\n[1] Fetching file list from Zenodo...")
    files = get_file_list()

    if not files:
        print("[!] Could not fetch file list. Manual download:")
        print(f"    https://zenodo.org/records/{ZENODO_RECORD}")
        return

    # Show available files
    print(f"\n[+] Found {len(files)} files:")
    total_size = 0
    for f in files:
        size_gb = f['size'] / (1024**3)
        total_size += f['size']
        priority = "**" if f['key'] in PRIORITY_FILES else "  "
        print(f"  {priority} {f['key']}: {size_gb:.1f} GB")

    print(f"\n    Total: {total_size / (1024**3):.1f} GB")
    print(f"    ** = Priority files for trading analysis")

    # Download priority files first
    print("\n[2] Downloading priority files...")

    for filename in PRIORITY_FILES:
        file_info = next((f for f in files if f['key'] == filename), None)
        if not file_info:
            print(f"  [!] {filename} not found")
            continue

        dest = output_dir / filename
        if dest.exists():
            print(f"  [✓] {filename} already exists, skipping")
            continue

        print(f"\n  Downloading {filename}...")
        print(f"  Size: {file_info['size'] / (1024**3):.1f} GB")
        print(f"  URL: {file_info['links']['self']}")

        success = download_with_progress(
            file_info['links']['self'],
            dest,
            file_info['size']
        )

        if success:
            print(f"  [✓] Downloaded {filename}")
        else:
            print(f"  [!] Failed to download {filename}")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Extract: tar -xzf orbitaal-stream_graph.tar.gz")
    print("  2. Run: python -m engine.sovereign.backtest.load_orbitaal")


if __name__ == '__main__':
    main()
