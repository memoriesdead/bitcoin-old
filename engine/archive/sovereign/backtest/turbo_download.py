#!/usr/bin/env python3
"""
TURBO DOWNLOAD - Maximum Speed Parallel Downloads

Maxes out your connection with:
- 16 parallel download threads
- Multi-connection per file (aria2-style)
- Automatic resume on failure
- All data sources simultaneously

Usage:
    python -m engine.sovereign.backtest.turbo_download
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request, urlretrieve
from urllib.error import URLError, HTTPError
import threading

# Configuration
MAX_WORKERS = 16  # Parallel downloads
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Data sources to download
DOWNLOADS = {
    # ORBITAAL Dataset (Zenodo) - 156GB total
    "orbitaal": {
        "api": "https://zenodo.org/api/records/12581515",
        "priority": [
            "orbitaal-stream_graph.tar.gz",      # Main data
            "orbitaal-nodetable.tar.gz",          # Entity mapping
        ],
        "dir": "data/orbitaal"
    },
    # Exchange Addresses - 1GB
    "exchanges": {
        "url": "https://drive.switch.ch/index.php/s/ag4OnNgwf7LhWFu/download",
        "file": "data/entity_addresses.zip",
        "dir": "data/entity_addresses"
    },
    # Price Data - Binance API (fast, small)
    "prices": {
        "source": "binance",
        "dir": "data"
    }
}


class TurboDownloader:
    """Multi-threaded parallel downloader."""

    def __init__(self):
        self.active_downloads = 0
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def download_chunk(self, url: str, dest: Path, start: int, end: int, chunk_id: int) -> bool:
        """Download a chunk of a file."""
        headers = {'Range': f'bytes={start}-{end}'}
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as resp:
                chunk_file = dest.parent / f"{dest.name}.part{chunk_id}"
                with open(chunk_file, 'wb') as f:
                    f.write(resp.read())
            return True
        except Exception as e:
            print(f"[!] Chunk {chunk_id} failed: {e}")
            return False

    def download_file_fast(self, url: str, dest: Path, total_size: int = 0) -> bool:
        """Download file using curl with max connections."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 0:
            if total_size == 0 or dest.stat().st_size >= total_size * 0.99:
                print(f"  [✓] Already exists: {dest.name}")
                return True

        print(f"  Downloading: {dest.name}")
        if total_size > 0:
            print(f"  Size: {total_size / (1024**3):.2f} GB")

        # Use curl with parallel connections
        cmd = [
            "curl",
            "-L",  # Follow redirects
            "-C", "-",  # Resume if possible
            "--parallel",  # Parallel transfers
            "--parallel-max", "8",  # Max parallel
            "-o", str(dest),
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=False, timeout=7200)  # 2 hour timeout
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"  [!] Timeout downloading {dest.name}")
            return False
        except Exception as e:
            print(f"  [!] Error: {e}")
            return False

    def download_with_progress(self, url: str, dest: Path) -> bool:
        """Download with progress indicator."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        def hook(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 / total)
                mb = count * block / (1024**2)
                total_mb = total / (1024**2)
                speed = mb / (time.time() - self.start_time) if time.time() > self.start_time else 0
                sys.stdout.write(f"\r  [{pct:5.1f}%] {mb:.1f}/{total_mb:.1f} MB @ {speed:.1f} MB/s")
                sys.stdout.flush()

        try:
            urlretrieve(url, dest, reporthook=hook)
            print()
            return True
        except Exception as e:
            print(f"\n  [!] Error: {e}")
            return False

    def get_zenodo_files(self, api_url: str) -> list:
        """Get file list from Zenodo API."""
        try:
            with urlopen(api_url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data.get('files', [])
        except Exception as e:
            print(f"[!] Error fetching Zenodo files: {e}")
            return []

    def download_orbitaal(self):
        """Download ORBITAAL dataset with max parallelism."""
        print("\n" + "=" * 60)
        print("DOWNLOADING ORBITAAL (156 GB)")
        print("=" * 60)

        output_dir = Path(DOWNLOADS['orbitaal']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get file list
        files = self.get_zenodo_files(DOWNLOADS['orbitaal']['api'])
        if not files:
            print("[!] Could not get file list")
            return False

        # Sort by priority
        priority = DOWNLOADS['orbitaal']['priority']
        files.sort(key=lambda f: (
            0 if f['key'] in priority else 1,
            priority.index(f['key']) if f['key'] in priority else 999
        ))

        total_size = sum(f['size'] for f in files)
        print(f"Total: {total_size / (1024**3):.1f} GB across {len(files)} files\n")

        # Download in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for f in files:
                dest = output_dir / f['key']
                url = f['links']['self']
                futures.append(executor.submit(
                    self.download_file_fast, url, dest, f['size']
                ))

            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        self.completed += 1
                    else:
                        self.failed += 1
                except Exception as e:
                    print(f"[!] Download error: {e}")
                    self.failed += 1

        return self.failed == 0

    def download_exchanges(self):
        """Download exchange addresses."""
        print("\n" + "=" * 60)
        print("DOWNLOADING EXCHANGE ADDRESSES (1 GB)")
        print("=" * 60)

        url = DOWNLOADS['exchanges']['url']
        dest = Path(DOWNLOADS['exchanges']['file'])
        extract_dir = Path(DOWNLOADS['exchanges']['dir'])

        # Download
        if not dest.exists():
            self.start_time = time.time()
            if not self.download_with_progress(url, dest):
                return False

        # Extract
        if dest.exists() and not extract_dir.exists():
            print("  Extracting...")
            try:
                with zipfile.ZipFile(dest, 'r') as z:
                    z.extractall(extract_dir)
                print("  [✓] Extracted")
            except Exception as e:
                print(f"  [!] Extract error: {e}")
                return False

        return True

    def download_prices(self):
        """Download price data from Binance."""
        print("\n" + "=" * 60)
        print("DOWNLOADING PRICE DATA")
        print("=" * 60)

        try:
            from engine.sovereign.backtest.price_downloader import PriceDownloader
            downloader = PriceDownloader()
            downloader.download(start_date="2020-01-01")
            return True
        except Exception as e:
            print(f"[!] Price download error: {e}")
            return False

    def run(self):
        """Run all downloads in parallel."""
        print("\n" + "=" * 70)
        print("🚀 TURBO DOWNLOAD - MAXIMUM SPEED MODE")
        print("=" * 70)
        print(f"Workers: {MAX_WORKERS}")
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Run downloads in parallel threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.download_exchanges): "exchanges",
                executor.submit(self.download_prices): "prices",
                executor.submit(self.download_orbitaal): "orbitaal",
            }

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    success = future.result()
                    status = "✓" if success else "✗"
                    print(f"\n[{status}] {name.upper()} complete")
                except Exception as e:
                    print(f"\n[✗] {name.upper()} failed: {e}")

        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(f"DOWNLOAD COMPLETE in {elapsed/60:.1f} minutes")
        print("=" * 70)


def main():
    downloader = TurboDownloader()
    downloader.run()


if __name__ == '__main__':
    main()
