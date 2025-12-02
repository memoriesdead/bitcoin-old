#!/usr/bin/env python3
"""
AUTOMATED BIGQUERY BLOCKCHAIN DATA DOWNLOAD
============================================
Downloads complete Bitcoin blockchain from Google BigQuery public dataset.
NO authentication needed - public dataset access.
"""

import os
import sys
from google.cloud import bigquery

print("=" * 80)
print("BIGQUERY BLOCKCHAIN DATA DOWNLOAD - AUTOMATED")
print("=" * 80)

# Create BigQuery client (anonymous access to public data)
try:
    client = bigquery.Client()
    print("[OK] BigQuery client created (anonymous public access)")
except Exception as e:
    print(f"[INFO] Creating client without credentials for public data access")
    # For public datasets, we can use anonymous access
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
    client = bigquery.Client(project='bigquery-public-data')

# SQL query for complete blockchain data
query = """
SELECT
    block_number as height,
    UNIX_SECONDS(block_timestamp) as timestamp,
    difficulty,
    transaction_count as tx_count,
    size as block_size,
    nonce
FROM `bigquery-public-data.crypto_bitcoin.blocks`
WHERE block_number >= 0
ORDER BY block_number ASC
"""

print("\n[QUERY] Fetching 926,109+ blocks from BigQuery...")
print("[QUERY] This may take 1-2 minutes...")
print()

try:
    # Execute query
    query_job = client.query(query)

    # Get results as dataframe
    print("[DOWNLOAD] Receiving data...")
    df = query_job.to_dataframe()

    print(f"[OK] Downloaded {len(df):,} blocks")
    print()

    # Save to CSV
    output_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'blockchain_complete.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"[SAVE] Writing to: {output_csv}")
    df.to_csv(output_csv, index=False)

    size_mb = os.path.getsize(output_csv) / 1024 / 1024
    print(f"[OK] Saved {len(df):,} blocks ({size_mb:.1f} MB)")
    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE - READY FOR NUMPY CONVERSION")
    print("=" * 80)
    print()
    print("Next step:")
    print("  python3 scripts/convert_csv_to_numpy.py")
    print()

except Exception as e:
    print(f"\n[ERROR] BigQuery download failed: {e}")
    print()
    print("Alternative: Manual download from BigQuery console")
    print("  1. Go to: https://console.cloud.google.com/bigquery")
    print("  2. Run query from: scripts/bigquery_blockchain_data.sql")
    print("  3. Export results to CSV")
    sys.exit(1)
