-- ============================================================================
-- BIGQUERY: COMPLETE BITCOIN BLOCKCHAIN DATA
-- ============================================================================
-- Source: Google BigQuery Public Datasets (FREE, institutional-grade)
-- Coverage: Genesis Block → Present (~890,000 blocks)
-- Usage: Run this in BigQuery console, export results to CSV
-- Time: ~30 seconds query, ~2 minutes export
-- ============================================================================

SELECT
    -- Core identifiers
    block_number as height,
    block_timestamp as timestamp,
    `hash` as block_hash,

    -- Mining data
    difficulty,
    nonce,

    -- Transaction data
    transaction_count as tx_count,
    input_count,
    output_count,
    input_value / 1e8 as input_btc,   -- Convert satoshis to BTC
    output_value / 1e8 as output_btc,

    -- Block properties
    size as block_size,
    stripped_size,
    weight,
    version,

    -- Merkle root for validation
    merkle_root,

    -- Coinbase (block reward)
    coinbase_param

FROM `bigquery-public-data.crypto_bitcoin.blocks`

-- Genesis → Current
WHERE block_number >= 0

-- Sort chronologically
ORDER BY block_number ASC;

-- ============================================================================
-- ALTERNATIVE: RECENT BLOCKS ONLY (Last 100,000 blocks for testing)
-- ============================================================================
-- Uncomment below to get just recent blocks for faster testing:
/*
SELECT
    block_number as height,
    block_timestamp as timestamp,
    `hash` as block_hash,
    difficulty,
    transaction_count as tx_count,
    size as block_size,
    output_value / 1e8 as output_btc
FROM `bigquery-public-data.crypto_bitcoin.blocks`
WHERE block_number >= (SELECT MAX(block_number) - 100000 FROM `bigquery-public-data.crypto_bitcoin.blocks`)
ORDER BY block_number ASC;
*/

-- ============================================================================
-- AFTER EXPORT:
-- ============================================================================
-- 1. Save as: blockchain_complete.csv
-- 2. Place in: livetrading/data/
-- 3. Run: python scripts/convert_csv_to_numpy.py
-- 4. Result: blockchain_complete.npy (ultra-fast binary format)
-- ============================================================================
