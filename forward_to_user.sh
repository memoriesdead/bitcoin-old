#!/bin/bash
# Forward funds from sweep address to user taproot address

USER_ADDR="bc1phym0yyuz3078rp80hkw4hg455l3dn5d8dudlek2sjy39altx3ynsn6yh3v"
SWEEP_WALLET="sweep"

# Check balance in sweep wallet
BALANCE=$(bitcoin-cli -rpcwallet=$SWEEP_WALLET getbalance)

if [ "$(echo "$BALANCE > 0.0001" | bc -l)" -eq 1 ]; then
    echo "Balance found: $BALANCE BTC"
    echo "Sending to user address: $USER_ADDR"

    # Send all funds minus small fee
    TXID=$(bitcoin-cli -rpcwallet=$SWEEP_WALLET sendtoaddress "$USER_ADDR" "$BALANCE" "" "" true)

    if [ $? -eq 0 ]; then
        echo "SUCCESS! Sent to user: $TXID"
    else
        echo "ERROR sending funds"
    fi
else
    echo "No balance in sweep wallet yet (Balance: $BALANCE)"
fi
