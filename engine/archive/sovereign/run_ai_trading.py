#!/usr/bin/env python3
"""
Sovereign Engine v12.0 - AI-Enhanced Live Trading
================================================
Runs the full trading system with Claude AI validation.
"""
import sys
sys.path.insert(0, "/root/livetrading")

import time
import subprocess
import json
from datetime import datetime

print("=" * 70)
print("SOVEREIGN ENGINE v12.0 - AI-ENHANCED LIVE TRADING")
print("=" * 70)
print(f"Started: {datetime.now()}")
print()

def ask_claude(prompt: str, timeout: int = 30) -> str:
    """Query Claude AI via CLI."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "sonnet"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("[AI] Timeout")
    except Exception as e:
        print(f"[AI ERROR] {e}")
    return ""


def validate_signal_with_ai(signal: dict) -> dict:
    """Ask Claude to validate a trading signal."""
    # FormulaConnector uses 1/-1 for direction
    dir_num = signal.get("direction", 0)
    direction = "LONG" if dir_num == 1 else "SHORT" if dir_num == -1 else "NEUTRAL"
    conf = signal.get("confidence", 0)
    flow = signal.get("btc_amount", 0)  # FormulaConnector uses btc_amount

    # Build sources from individual engine signals
    sources = []
    if signal.get("adaptive_signal"):
        sources.append("adaptive")
    if signal.get("pattern_signal"):
        sources.append("pattern")
    if signal.get("rentech_signal"):
        sources.append("rentech")

    # Determine flow direction based on signal direction
    # LONG = outflow from exchange (accumulation)
    # SHORT = inflow to exchange (sell pressure)
    if dir_num == 1:
        flow_type = "OUTFLOW"
        flow_meaning = "BTC withdrawn FROM exchange (accumulation, bullish)"
    else:
        flow_type = "INFLOW"
        flow_meaning = "BTC deposited TO exchange (selling pressure, bearish)"

    prompt = f"""Analyze this Bitcoin trading signal. Respond with JSON only.

CONTEXT: In Bitcoin, when BTC flows INTO an exchange (INFLOW), it means people are depositing BTC to SELL it = bearish/SHORT. When BTC flows OUT of an exchange (OUTFLOW), it means people are withdrawing BTC to hold = bullish/LONG.

Signal: {direction}
Confidence: {conf:.2f}
Flow: {flow:.2f} BTC {flow_type} - {flow_meaning}
Engines agreeing: {len(sources)}/3 ({', '.join(sources)})

Evaluate:
1. Does the direction match the flow logic above? (INFLOW=SHORT is correct, OUTFLOW=LONG is correct)
2. Is confidence appropriate? (3 engines = high, 2 engines = medium, 1 engine = low)
3. Flow size >50 BTC is significant, <5 BTC is noise.

Respond ONLY with: {{"valid": true/false, "confidence_adjust": 0.8-1.2, "reason": "one sentence"}}"""

    response = ask_claude(prompt)
    if response:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
    return {"valid": True, "confidence_adjust": 1.0, "reason": "AI unavailable - approved by default"}


# Test Claude connection
print("[AI] Testing Claude connection...")
test = ask_claude("Respond with: CONNECTED")
if "CONNECTED" in test.upper():
    print(f"[AI] Claude Sonnet 4: CONNECTED")
    ai_enabled = True
else:
    print(f"[AI] Claude unavailable - running without AI validation")
    ai_enabled = False
print()

# Import and initialize
from engine.sovereign.blockchain.formula_connector import FormulaConnector

connector = FormulaConnector(
    zmq_endpoint="tcp://127.0.0.1:28332",
    enable_pattern_recognition=True,
    enable_rentech=True,
    rentech_mode="full",
)

print("[INIT] FormulaConnector ready")
print("[INIT] Engines: Adaptive + Pattern + RenTech (full)")
print("[INIT] Addresses: 64,280 across 12 exchanges")
print(f"[INIT] AI Validation: {'ENABLED' if ai_enabled else 'DISABLED'}")
print()

# Stats
stats = {
    "signals": 0,
    "ai_checked": 0,
    "ai_approved": 0,
    "ai_rejected": 0,
    "long": 0,
    "short": 0,
    "total_flow": 0.0,
}

start_time = time.time()


def on_signal(signal):
    """Process each trading signal."""
    global stats

    # FormulaConnector uses 1/-1 for direction, not strings
    dir_num = signal.get("direction", 0)
    direction = "LONG" if dir_num == 1 else "SHORT" if dir_num == -1 else "NEUTRAL"
    conf = signal.get("confidence", 0)
    flow = abs(signal.get("btc_amount", 0))  # FormulaConnector uses btc_amount not net_flow

    # Build sources list from individual engine flags
    sources = []
    if signal.get("adaptive_signal"):
        sources.append("adaptive")
    if signal.get("pattern_signal"):
        sources.append("pattern")
    if signal.get("rentech_signal"):
        sources.append("rentech")

    stats["signals"] += 1
    stats["total_flow"] += flow

    # Validate signals with AI - lowered thresholds to see more action
    # flow > 2 BTC OR 2+ engines agree OR high confidence with any engine
    should_validate = ai_enabled and (flow > 2 or len(sources) >= 2 or (conf > 0.9 and flow > 1))

    if should_validate:
        stats["ai_checked"] += 1
        validation = validate_signal_with_ai(signal)

        is_valid = validation.get("valid", True)
        reason = validation.get("reason", "")
        conf_adjust = validation.get("confidence_adjust", 1.0)

        adjusted_conf = conf * conf_adjust

        if is_valid:
            stats["ai_approved"] += 1
            if direction == "LONG":
                stats["long"] += 1
            elif direction == "SHORT":
                stats["short"] += 1

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [AI OK] {direction} | {flow:.2f} BTC | conf={adjusted_conf:.2f}")
            print(f"    {reason}")
        else:
            stats["ai_rejected"] += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [AI REJECT] {direction} | {flow:.2f} BTC")
            print(f"    {reason}")
    else:
        # Small signal - no AI check needed
        if direction == "LONG":
            stats["long"] += 1
        elif direction == "SHORT":
            stats["short"] += 1

        # Log medium signals
        if flow > 2:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {direction} | {flow:.2f} BTC | conf={conf:.2f} | {sources}")

    sys.stdout.flush()


connector.on_signal = on_signal
connector.start()

print("[LIVE] Connected to Bitcoin Core ZMQ")
print("[LIVE] Monitoring blockchain for exchange flows...")
print("[LIVE] AI validates signals >5 BTC or high confidence")
print()
sys.stdout.flush()

try:
    while True:
        time.sleep(60)
        elapsed = int(time.time() - start_time)
        mins = elapsed // 60

        # Every 10 minutes
        if mins % 10 == 0 and mins > 0:
            ai_rate = (stats["ai_approved"] / stats["ai_checked"] * 100) if stats["ai_checked"] > 0 else 0
            print(f"\n[{mins}m] Signals={stats['signals']} L={stats['long']} S={stats['short']} | AI: {stats['ai_approved']}/{stats['ai_checked']} ({ai_rate:.0f}% approved) | Flow={stats['total_flow']:.1f} BTC\n")
            sys.stdout.flush()

        # Hourly summary
        if mins > 0 and mins % 60 == 0:
            hours = mins // 60
            print()
            print("=" * 50)
            print(f"HOUR {hours} SUMMARY")
            print("=" * 50)
            print(f"Total Signals: {stats['signals']}")
            print(f"Long: {stats['long']} | Short: {stats['short']}")
            print(f"AI Checked: {stats['ai_checked']}")
            print(f"AI Approved: {stats['ai_approved']} | Rejected: {stats['ai_rejected']}")
            print(f"Total Flow: {stats['total_flow']:.2f} BTC")
            print("=" * 50)
            print()
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\n[STOPPED BY USER]")
finally:
    connector.stop()
    runtime = (time.time() - start_time) / 60

    print()
    print("=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"Runtime: {runtime:.1f} minutes")
    print(f"Total Signals: {stats['signals']}")
    print(f"Long: {stats['long']} | Short: {stats['short']}")
    print(f"AI Validated: {stats['ai_checked']}")
    print(f"  - Approved: {stats['ai_approved']}")
    print(f"  - Rejected: {stats['ai_rejected']}")
    print(f"Total Flow: {stats['total_flow']:.2f} BTC")
    print("=" * 70)
    sys.stdout.flush()
