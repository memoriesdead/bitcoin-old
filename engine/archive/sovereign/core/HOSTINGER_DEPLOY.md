# HOSTINGER VPN PARALLEL DOWNLOAD - COMPLETE GUIDE

## GOAL: Download 264K blocks TODAY using 4 parallel VPN connections

Each VPN = different IP = bypasses rate limits = ~1 hour total instead of 8 hours.

---

## STEP 1: SSH INTO YOUR HOSTINGER VPS

```bash
ssh root@YOUR_HOSTINGER_IP
```

---

## STEP 2: INSTALL DEPENDENCIES

```bash
# Update system
apt update && apt upgrade -y

# Install Python and tools
apt install -y python3 python3-pip tmux openvpn curl wget unzip

# Install Python requests
pip3 install requests
```

---

## STEP 3: SET UP FREE VPN CONNECTIONS

Option A: Use VPNGate (Free, Multiple Countries)
```bash
# Download VPNGate config list
mkdir -p /root/vpn
cd /root/vpn
curl -o vpngate.csv "http://www.vpngate.net/api/iphone/"

# Extract configs (you'll get 10+ free VPN configs)
# Or manually download .ovpn files from https://www.vpngate.net/en/
```

Option B: Use ProtonVPN Free (Recommended - More Stable)
```bash
# Install ProtonVPN
wget https://repo.protonvpn.com/debian/dists/stable/main/binary-all/protonvpn-stable-release_1.0.3-3_all.deb
dpkg -i protonvpn-stable-release_1.0.3-3_all.deb
apt update
apt install -y protonvpn-cli

# Login (free account works)
protonvpn-cli login YOUR_EMAIL
```

Option C: Use Multiple Free VPN Services
```bash
# Download configs from multiple providers:
# - https://www.vpnbook.com/ (free, no registration)
# - https://freevpn.me/accounts/ (free accounts)
# - https://www.vpngate.net/en/ (academic project, many IPs)
```

---

## STEP 4: UPLOAD THE DOWNLOAD SCRIPT

From your Windows machine:
```powershell
scp C:\Users\kevin\livetrading\engine\sovereign\core\hostinger_download.py root@YOUR_HOSTINGER_IP:/root/
```

Or create it directly on the server (paste the script content).

---

## STEP 5: RUN 4 PARALLEL DOWNLOADS WITH TMUX

```bash
# Create 4 tmux sessions, each with a different VPN

# Session 1 - VPN1 (or no VPN for first chunk)
tmux new-session -d -s chunk1
tmux send-keys -t chunk1 'cd /root && python3 hostinger_download.py 664000 730000 chunk1' Enter

# Session 2 - Connect to VPN, then run
tmux new-session -d -s chunk2
tmux send-keys -t chunk2 'protonvpn-cli c --fastest && cd /root && python3 hostinger_download.py 730000 796000 chunk2' Enter

# Session 3 - Different VPN server
tmux new-session -d -s chunk3
tmux send-keys -t chunk3 'protonvpn-cli c NL && cd /root && python3 hostinger_download.py 796000 862000 chunk3' Enter

# Session 4 - Another VPN server
tmux new-session -d -s chunk4
tmux send-keys -t chunk4 'protonvpn-cli c JP && cd /root && python3 hostinger_download.py 862000 928000 chunk4' Enter
```

**ALTERNATIVE: Use Multiple VPS SSH Sessions**
If VPNs are tricky, just use your Hostinger server's IP + any other servers you have:
- Hostinger VPS (your IP)
- Any other cloud VM (AWS free tier, Google Cloud free tier, Oracle Cloud free tier)
- Home PC
- Friend's computer

Each machine runs a different chunk = same result.

---

## STEP 6: MONITOR PROGRESS

```bash
# Attach to any session to see progress
tmux attach -t chunk1

# Detach with: Ctrl+B then D

# Check all sessions
tmux ls

# Check downloaded files
ls -la /root/blockchain_data/
```

---

## STEP 7: COMBINE CHUNKS (After All Complete)

```bash
cd /root
python3 hostinger_download.py combine
```

This creates `/root/blockchain_data/bitcoin_2021_2025.db`

---

## STEP 8: TRANSFER BACK TO WINDOWS

From your Windows machine:
```powershell
scp root@YOUR_HOSTINGER_IP:/root/blockchain_data/bitcoin_2021_2025.db C:\Users\kevin\livetrading\data\
```

---

## ESTIMATED TIME

| Approach | Time |
|----------|------|
| Single IP, slow (current) | 8+ hours |
| 2 VPNs parallel | ~2 hours |
| 4 VPNs parallel | ~1 hour |
| 4 VPNs + aggressive | ~30 min |

---

## QUICK ONE-LINER (If You Have 4 SSH Sessions)

```bash
# Terminal 1 (your main IP)
python3 hostinger_download.py 664000 730000 chunk1

# Terminal 2 (VPN to Netherlands)
openvpn --config nl.ovpn --daemon && sleep 5 && python3 hostinger_download.py 730000 796000 chunk2

# Terminal 3 (VPN to Japan)
openvpn --config jp.ovpn --daemon && sleep 5 && python3 hostinger_download.py 796000 862000 chunk3

# Terminal 4 (VPN to US)
openvpn --config us.ovpn --daemon && sleep 5 && python3 hostinger_download.py 862000 928000 chunk4
```

---

## TROUBLESHOOTING

**Rate Limited?**
- Switch to a different VPN server
- Add longer delays in the script (change 0.15 to 0.3)

**VPN Won't Connect?**
- Try different VPN providers
- Use multiple cloud VMs instead of VPNs

**Connection Drops?**
- Script has resume capability
- Just restart it with same arguments

---

## FASTEST OPTION: MULTIPLE CLOUD VMS

If VPNs are unreliable, spin up free VMs:

1. **Oracle Cloud** - 2 free forever VMs (always free tier)
2. **Google Cloud** - $300 free credits
3. **AWS** - Free tier for 1 year
4. **Azure** - $200 free credits

Run one chunk per VM = guaranteed different IPs = no rate limits.
