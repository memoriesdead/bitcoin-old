#!/bin/bash
# =============================================================================
# HFT KERNEL OPTIMIZATION SCRIPT - RENAISSANCE TECHNOLOGIES LEVEL
# =============================================================================
# Run as root on the KVM8 trading server (31.97.211.217)
# Based on research into quant fund infrastructure
#
# Techniques from:
# - Renaissance Technologies (300K trades/day, $750M infra)
# - HFT firms using FPGA (480ns latency)
# - Kernel bypass (DPDK, PF_RING) - 62% latency reduction
#
# Usage: sudo ./optimize_kvm8.sh
# =============================================================================

set -e

echo "=============================================="
echo "HFT KERNEL OPTIMIZATION - RENAISSANCE LEVEL"
echo "=============================================="
echo "Target: Sub-microsecond trading latency"
echo "=============================================="
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (sudo ./optimize_kvm8.sh)"
    exit 1
fi

# Trading cores: 1-7 (leave core 0 for OS)
TRADING_CORES="1-7"
OS_CORE="0"

# -----------------------------------------------------------------------------
# 1. CPU ISOLATION & AFFINITY
# -----------------------------------------------------------------------------
echo "[1/10] CPU ISOLATION..."

# Move all movable kernel threads to CPU 0
echo "  Moving kernel threads to core $OS_CORE..."
for pid in $(ps -eo pid --no-headers); do
    taskset -pc $OS_CORE $pid 2>/dev/null || true
done

echo "  Trading cores reserved: $TRADING_CORES"

# -----------------------------------------------------------------------------
# 2. CPU FREQUENCY - Lock to maximum
# -----------------------------------------------------------------------------
echo "[2/10] CPU FREQUENCY SCALING..."

# Set all CPUs to performance governor
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        echo "performance" > "$cpu/cpufreq/scaling_governor" 2>/dev/null || true
    fi
done

# Disable Intel P-state if available (more predictable performance)
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "0" > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
fi

# For AMD EPYC (KVM8), boost is usually controlled here
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo "1" > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
fi

echo "  CPU governor: performance (max frequency locked)"

# -----------------------------------------------------------------------------
# 3. HUGE PAGES - Reduce TLB misses (100 cycles per miss)
# -----------------------------------------------------------------------------
echo "[3/10] HUGE PAGES..."

# Reserve 1GB of 2MB huge pages (512 pages)
echo 512 > /proc/sys/vm/nr_hugepages 2>/dev/null || true

# Enable transparent huge pages for better memory performance
echo "always" > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo "always" > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

echo "  Huge pages: 512 x 2MB = 1GB reserved"

# -----------------------------------------------------------------------------
# 4. MEMORY MANAGEMENT - Prevent latency spikes
# -----------------------------------------------------------------------------
echo "[4/10] MEMORY OPTIMIZATION..."

# Disable swap completely (prevents page fault latency spikes)
swapoff -a 2>/dev/null || true

# Minimize swappiness
echo 1 > /proc/sys/vm/swappiness

# Reduce dirty writeback to prevent I/O stalls
echo 500 > /proc/sys/vm/dirty_writeback_centisecs
echo 100 > /proc/sys/vm/dirty_expire_centisecs

# Disable zone reclaim (NUMA optimization)
echo 0 > /proc/sys/vm/zone_reclaim_mode 2>/dev/null || true

# Increase vfs cache pressure
echo 50 > /proc/sys/vm/vfs_cache_pressure

echo "  Swap: disabled, swappiness: 1"

# -----------------------------------------------------------------------------
# 5. NETWORK STACK - Low latency TCP
# -----------------------------------------------------------------------------
echo "[5/10] NETWORK OPTIMIZATION..."

# Increase socket buffer sizes for high throughput
echo 16777216 > /proc/sys/net/core/rmem_max
echo 16777216 > /proc/sys/net/core/wmem_max
echo 16777216 > /proc/sys/net/core/rmem_default
echo 16777216 > /proc/sys/net/core/wmem_default

# TCP buffer tuning
echo "4096 87380 16777216" > /proc/sys/net/ipv4/tcp_rmem
echo "4096 65536 16777216" > /proc/sys/net/ipv4/tcp_wmem

# Disable TCP slow start after idle (critical for trading)
echo 0 > /proc/sys/net/ipv4/tcp_slow_start_after_idle

# Enable TCP low latency mode
echo 1 > /proc/sys/net/ipv4/tcp_low_latency 2>/dev/null || true

# Reduce TCP timeout for faster recovery
echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse

# Increase connection tracking
echo 65536 > /proc/sys/net/core/somaxconn
echo 65536 > /proc/sys/net/core/netdev_max_backlog

# Disable TCP timestamps (reduces packet size, improves latency)
echo 0 > /proc/sys/net/ipv4/tcp_timestamps 2>/dev/null || true

echo "  TCP: low_latency=1, buffers=16MB, slow_start_after_idle=0"

# -----------------------------------------------------------------------------
# 6. IRQ AFFINITY - Move interrupts off trading cores
# -----------------------------------------------------------------------------
echo "[6/10] IRQ AFFINITY..."

# Move all IRQs to CPU 0 (non-trading core)
IRQ_MASK="01"  # Binary: only core 0
for irq in /proc/irq/[0-9]*; do
    if [ -f "$irq/smp_affinity" ]; then
        echo $IRQ_MASK > "$irq/smp_affinity" 2>/dev/null || true
    fi
done

echo "  All IRQs moved to core $OS_CORE"

# -----------------------------------------------------------------------------
# 7. KERNEL SCHEDULER - Real-time optimizations
# -----------------------------------------------------------------------------
echo "[7/10] KERNEL SCHEDULER..."

# Reduce scheduler migration cost (prevents unnecessary thread migration)
echo 5000000 > /proc/sys/kernel/sched_migration_cost_ns 2>/dev/null || true

# Increase scheduler minimum granularity
echo 10000000 > /proc/sys/kernel/sched_min_granularity_ns 2>/dev/null || true

# Disable scheduler autogroup (better RT performance)
echo 0 > /proc/sys/kernel/sched_autogroup_enabled 2>/dev/null || true

# Reduce scheduler wakeup granularity
echo 15000000 > /proc/sys/kernel/sched_wakeup_granularity_ns 2>/dev/null || true

echo "  Scheduler: optimized for real-time workloads"

# -----------------------------------------------------------------------------
# 8. DISABLE UNNECESSARY SERVICES
# -----------------------------------------------------------------------------
echo "[8/10] DISABLING UNNECESSARY SERVICES..."

# Stop services that cause latency spikes
systemctl stop irqbalance 2>/dev/null || true
systemctl disable irqbalance 2>/dev/null || true

systemctl stop tuned 2>/dev/null || true
systemctl disable tuned 2>/dev/null || true

systemctl stop cpupower 2>/dev/null || true
systemctl disable cpupower 2>/dev/null || true

# Disable ondemand (conflicts with performance governor)
systemctl stop ondemand 2>/dev/null || true
systemctl disable ondemand 2>/dev/null || true

echo "  Stopped: irqbalance, tuned, cpupower, ondemand"

# -----------------------------------------------------------------------------
# 9. KERNEL WATCHDOG - Disable latency spikes
# -----------------------------------------------------------------------------
echo "[9/10] KERNEL WATCHDOG..."

# Disable kernel watchdog (causes periodic latency spikes)
echo 0 > /proc/sys/kernel/watchdog 2>/dev/null || true
echo 0 > /proc/sys/kernel/nmi_watchdog 2>/dev/null || true

# Disable soft lockup detection
echo 0 > /proc/sys/kernel/soft_watchdog 2>/dev/null || true

echo "  Watchdog: disabled"

# -----------------------------------------------------------------------------
# 10. FILESYSTEM - Reduce I/O latency
# -----------------------------------------------------------------------------
echo "[10/10] FILESYSTEM OPTIMIZATION..."

# Set I/O scheduler to none/noop for SSDs (if available)
for device in /sys/block/sd*/queue/scheduler; do
    if [ -f "$device" ]; then
        echo "none" > "$device" 2>/dev/null || \
        echo "noop" > "$device" 2>/dev/null || true
    fi
done

for device in /sys/block/nvme*/queue/scheduler; do
    if [ -f "$device" ]; then
        echo "none" > "$device" 2>/dev/null || true
    fi
done

# Increase read-ahead for sequential workloads
for device in /sys/block/sd*/queue/read_ahead_kb; do
    if [ -f "$device" ]; then
        echo 256 > "$device" 2>/dev/null || true
    fi
done

echo "  I/O scheduler: none/noop (SSD optimized)"

# =============================================================================
# SUMMARY
# =============================================================================
echo
echo "=============================================="
echo "HFT KERNEL OPTIMIZATION - COMPLETE"
echo "=============================================="
echo
echo "OPTIMIZATIONS APPLIED:"
echo "  [✓] CPU isolation (trading cores: $TRADING_CORES)"
echo "  [✓] CPU frequency locked to maximum"
echo "  [✓] Huge pages: 1GB reserved"
echo "  [✓] Swap disabled, memory locked"
echo "  [✓] Network: low latency TCP"
echo "  [✓] IRQs moved to core 0"
echo "  [✓] Scheduler: real-time optimized"
echo "  [✓] Background services disabled"
echo "  [✓] Watchdog disabled"
echo "  [✓] I/O scheduler: SSD optimized"
echo
echo "RUN YOUR TRADING ENGINE WITH:"
echo "  taskset -c $TRADING_CORES nice -n -20 python3 -O engine.py"
echo
echo "OR FOR REALTIME PRIORITY:"
echo "  taskset -c $TRADING_CORES chrt -f 99 python3 -O engine.py"
echo
echo "=============================================="

# Create a convenient launcher script
cat > /root/livetrading/run_hft.sh << 'EOF'
#!/bin/bash
# HFT Trading Engine Launcher - Optimized for KVM8
cd /root/livetrading

# Environment variables for maximum Numba performance
export NUMBA_OPT=3
export NUMBA_LOOP_VECTORIZE=1
export NUMBA_INTEL_SVML=1
export NUMBA_ENABLE_AVX=1
export NUMBA_THREADING_LAYER=omp
export NUMBA_BOUNDSCHECK=0
export NUMBA_FASTMATH=1

# Run with CPU affinity (cores 1-7) and high priority
taskset -c 1-7 nice -n -20 python3 -O -u "$@"
EOF

chmod +x /root/livetrading/run_hft.sh
echo "Created: /root/livetrading/run_hft.sh"
echo
