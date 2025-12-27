/**
 * SCT Wilson CI Calculator - Implementation
 *
 * Nanosecond-speed statistical certainty calculations.
 * Only trades when 99% confident that win rate >= 50.75%.
 */

#include "sct_wilson.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace sct {

// ============================================================================
// WilsonCalculator Implementation
// ============================================================================

double WilsonCalculator::get_z_score(double confidence) {
    // Fast lookup for common values
    if (confidence >= 0.999) return Z_999;
    if (confidence >= 0.99) return Z_99;
    if (confidence >= 0.95) return Z_95;
    if (confidence >= 0.90) return Z_90;

    // Approximate for other values using inverse error function approximation
    // This is a simplified approximation good enough for trading
    double p = (1.0 + confidence) / 2.0;
    double t = std::sqrt(-2.0 * std::log(1.0 - p));

    // Abramowitz and Stegun approximation
    constexpr double c0 = 2.515517;
    constexpr double c1 = 0.802853;
    constexpr double c2 = 0.010328;
    constexpr double d1 = 1.432788;
    constexpr double d2 = 0.189269;
    constexpr double d3 = 0.001308;

    return t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t);
}

WilsonInterval WilsonCalculator::wilson_interval(int wins, int total, double confidence) {
    WilsonInterval result{0.0, 0.0, 0.0};

    if (total <= 0) return result;

    double p = static_cast<double>(wins) / total;
    double z = get_z_score(confidence);
    double z2 = z * z;
    double n = static_cast<double>(total);

    double denominator = 1.0 + z2 / n;
    double center = (p + z2 / (2.0 * n)) / denominator;
    double spread = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator;

    result.lower = std::max(0.0, center - spread);
    result.upper = std::min(1.0, center + spread);
    result.center = center;

    return result;
}

double WilsonCalculator::wilson_lower_bound(int wins, int total, double confidence) {
    if (total <= 0) return 0.0;

    double p = static_cast<double>(wins) / total;
    double z = get_z_score(confidence);
    double z2 = z * z;
    double n = static_cast<double>(total);

    double denominator = 1.0 + z2 / n;
    double center = (p + z2 / (2.0 * n)) / denominator;
    double spread = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator;

    return std::max(0.0, center - spread);
}

// ============================================================================
// CertaintyChecker Implementation
// ============================================================================

CertaintyChecker::CertaintyChecker(double min_wr, double confidence)
    : min_wr_(min_wr)
    , confidence_(confidence)
    , z_score_(WilsonCalculator::get_z_score(confidence))
{
}

bool CertaintyChecker::is_certain(int wins, int total) const {
    double lower = WilsonCalculator::wilson_lower_bound(wins, total, confidence_);
    return lower >= min_wr_;
}

int CertaintyChecker::trades_needed(double observed_wr) const {
    // If observed win rate is below threshold, impossible
    if (observed_wr <= min_wr_) {
        return -1;
    }

    // Binary search for minimum sample size
    int low = 10;
    int high = 10000;
    int result = -1;

    while (low <= high) {
        int mid = (low + high) / 2;
        int wins = static_cast<int>(mid * observed_wr);

        if (is_certain(wins, mid)) {
            result = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    return result;
}

CertaintyResult CertaintyChecker::check(int wins, int total) const {
    auto start = std::chrono::high_resolution_clock::now();

    CertaintyResult result;
    result.target_wr = min_wr_;
    result.confidence = confidence_;

    if (total == 0) {
        result.observed_wr = 0.0;
        result.lower_bound = 0.0;
        result.upper_bound = 0.0;
        result.trades_needed = -1;
        result.status = CertaintyStatus::NEED_MORE_DATA;

        auto end = std::chrono::high_resolution_clock::now();
        result.calc_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        return result;
    }

    result.observed_wr = static_cast<double>(wins) / total;

    auto interval = WilsonCalculator::wilson_interval(wins, total, confidence_);
    result.lower_bound = interval.lower;
    result.upper_bound = interval.upper;

    // Determine status
    if (result.lower_bound >= min_wr_) {
        result.status = CertaintyStatus::CERTAIN;
        result.trades_needed = 0;
    } else if (result.observed_wr > min_wr_) {
        result.status = CertaintyStatus::NEED_MORE_DATA;
        result.trades_needed = trades_needed(result.observed_wr);
    } else {
        result.status = CertaintyStatus::NO_EDGE;
        result.trades_needed = -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.calc_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();

    return result;
}

// ============================================================================
// KellyPositionSizer Implementation
// ============================================================================

KellyPositionSizer::KellyPositionSizer(double kelly_fraction, double min_wr)
    : kelly_mult_(kelly_fraction)
    , min_wr_(min_wr)
    , checker_(min_wr)
{
}

double KellyPositionSizer::kelly_fraction(double win_rate, double risk_reward) {
    // f* = (b*p - q) / b
    // Where: p = win probability, q = 1-p, b = risk/reward

    if (win_rate <= 0.5) return 0.0;

    double p = win_rate;
    double q = 1.0 - p;
    double b = risk_reward;

    return (b * p - q) / b;
}

PositionSize KellyPositionSizer::size_from_win_rate(double win_rate, double risk_reward) const {
    PositionSize result;
    result.win_rate_used = win_rate;

    if (win_rate < min_wr_) {
        result.full_kelly = 0.0;
        result.quarter_kelly = 0.0;
        result.recommended = 0.0;
        result.capital_pct = 0.0;
        return result;
    }

    result.full_kelly = kelly_fraction(win_rate, risk_reward);
    result.quarter_kelly = result.full_kelly * kelly_mult_;

    // Apply bounds
    result.recommended = std::max(MIN_POSITION_PCT,
                                  std::min(result.quarter_kelly, MAX_POSITION_PCT));
    result.capital_pct = result.recommended * 100.0;

    return result;
}

PositionSize KellyPositionSizer::size_from_stats(int wins, int total, double risk_reward) const {
    PositionSize result;

    if (total == 0) {
        result.win_rate_used = 0.0;
        result.full_kelly = 0.0;
        result.quarter_kelly = 0.0;
        result.recommended = 0.0;
        result.capital_pct = 0.0;
        return result;
    }

    // Use lower bound for safety
    double lower = WilsonCalculator::wilson_lower_bound(wins, total, DEFAULT_CONFIDENCE);

    return size_from_win_rate(lower, risk_reward);
}

// ============================================================================
// StrategyTracker Implementation
// ============================================================================

StrategyTracker::StrategyTracker(double min_wr)
    : checker_(min_wr)
{
}

void StrategyTracker::record_trade(const std::string& strategy, bool won) {
    auto now = std::chrono::high_resolution_clock::now();
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();

    auto it = strategies_.find(strategy);
    if (it == strategies_.end()) {
        strategies_[strategy] = StrategyStats{strategy, 0, 0, now_ns, now_ns};
        it = strategies_.find(strategy);
    }

    if (won) {
        it->second.wins++;
    } else {
        it->second.losses++;
    }
    it->second.updated_at = now_ns;
}

const StrategyTracker::StrategyStats* StrategyTracker::get_stats(const std::string& strategy) const {
    auto it = strategies_.find(strategy);
    if (it == strategies_.end()) return nullptr;
    return &it->second;
}

CertaintyResult StrategyTracker::check_strategy(const std::string& strategy) const {
    auto stats = get_stats(strategy);
    if (!stats) {
        return CertaintyResult{0, 0, 0, MIN_WIN_RATE, DEFAULT_CONFIDENCE, -1,
                              CertaintyStatus::NEED_MORE_DATA, 0};
    }
    return checker_.check(stats->wins, stats->total());
}

std::vector<std::string> StrategyTracker::get_tradeable() const {
    std::vector<std::string> result;
    for (const auto& [name, stats] : strategies_) {
        if (checker_.is_certain(stats.wins, stats.total())) {
            result.push_back(name);
        }
    }
    return result;
}

std::vector<std::string> StrategyTracker::get_pending() const {
    std::vector<std::string> result;
    for (const auto& [name, stats] : strategies_) {
        if (!checker_.is_certain(stats.wins, stats.total()) && stats.win_rate() > MIN_WIN_RATE) {
            result.push_back(name);
        }
    }
    return result;
}

}  // namespace sct

// ============================================================================
// STANDALONE MAIN FOR TESTING
// ============================================================================
#ifdef SCT_STANDALONE

#include <iostream>

int main() {
    std::cout << "========================================\n";
    std::cout << "SCT C++ WILSON CI CALCULATOR\n";
    std::cout << "Nanosecond Speed\n";
    std::cout << "========================================\n\n";

    sct::CertaintyChecker checker;
    sct::KellyPositionSizer sizer;

    // Test cases
    struct TestCase {
        int wins;
        int total;
        const char* desc;
    };

    TestCase tests[] = {
        {700, 1000, "70% WR - Strong"},
        {580, 1000, "58% WR - Good"},
        {520, 1000, "52% WR - Marginal"},
        {510, 1000, "51% WR - Borderline"},
        {490, 1000, "49% WR - No edge"},
        {55, 100, "55% WR - Small sample"},
        {580, 1000, "58% WR - Medium sample"},
    };

    std::cout << std::fixed << std::setprecision(2);

    for (const auto& test : tests) {
        std::cout << ">>> " << test.desc << " (" << test.wins << "/" << test.total << ")\n";

        auto result = checker.check(test.wins, test.total);

        std::cout << "    Observed WR: " << (result.observed_wr * 100) << "%\n";
        std::cout << "    Wilson CI:   [" << (result.lower_bound * 100) << "%, "
                  << (result.upper_bound * 100) << "%]\n";
        std::cout << "    Status:      ";

        switch (result.status) {
            case sct::CertaintyStatus::CERTAIN:
                std::cout << "CERTAIN - Safe to trade\n";
                break;
            case sct::CertaintyStatus::NEED_MORE_DATA:
                std::cout << "NEED_MORE_DATA - " << result.trades_needed << " trades needed\n";
                break;
            case sct::CertaintyStatus::NO_EDGE:
                std::cout << "NO_EDGE - Don't trade\n";
                break;
        }

        std::cout << "    Calc time:   " << result.calc_time_ns << " ns\n";

        // Position sizing
        if (result.status == sct::CertaintyStatus::CERTAIN) {
            auto size = sizer.size_from_stats(test.wins, test.total);
            std::cout << "    Kelly:       " << (size.full_kelly * 100) << "% full, "
                      << (size.quarter_kelly * 100) << "% quarter\n";
            std::cout << "    Recommended: " << size.capital_pct << "% of capital\n";
        }

        std::cout << "\n";
    }

    // Sample size table
    std::cout << "========================================\n";
    std::cout << "SAMPLE SIZE REQUIREMENTS\n";
    std::cout << "For 99% confidence that lower bound >= 50.75%\n";
    std::cout << "========================================\n\n";

    double test_wrs[] = {0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.65, 0.70, 0.80, 0.90};

    std::cout << "Observed WR | Min Trades | Status\n";
    std::cout << "-----------|------------|--------\n";

    for (double wr : test_wrs) {
        int trades = checker.trades_needed(wr);
        std::cout << std::setw(10) << (wr * 100) << "% | ";

        if (trades > 0) {
            std::cout << std::setw(10) << trades << " | POSSIBLE\n";
        } else {
            std::cout << std::setw(10) << "N/A" << " | IMPOSSIBLE\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "INTERPRETATION:\n";
    std::cout << "- Higher observed WR needs fewer samples\n";
    std::cout << "- Below 51%, impossible to be certain\n";
    std::cout << "- CI narrows as sample size increases\n";
    std::cout << "========================================\n";

    return 0;
}

#endif
