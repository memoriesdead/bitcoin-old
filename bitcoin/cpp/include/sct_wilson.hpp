/**
 * SCT Wilson CI Calculator - Nanosecond Speed
 *
 * Statistical Certainty Trading using Wilson score intervals.
 * Only trades when 99% confident that win rate >= 50.75%.
 *
 * Wilson CI is more accurate than normal approximation for
 * binomial proportions, especially near 0 or 1.
 */

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <chrono>

namespace sct {

// Z-scores for common confidence levels
constexpr double Z_90 = 1.645;
constexpr double Z_95 = 1.960;
constexpr double Z_99 = 2.576;
constexpr double Z_999 = 3.291;

// RenTech threshold
constexpr double MIN_WIN_RATE = 0.5075;  // 50.75%
constexpr double DEFAULT_CONFIDENCE = 0.99;

enum class CertaintyStatus {
    CERTAIN,        // Lower bound >= MIN_WIN_RATE
    NEED_MORE_DATA, // Win rate looks good but need more samples
    NO_EDGE         // Win rate below threshold
};

struct WilsonInterval {
    double lower;
    double upper;
    double center;
};

struct CertaintyResult {
    double observed_wr;
    double lower_bound;
    double upper_bound;
    double target_wr;
    double confidence;
    int trades_needed;  // 0 if already certain, -1 if impossible
    CertaintyStatus status;
    int64_t calc_time_ns;  // Calculation time in nanoseconds
};

struct PositionSize {
    double win_rate_used;    // Lower bound for safety
    double full_kelly;
    double quarter_kelly;
    double recommended;
    double capital_pct;
};

class WilsonCalculator {
public:
    /**
     * Calculate Wilson score confidence interval.
     *
     * More accurate than normal approximation for binomial proportions.
     *
     * @param wins Number of winning trades
     * @param total Total trades
     * @param confidence Confidence level (default 0.99 for 99%)
     * @return WilsonInterval with lower and upper bounds
     */
    static WilsonInterval wilson_interval(int wins, int total,
                                          double confidence = DEFAULT_CONFIDENCE);

    /**
     * Get Wilson lower bound only (faster).
     */
    static double wilson_lower_bound(int wins, int total,
                                     double confidence = DEFAULT_CONFIDENCE);

    /**
     * Get z-score for confidence level.
     */
    static double get_z_score(double confidence);
};

class CertaintyChecker {
public:
    CertaintyChecker(double min_wr = MIN_WIN_RATE,
                     double confidence = DEFAULT_CONFIDENCE);

    /**
     * Check if we are certain of the minimum win rate.
     *
     * Returns true only if the lower bound of the Wilson CI >= min_win_rate.
     * This is MATHEMATICAL certainty, not just observed win rate.
     *
     * @param wins Number of winning trades
     * @param total Total trades
     * @return CertaintyResult with full analysis
     */
    CertaintyResult check(int wins, int total) const;

    /**
     * Calculate how many more trades needed to reach certainty.
     *
     * @param observed_wr Current observed win rate
     * @return Number of trades needed, or -1 if impossible
     */
    int trades_needed(double observed_wr) const;

    /**
     * Quick check: is_certain?
     */
    bool is_certain(int wins, int total) const;

    // Getters
    double min_win_rate() const { return min_wr_; }
    double confidence() const { return confidence_; }

private:
    double min_wr_;
    double confidence_;
    double z_score_;
};

class KellyPositionSizer {
public:
    // Safety parameters
    static constexpr double KELLY_FRACTION = 0.25;    // Quarter-Kelly
    static constexpr double MAX_POSITION_PCT = 0.05;  // 5% max
    static constexpr double MIN_POSITION_PCT = 0.001; // 0.1% min

    KellyPositionSizer(double kelly_fraction = KELLY_FRACTION,
                       double min_wr = MIN_WIN_RATE);

    /**
     * Calculate Kelly fraction.
     *
     * f* = (b*p - q) / b
     *
     * @param win_rate Win probability
     * @param risk_reward Risk/reward ratio (default 1:1)
     * @return Optimal fraction of capital
     */
    static double kelly_fraction(double win_rate, double risk_reward = 1.0);

    /**
     * Calculate position size from trade stats.
     *
     * Uses Wilson CI lower bound for safety.
     *
     * @param wins Winning trades
     * @param total Total trades
     * @param risk_reward Risk/reward ratio
     * @return PositionSize with recommendations
     */
    PositionSize size_from_stats(int wins, int total,
                                 double risk_reward = 1.0) const;

    /**
     * Calculate position size from known win rate.
     */
    PositionSize size_from_win_rate(double win_rate,
                                    double risk_reward = 1.0) const;

private:
    double kelly_mult_;  // Kelly multiplier (0.25 for quarter)
    double min_wr_;
    CertaintyChecker checker_;
};

class StrategyTracker {
public:
    struct StrategyStats {
        std::string name;
        int wins = 0;
        int losses = 0;
        int64_t created_at;
        int64_t updated_at;

        int total() const { return wins + losses; }
        double win_rate() const {
            return total() > 0 ? static_cast<double>(wins) / total() : 0.0;
        }
    };

    StrategyTracker(double min_wr = MIN_WIN_RATE);

    // Record trade outcome
    void record_trade(const std::string& strategy, bool won);

    // Get strategy stats
    const StrategyStats* get_stats(const std::string& strategy) const;

    // Get all tradeable strategies (certain of edge)
    std::vector<std::string> get_tradeable() const;

    // Get all strategies needing more data
    std::vector<std::string> get_pending() const;

    // Check strategy certainty
    CertaintyResult check_strategy(const std::string& strategy) const;

private:
    CertaintyChecker checker_;
    std::unordered_map<std::string, StrategyStats> strategies_;
};

}  // namespace sct
