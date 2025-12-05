# MASTER FORMULA EXTRACTION INDEX
## Context Retention System

**Purpose:** This file tracks ALL extracted formulas across sessions. Read this first in any new session.

---

## EXTRACTION STATUS

| Source | File | IDs | Status | Formulas |
|--------|------|-----|--------|----------|
| USPTO Patents | `patents.md` | 591-640 | ✅ COMPLETE | 50 |
| arXiv q-fin / Academic | `academic_core.md` | 641-719 | ✅ COMPLETE | 79 |
| Hedge Fund SEC | `sec_filings.md` | 761-800 | ✅ COMPLETE | 40 |
| Blockchain MEV | `blockchain_mev.md` | 482-519 | ✅ COMPLETE | 38 |
| Risk Management | `risk_mgmt.md` | 223-238 | ✅ COMPLETE | 16 |
| Top Journals | `journals.md` | 741-760 | ✅ COMPLETE | 20 |
| QuantLib | `quantlib.md` | 811-900 | PENDING | 0 |

**TOTAL EXTRACTED: 259 formulas**

### Sources Used:
- [Flashbots REV](https://writings.flashbots.net/quantifying-rev)
- [arXiv:2405.17944](https://arxiv.org/html/2405.17944v1) - MEV Sandwich Attacks
- [arXiv:2408.03594](https://arxiv.org/html/2408.03594v1) - Hawkes Order Flow
- [US8140416B2](https://patents.google.com/patent/US8140416B2/en) - Hidden Orders Patent
- [US8719146B2](https://patents.google.com/patent/US8719146B2/en) - Micro Auction Patent
- Kyle (1985) Econometrica - Market Impact
- Almgren-Chriss (2000) J. Risk - Optimal Execution
- Easley et al. (2012) RFS - VPIN
- Heston (1993) RFS - Stochastic Volatility
- Gatheral et al. (2018) Quant Finance - Rough Volatility
- Avellaneda & Stoikov (2008) Quant Finance - Market Making
- Bollerslev (1986) J. Econometrics - GARCH
- Nelson (1991) Econometrica - EGARCH
- Uniswap V2 Whitepaper (2020) - AMM Pricing
- Artzner et al. (1999) Mathematical Finance - Coherent Risk Measures
- Black & Scholes (1973) J. Political Economy - Option Pricing
- Fama & French (1993) J. Financial Economics - Factor Models
- Merton (1976) J. Financial Economics - Jump Diffusion
- Uhlenbeck & Ornstein (1930) Physical Review - Mean Reversion
- Kelly (1956) Bell System Technical Journal - Optimal Betting
- Sharpe (1966) Journal of Business - Performance Measurement
- Sortino & Price (1994) Journal of Investing - Downside Risk
- Vasicek (1977) J. Financial & Quantitative Analysis - Interest Rates
- Cox, Ingersoll, Ross (1985) Econometrica - CIR Model
- Atis Elsts (2021) "Liquidity Math in Uniswap V3"
- ETH Zurich (2021) "Analyzing and Preventing Sandwich Attacks"
- Flashbots Docs - Bundle Pricing
- US8571967B1 - Algorithmic Trading Strategies Patent
- US20150066727A1 - Order Execution Delay Patent
- Perold (1988) "Implementation Shortfall"
- Amihud (2002) J. Financial Markets - Illiquidity
- Sharpe (1964) J. Finance - CAPM
- Lintner (1965) RFS - Security Prices
- Ross (1976) J. Economic Theory - APT
- Markowitz (1952) J. Finance - Portfolio Selection
- Carhart (1997) J. Finance - Four-Factor Model
- Jegadeesh & Titman (1993) J. Finance - Momentum
- Fama (1970) J. Finance - Efficient Markets
- Lo & MacKinlay (1988) RFS - Variance Ratio
- Pastor & Stambaugh (2003) J. Political Economy - Liquidity Risk
- Jorion (2006) Value at Risk - VaR Decomposition
- McNeil & Frey (2000) J. Empirical Finance - EVT
- Maillard et al. (2010) J. Portfolio Management - Risk Parity

---

## NEXT AVAILABLE IDs

- Filled: 223-238, 482-519, 591-719, 741-800
- Remaining Gaps: 308-310, 427-430, 436-445, 454-460, 468-475, 524-589, 720-740
- Next Available: 811+ (QuantLib)

---

## SESSION LOG

| Date | Session | Source | IDs Added | Commit |
|------|---------|--------|-----------|--------|
| 2024-XX-XX | 1 | - | - | - |

---

## HOW TO USE

1. **New Session:** Read this file first
2. **Research:** Pick ONE source from table above
3. **Extract:** Add formulas to corresponding file
4. **Update:** Mark status as COMPLETE, update formula count
5. **Commit:** Save immediately

---

## FORMULA TEMPLATE

```python
# ID: XXX
# Source: Author (Year) "Title" Journal
# URL: https://...
# Formula: LaTeX here

@FormulaRegistry.register(XXX, "Name", "category")
class ClassName(BaseFormula):
    """
    Source: Author (Year) "Title" Journal
    URL: https://...

    Formula: $$formula$$

    Edge: Description of alpha generation
    """

    def _compute(self):
        pass
```
