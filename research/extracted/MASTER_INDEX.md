# MASTER FORMULA EXTRACTION INDEX
## Context Retention System

**Purpose:** This file tracks ALL extracted formulas across sessions. Read this first in any new session.

---

## EXTRACTION STATUS

| Source | File | IDs | Status | Formulas |
|--------|------|-----|--------|----------|
| USPTO Patents | `patents.md` | 591-640 | IN PROGRESS | 11 |
| arXiv q-fin / Academic | `academic_core.md` | 641-719 | IN PROGRESS | 29 |
| Hedge Fund SEC | `sec_filings.md` | 761-800 | PENDING | 0 |
| Blockchain MEV | `blockchain_mev.md` | 482-519 | IN PROGRESS | 9 |
| Risk Management | `risk_mgmt.md` | 223-238 | PENDING | 0 |
| Top Journals | `journals.md` | 741-760 | PENDING | 0 |
| QuantLib | `quantlib.md` | 811-900 | PENDING | 0 |

**TOTAL EXTRACTED: 49 formulas**

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

---

## NEXT AVAILABLE IDs

- Gap IDs: 223-238, 308-310, 427-430, 436-445, 454-460, 468-475, 482-519, 524-589, 591-719, 761-800
- New IDs: 811+

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
