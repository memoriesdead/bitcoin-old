# MASTER FORMULA EXTRACTION INDEX
## Context Retention System

**Purpose:** This file tracks ALL extracted formulas across sessions. Read this first in any new session.

---

## EXTRACTION STATUS

| Source | File | IDs | Status | Formulas |
|--------|------|-----|--------|----------|
| USPTO Patents | `patents.md` | 591-640 | IN PROGRESS | 11 |
| arXiv q-fin / Academic | `academic_core.md` | 641-719 | IN PROGRESS | 20 |
| Hedge Fund SEC | `sec_filings.md` | 761-800 | PENDING | 0 |
| Blockchain MEV | `blockchain_mev.md` | 482-519 | IN PROGRESS | 9 |
| Risk Management | `risk_mgmt.md` | 223-238 | PENDING | 0 |
| Top Journals | `journals.md` | 741-760 | PENDING | 0 |
| QuantLib | `quantlib.md` | 811-900 | PENDING | 0 |

**TOTAL EXTRACTED: 40 formulas**

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
