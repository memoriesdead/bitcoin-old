# Research Methodology - Systematic Knowledge Extraction

## The Research Formula

**K = Σ(S × A × V × R) / N**

Where:
- **K** = Knowledge Output (actionable findings)
- **S** = Source Quality Score
- **A** = Authority Weight
- **V** = Verification Count
- **R** = Relevance to Our Problem
- **N** = Noise Filtered Out

---

## Problem Definition

### Our Edge Equation
```
Edge = (WR × AvgWin) - ((1-WR) × AvgLoss) - Fees
```

### Current State (NEGATIVE EDGE)
- **V1**: WR = 0%, Final Capital = $0 (died)
- **V2**: WR = 1%, Final Capital = $0 (died)
- **V3**: WR = 26%, Final Capital = $0.0003
- **V4**: WR = 28%, Final Capital = $0.008
- **V5**: WR = 0%, Final Capital = $0 (died)
- **V6**: WR = 27%, Final Capital = $0.001
- **V7**: WR = 5%, Final Capital = $0
- **V8**: WR = 0%, Final Capital = $0 (died)

**Fees**: 0.04% per trade (taker)

### Target State (POSITIVE EDGE)
- **Edge** > $0.02 per trade
- **WR** ≥ 55%
- **Profit Factor** ≥ 2.0
- **AvgWin/AvgLoss** > 2.7 (at WR=27%)

---

## Research Variables

### Variable 1: Win Rate (WR)
- **Current**: 0%-28%
- **Target**: 55%+
- **Research Focus**: What parameters increase WR?

### Variable 2: Average Win (AvgWin)
- **Current**: Small (exits at z=0.03)
- **Target**: 2.7× AvgLoss minimum
- **Research Focus**: Optimal exit parameters

### Variable 3: Average Loss (AvgLoss)
- **Current**: Unknown (includes fees)
- **Target**: Minimize while allowing wins
- **Research Focus**: Stop-loss approach

### Variable 4: Trade Frequency (N)
- **Current**: 108k-155k trades
- **Target**: Quality over quantity
- **Research Focus**: Optimal frequency

### Variable 5: Fee Impact (F)
- **Current**: 0.04% eating all edge
- **Target**: Edge >> Fees
- **Research Focus**: Minimum edge threshold

---

## Authority Scoring Matrix

| Source Type | Score | Verification Weight |
|-------------|-------|---------------------|
| Peer-reviewed journal (JoF, RFS) | 10 | 3× |
| Academic working paper (SSRN, arXiv) | 9 | 2.5× |
| Institutional research (SEC, Fed) | 8 | 2.5× |
| Quant practitioner book (Chan, etc) | 8 | 2× |
| Hedge fund research (AQR, Two Sigma) | 7 | 2× |
| PhD thesis | 7 | 2× |
| CFA/FRM curriculum | 6 | 1.5× |
| Wilmott/Nuclear Phynance forum | 5 | 1.5× |
| Verified quant blog | 4 | 1× |
| QuantConnect/Quantopian | 4 | 1× |
| Reddit r/quant (verified) | 3 | 1× |
| Medium (with citations) | 2 | 0.5× |
| Anonymous forum post | 1 | 0.5× |
| No citation/source | 0 | 0× (REJECT) |

**Minimum Score**: 5
**Minimum Verifications**: 2

---

## Relevance Scoring

| Criterion | Score |
|-----------|-------|
| Directly addresses our exact problem | 10 |
| Addresses mean reversion trading | 9 |
| Addresses HFT/high-frequency | 8 |
| Addresses crypto specifically | 8 |
| Addresses transaction cost impact | 8 |
| Addresses win rate optimization | 7 |
| Addresses position sizing | 6 |
| General trading edge formula | 5 |
| Addresses different asset class | 3 |
| Theoretical only, no practical app | 2 |
| Unrelated to our problem | 0 |

**Minimum Relevance**: 6

---

## Success Criteria

✓ 50+ sources scored and filtered
✓ 20+ formulas extracted and verified
✓ Each formula verified in 2+ sources
✓ Authority score ≥ 5 for all included
✓ Relevance score ≥ 6 for all included
✓ Clear implementation order defined
✓ Expected edge improvement quantified
✓ V1-V8 each have specific fix plan

**ULTIMATE SUCCESS**: Edge equation flips from NEGATIVE to POSITIVE
