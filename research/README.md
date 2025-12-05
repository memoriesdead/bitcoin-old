# Research Workflow

## Context Window Limitation

Claude Code has a 200k context window. When doing internet research, the context fills up and Claude loses track of:
- What was already researched
- Source URLs and citations
- Extracted formulas and their origins

## Solution: Chunked Research Workflow

### Step 1: Research Phase
- Pick ONE topic/paper source
- Fetch and extract formulas
- Save to a dedicated file immediately

### Step 2: Extract & Label
- Each formula gets:
  - Unique ID
  - Source URL
  - Paper citation (Author, Year, Journal)
  - Mathematical formula (LaTeX)
  - Python implementation

### Step 3: Save & Clear
- Save research to `research/[topic].md`
- Commit the file
- Start new session for next topic

### Step 4: Rinse & Repeat
- New session reads only the specific research file needed
- Implements formulas from that file
- Keeps context tight

## File Structure

```
research/
├── README.md                 # This file
├── arxiv_q-fin/             # arXiv quantitative finance papers
├── journals/                 # Peer-reviewed journal papers
├── books/                    # Textbook formulas
└── extracted/               # Final extracted formulas ready for implementation
```

## Bypass Mode

Bypass mode is ON - allows unrestricted web research for academic sources.

## Gold Standard Sources Only

- arXiv q-fin: https://arxiv.org/list/q-fin/recent
- SSRN: https://www.ssrn.com/index.cfm/en/fin/
- Journal of Finance
- Review of Financial Studies
- Econometrica
- Mathematical Finance

NO: Medium, blogs, YouTube, Reddit, Wikipedia, Investopedia
