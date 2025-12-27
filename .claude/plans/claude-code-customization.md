# Claude Code Customization Plan: Renaissance-Grade Enhancement

## Executive Summary

Transform Claude Code into a specialized trading system development environment optimized for the Sovereign Engine codebase. This plan applies Renaissance Technologies-style principles: **systematic optimization, information advantage, and speed at every layer**.

---

## Current Limitations

| Problem | Impact | Solution |
|---------|--------|----------|
| 200K token context window | Can't hold full codebase | Vector DB semantic search |
| Generic prompts | Wastes tokens explaining trading concepts | Custom CLAUDE.md with domain knowledge |
| Manual workflows | Slow iteration cycles | Custom subagents + slash commands |
| No formula awareness | Doesn't understand 900+ formula IDs | Specialized MCP server |
| Generic code style | Inconsistent with engine patterns | Path-specific rules |

---

## Phase 1: Memory Architecture (CLAUDE.md System)

### 1.1 Project Memory Structure

```
livetrading/
├── CLAUDE.md                          # Master instructions
├── .claude/
│   ├── settings.json                  # Tool permissions, model settings
│   ├── settings.local.json            # Local API keys, personal prefs
│   ├── CLAUDE.md                      # Engine-specific context
│   ├── rules/
│   │   ├── trading-domain.md          # Trading terminology, concepts
│   │   ├── formula-system.md          # Formula ID conventions (1-72099)
│   │   ├── code-style.md              # Python patterns, typing
│   │   ├── engine-architecture.md     # Sovereign Engine structure
│   │   ├── testing.md                 # Test patterns, backtest conventions
│   │   ├── blockchain.md              # Bitcoin/on-chain specifics
│   │   └── security.md                # API key handling, no hardcoding
│   ├── agents/
│   │   ├── formula-dev.md             # Formula development specialist
│   │   ├── backtest-runner.md         # Backtest execution agent
│   │   ├── signal-analyzer.md         # Signal quality analysis
│   │   ├── regime-detector.md         # Market regime analysis
│   │   └── performance-auditor.md     # PnL and metrics review
│   └── commands/
│       ├── new-formula.md             # Create new formula template
│       ├── backtest.md                # Run backtests
│       ├── signal-check.md            # Validate signal generation
│       └── deploy-check.md            # Pre-deployment validation
```

### 1.2 Master CLAUDE.md Content

```markdown
# Sovereign Engine Development Environment

## Project Overview
This is a Bitcoin trading system using 900+ mathematical formulas for signal generation.
The system trades BTC/USD using blockchain on-chain data, exchange flows, and market microstructure.

## Architecture
@.claude/rules/engine-architecture.md

## Key Commands
- Build: `python -m pytest tests/ -v`
- Run Paper: `python -m engine.sovereign.run_sovereign --mode paper`
- Run Backtest: `python -m engine.backtest.runner --start 2024-01-01`

## Formula System
@.claude/rules/formula-system.md

## Code Conventions
- Type hints required on all functions
- Dataclasses for data structures, not dicts
- NumPy vectorization over loops
- No hardcoded API keys

## Critical Directories
- `engine/sovereign/` - Main trading engine
- `engine/sovereign/formulas/` - All 900+ formula implementations
- `engine/sovereign/core/` - Types, config, main engine
- `data/` - Historical data, parquet files

## Never Do
- Never modify .env files
- Never commit API keys
- Never change formula IDs once assigned
- Never use `print()` in production code (use logging)
```

### 1.3 Path-Specific Rules

**`.claude/rules/formulas.md`:**
```markdown
---
paths: engine/sovereign/formulas/**/*.py
---

# Formula Development Rules

## Formula ID Conventions
- 10001-10999: Adaptive formulas (Kelly, regime-adaptive)
- 20001-20999: Pattern recognition (chart patterns, momentum)
- 30001-30999: Statistical arbitrage
- 70001-70999: ML-based (QLib, FinRL)
- 72001-72099: RenTech patterns (HMM, GARCH, ensemble)

## Required Interface
Every formula engine MUST implement BaseEngine:
```python
from .base import BaseEngine
from ..core.types import Tick, Signal, TradeOutcome

class MyEngine(BaseEngine):
    def __init__(self):
        super().__init__(name="my_engine", formula_ids=[...])

    def initialize(self, config: Dict[str, Any]) -> None: ...
    def process(self, tick: Tick) -> Signal: ...
    def learn(self, outcome: TradeOutcome) -> None: ...
```

## Signal Generation
- Confidence must be 0.0-1.0
- Direction: 1=LONG, -1=SHORT, 0=NEUTRAL
- Always include formula_ids in Signal for attribution
```

---

## Phase 2: Custom Subagents

### 2.1 Formula Development Agent

**`.claude/agents/formula-dev.md`:**
```markdown
---
name: formula-dev
description: Specialized agent for developing and debugging trading formulas
tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
---

# Formula Development Specialist

You are an expert in quantitative trading formula development for the Sovereign Engine.

## Your Expertise
- Mathematical trading formulas (momentum, mean-reversion, volatility)
- HMM regime detection with hmmlearn
- GARCH volatility modeling with arch library
- LightGBM ensemble methods
- Bitcoin on-chain analytics (whale flows, exchange flows)

## When Developing Formulas
1. Check existing formula IDs in use: `grep -r "formula_ids" engine/sovereign/formulas/`
2. Use next available ID in the appropriate range
3. Implement BaseEngine interface
4. Add to FormulaRegistry in `formulas/registry.py`
5. Write unit tests in `tests/formulas/`

## Signal Quality Requirements
- Minimum 100 ticks before generating signals
- Confidence threshold > 0.5 for tradeable signals
- Include stop_loss and take_profit in all signals

## Common Patterns
@.claude/rules/formula-patterns.md
```

### 2.2 Backtest Runner Agent

**`.claude/agents/backtest-runner.md`:**
```markdown
---
name: backtest-runner
description: Runs and analyzes backtests on historical data
tools:
  - Read
  - Bash
  - Glob
---

# Backtest Execution Specialist

You execute and analyze backtests for the Sovereign Engine.

## Running Backtests
```bash
python -m engine.backtest.runner \
    --start 2024-01-01 \
    --end 2024-12-01 \
    --engines rentech,adaptive \
    --capital 10000
```

## Key Metrics to Report
- Total Return (%)
- Sharpe Ratio (annualized)
- Max Drawdown (%)
- Win Rate (%)
- Profit Factor
- Number of Trades

## Data Locations
- Historical prices: `data/bitcoin_2021_2025.db`
- Exchange flows: `data/exchange_flows_2022_2025.db`
- Features DB: `data/bitcoin_features.db`

## Output Analysis
After backtest, always:
1. Load results from `backtest_results.json`
2. Calculate key metrics
3. Compare to buy-and-hold baseline
4. Identify worst drawdown periods
```

### 2.3 Signal Analyzer Agent

**`.claude/agents/signal-analyzer.md`:**
```markdown
---
name: signal-analyzer
description: Analyzes signal quality and formula performance
tools:
  - Read
  - Grep
  - Glob
  - mcp__ide__executeCode
---

# Signal Quality Analyst

You analyze the quality of trading signals from the Sovereign Engine.

## Analysis Framework
1. **Signal Distribution**: Long vs Short ratio
2. **Confidence Distribution**: Histogram of confidence values
3. **Formula Attribution**: Which formulas generate most signals
4. **Regime Correlation**: Signal quality by market regime
5. **Time-of-Day Patterns**: When signals perform best

## Key Questions to Answer
- Are signals clustered or well-distributed?
- Do high-confidence signals outperform?
- Which formula IDs have best win rates?
- Are there regime-specific patterns?

## SQL Queries for Analysis
Use the historical database at `data/bitcoin_features.db`
```

---

## Phase 3: Custom Slash Commands

### 3.1 New Formula Template

**`.claude/commands/new-formula.md`:**
```markdown
Create a new trading formula with the following specifications:

Formula Name: $ARGUMENTS

## Steps
1. Determine appropriate formula ID range
2. Create new file in `engine/sovereign/formulas/`
3. Implement BaseEngine interface
4. Add wrapper if needed for integration
5. Register in FormulaRegistry
6. Create basic unit test

## Template
Use this structure:
```python
"""
{Name} Formula - Sovereign Engine
{Description}
"""
from typing import Dict, Any
from .base import BaseEngine
from ..core.types import Tick, Signal, TradeOutcome

class {Name}Engine(BaseEngine):
    def __init__(self):
        super().__init__(
            name="{name_lower}",
            formula_ids=[{next_id}]
        )

    def initialize(self, config: Dict[str, Any]) -> None:
        self._initialized = True

    def process(self, tick: Tick) -> Signal:
        self.state.ticks_processed += 1
        # Implementation here
        return self._no_signal()

    def learn(self, outcome: TradeOutcome) -> None:
        self.state.update_from_outcome(outcome)
```
```

### 3.2 Quick Backtest

**`.claude/commands/backtest.md`:**
```markdown
Run a quick backtest with the specified parameters.

Arguments: $ARGUMENTS
(Format: "engine_name start_date end_date" or just "engine_name" for last 30 days)

## Execution
```bash
python -m engine.backtest.runner --engines $ENGINE --start $START --end $END --capital 10000
```

## After Completion
1. Read the results file
2. Calculate Sharpe ratio
3. Report max drawdown
4. Compare to baseline
```

### 3.3 Signal Check

**`.claude/commands/signal-check.md`:**
```markdown
Validate signal generation for specified engine.

Engine: $ARGUMENTS

## Validation Steps
1. Run engine on last 1000 ticks
2. Count signals generated
3. Check confidence distribution
4. Verify no errors in logs
5. Report signal quality metrics

## Quick Test
```python
from engine.sovereign.core import create_engine, create_paper_config

config = create_paper_config()
engine = create_engine(config)
# Run 1000 tick simulation
```
```

---

## Phase 4: MCP Server for Formula Context

### 4.1 Formula Knowledge MCP Server

Create a custom MCP server that provides formula-specific context:

**`mcp_servers/formula_server.py`:**
```python
"""
Formula Knowledge MCP Server
Provides semantic search over 900+ trading formulas
"""
from mcp.server import Server
from mcp.types import Tool, TextContent
import sqlite3
import json

server = Server("formula-knowledge")

# Formula database
FORMULA_DB = {
    # ID ranges and descriptions
    "10001-10005": "Adaptive Kelly betting formulas",
    "20001-20012": "Chart pattern recognition",
    "72001-72099": "RenTech patterns (HMM, GARCH, ensemble)",
    # ... more formulas
}

@server.tool()
async def search_formulas(query: str) -> list[TextContent]:
    """Search for formulas by description or ID range."""
    results = []
    for id_range, description in FORMULA_DB.items():
        if query.lower() in description.lower():
            results.append(f"{id_range}: {description}")
    return [TextContent(type="text", text="\n".join(results))]

@server.tool()
async def get_formula_usage(formula_id: int) -> TextContent:
    """Get usage examples and performance stats for a formula."""
    # Query historical performance from database
    pass

@server.tool()
async def list_engine_formulas(engine_name: str) -> TextContent:
    """List all formulas for a specific engine (adaptive, rentech, etc)."""
    pass
```

### 4.2 MCP Configuration

**`.claude/settings.json`:**
```json
{
  "mcpServers": {
    "formula-knowledge": {
      "command": "python",
      "args": ["-m", "mcp_servers.formula_server"],
      "env": {
        "FORMULA_DB_PATH": "./data/formulas.db"
      }
    },
    "claude-context": {
      "command": "npx",
      "args": ["-y", "@anthropics/claude-context"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

---

## Phase 5: Hooks for Workflow Automation

### 5.1 Pre-Commit Validation Hook

**`.claude/settings.json` (hooks section):**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python -m py_compile \"$TOOL_INPUT_PATH\" 2>&1 || true"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/post-edit.sh"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/load-context.sh"
          }
        ]
      }
    ]
  }
}
```

### 5.2 Context Loading Hook

**`.claude/hooks/load-context.sh`:**
```bash
#!/bin/bash
# Load current market context at session start

# Get current regime from last backtest
REGIME=$(python -c "
from engine.sovereign.formulas.rentech_engine import HMMSubEngine
hmm = HMMSubEngine()
# Load recent prices and detect regime
print('neutral')  # Simplified
" 2>/dev/null || echo "unknown")

# Output context for Claude
cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "Current market regime: $REGIME. Recent BTC price: \$$(curl -s 'https://api.coinbase.com/v2/prices/BTC-USD/spot' | jq -r '.data.amount' 2>/dev/null || echo 'N/A')"
  }
}
EOF
```

---

## Phase 6: Vector Database for Million-Token Context

### 6.1 Claude Context Integration

Use [Claude Context](https://github.com/zilliztech/claude-context) for semantic code search:

```bash
# Install
npm install -g @anthropics/claude-context

# Configure in settings
claude mcp add claude-context \
  --env OPENAI_API_KEY=$OPENAI_API_KEY \
  --env ZILLIZ_CLUSTER_ID=$ZILLIZ_CLUSTER_ID \
  --env ZILLIZ_API_KEY=$ZILLIZ_API_KEY

# Index codebase
claude
> Index this codebase
```

### 6.2 Custom Vector Index for Formulas

Create specialized embeddings for formula code:

```python
# scripts/index_formulas.py
"""
Create semantic index of all 900+ formulas for instant retrieval.
"""
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./data/formula_vectors")
collection = client.create_collection("formulas")

# Index all formula files
for formula_file in Path("engine/sovereign/formulas").glob("*.py"):
    code = formula_file.read_text()
    embedding = model.encode(code)
    collection.add(
        documents=[code],
        embeddings=[embedding.tolist()],
        ids=[formula_file.stem],
        metadatas=[{"path": str(formula_file)}]
    )
```

---

## Phase 7: Speed Optimizations (Renaissance Style)

### 7.1 Parallel Agent Execution

Run multiple Claude instances for parallel development:

```bash
# Terminal 1: Formula development
cd livetrading && git worktree add ../livetrading-formulas feature/new-formula
cd ../livetrading-formulas && claude

# Terminal 2: Backtest analysis
cd livetrading && git worktree add ../livetrading-backtest feature/backtest
cd ../livetrading-backtest && claude

# Terminal 3: Code review
cd livetrading && claude
```

### 7.2 Cached Context Loading

Pre-compute expensive context to load instantly:

```bash
# Generate daily context cache
python scripts/generate_context_cache.py > .claude/cache/daily-context.md

# Reference in CLAUDE.md
@.claude/cache/daily-context.md
```

### 7.3 Optimized Settings

**`.claude/settings.json`:**
```json
{
  "model": "claude-sonnet-4-20250514",
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(python:*)",
      "Bash(pytest:*)",
      "Write(engine/**)",
      "Edit(engine/**)"
    ],
    "deny": [
      "Write(.env*)",
      "Bash(rm -rf *)",
      "Bash(*api_key*)"
    ]
  },
  "contextOptimization": {
    "maxMcpOutputTokens": 50000,
    "compactThreshold": 150000
  }
}
```

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Create CLAUDE.md master file
- [ ] Set up `.claude/rules/` directory with domain rules
- [ ] Configure basic settings.json

### Week 2: Subagents
- [ ] Create formula-dev agent
- [ ] Create backtest-runner agent
- [ ] Create signal-analyzer agent
- [ ] Test agent delegation

### Week 3: Commands & Hooks
- [ ] Implement slash commands (/new-formula, /backtest, /signal-check)
- [ ] Create SessionStart hook for context loading
- [ ] Create PostToolUse hooks for validation

### Week 4: MCP & Vector DB
- [ ] Set up Claude Context for semantic search
- [ ] Create formula knowledge MCP server
- [ ] Index full codebase

### Week 5: Optimization
- [ ] Configure parallel worktrees
- [ ] Implement context caching
- [ ] Fine-tune token usage
- [ ] Benchmark speed improvements

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Context per session | ~50K tokens | ~500K effective (via vector search) |
| Formula lookup time | Manual grep | Instant semantic search |
| New formula creation | 30+ minutes | 5 minutes (template + agent) |
| Backtest iteration | Manual commands | Single slash command |
| Code review coverage | Partial | Full (parallel agents) |

---

## Sources

- [Claude Code Memory Management](https://code.claude.com/docs/en/memory)
- [Claude Code Hooks Reference](https://code.claude.com/docs/en/hooks)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Code Plugins](https://claude.com/blog/claude-code-plugins)
- [Claude Context (Vector Search)](https://github.com/zilliztech/claude-context)
- [Awesome Claude Code Subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
