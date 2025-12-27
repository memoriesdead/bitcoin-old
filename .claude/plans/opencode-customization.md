# OpenCode Customization Plan: Renaissance-Grade Trading AI

## Executive Summary

**OpenCode** is the superior choice for full source code customization:
- 100% open source (MIT license)
- 38K+ stars, battle-tested
- TypeScript codebase (easy to modify)
- 75+ provider support (Anthropic, OpenAI, Ollama, local models)
- Clean modular architecture

**Location**: `tools/opencode/` (already cloned)

---

## Architecture Overview

```
packages/opencode/src/
├── agent/           # Agent definitions (general, explore, build, plan)
├── provider/        # Model providers (Anthropic, OpenAI, Ollama, etc.)
├── tool/            # All tools (bash, edit, read, write, grep, glob)
├── config/          # Configuration system
├── mcp/             # Model Context Protocol integration
├── session/         # Conversation/session management
├── lsp/             # Language Server Protocol integration
├── storage/         # SQLite persistence
└── cli/             # CLI interface
```

---

## Customization Phases

### Phase 1: Custom Trading Tools

Create specialized tools for Sovereign Engine development.

**`packages/opencode/src/tool/formula.ts`:**
```typescript
import { z } from "zod"
import { Tool } from "./tool"

export const FormulaTool: Tool.Info = {
  id: "formula",
  init: async () => ({
    description: "Search and analyze Sovereign Engine trading formulas (IDs 1-72099)",
    parameters: z.object({
      action: z.enum(["search", "info", "validate", "stats"]),
      query: z.string().optional(),
      formula_id: z.number().optional(),
    }),
    execute: async (args, ctx) => {
      const formulaDb = {
        "10001-10005": { name: "Adaptive Kelly", engine: "adaptive" },
        "20001-20012": { name: "Pattern Recognition", engine: "pattern" },
        "72001-72099": { name: "RenTech Patterns", engine: "rentech" },
      }

      if (args.action === "search") {
        // Search formulas by name/description
        const results = Object.entries(formulaDb)
          .filter(([_, v]) => v.name.toLowerCase().includes(args.query?.toLowerCase() || ""))
        return {
          title: `Found ${results.length} formulas`,
          output: JSON.stringify(results, null, 2),
          metadata: {},
        }
      }

      if (args.action === "info" && args.formula_id) {
        // Get formula details
        // Query actual formula from engine
        return {
          title: `Formula ${args.formula_id}`,
          output: `Formula ID: ${args.formula_id}\nEngine: ...\nDescription: ...`,
          metadata: {},
        }
      }

      return { title: "Unknown action", output: "", metadata: {} }
    },
  }),
}
```

**`packages/opencode/src/tool/backtest.ts`:**
```typescript
import { z } from "zod"
import { Tool } from "./tool"
import { spawn } from "child_process"

export const BacktestTool: Tool.Info = {
  id: "backtest",
  init: async () => ({
    description: "Run backtests on Sovereign Engine formulas",
    parameters: z.object({
      engines: z.string().describe("Comma-separated engine names: adaptive,pattern,rentech"),
      start_date: z.string().optional(),
      end_date: z.string().optional(),
      capital: z.number().default(10000),
    }),
    execute: async (args, ctx) => {
      const cmd = `python -m engine.backtest.runner --engines ${args.engines} --capital ${args.capital}`

      return new Promise((resolve) => {
        const proc = spawn("python", ["-m", "engine.backtest.runner", ...])
        let output = ""
        proc.stdout.on("data", (data) => output += data)
        proc.on("close", () => {
          resolve({
            title: "Backtest Complete",
            output,
            metadata: { engines: args.engines },
          })
        })
      })
    },
  }),
}
```

**`packages/opencode/src/tool/signal.ts`:**
```typescript
import { z } from "zod"
import { Tool } from "./tool"

export const SignalTool: Tool.Info = {
  id: "signal",
  init: async () => ({
    description: "Analyze trading signals from Sovereign Engine",
    parameters: z.object({
      action: z.enum(["live", "history", "quality"]),
      engine: z.string().optional(),
      limit: z.number().default(100),
    }),
    execute: async (args, ctx) => {
      // Connect to engine and fetch signals
      return {
        title: `Signal Analysis: ${args.action}`,
        output: "Signal data...",
        metadata: {},
      }
    },
  }),
}
```

**Register in `tool/registry.ts`:**
```typescript
import { FormulaTool } from "./formula"
import { BacktestTool } from "./backtest"
import { SignalTool } from "./signal"

// Add to the all() function:
return [
  // ... existing tools
  FormulaTool,
  BacktestTool,
  SignalTool,
  ...custom,
]
```

---

### Phase 2: Custom Trading Agent

Create a specialized "sovereign" agent for trading development.

**`packages/opencode/src/agent/sovereign.ts`:**
```typescript
// Add to agent.ts state initialization:

sovereign: {
  name: "sovereign",
  description: `Sovereign Engine trading system specialist. Expert in:
- 900+ trading formulas (IDs 1-72099)
- HMM regime detection, GARCH volatility
- Bitcoin on-chain analytics
- Backtest execution and analysis
- Signal quality evaluation`,
  tools: {
    formula: true,
    backtest: true,
    signal: true,
    bash: true,
    read: true,
    edit: true,
    write: true,
    grep: true,
    glob: true,
    ...defaultTools,
  },
  prompt: `You are a quantitative trading specialist for the Sovereign Engine.

## Your Expertise
- Mathematical trading formulas (momentum, mean-reversion, volatility)
- HMM regime detection with hmmlearn
- GARCH volatility modeling with arch library
- LightGBM ensemble methods
- Bitcoin on-chain analytics (whale flows, exchange flows)

## Formula ID Conventions
- 10001-10005: Adaptive formulas (Kelly, regime-adaptive)
- 20001-20012: Pattern recognition (chart patterns, momentum)
- 72001-72099: RenTech patterns (HMM, GARCH, ensemble)

## Code Standards
- Type hints required on all functions
- Dataclasses for data structures
- NumPy vectorization over loops
- BaseEngine interface for all formula engines

## Project Structure
- engine/sovereign/formulas/ - Formula implementations
- engine/sovereign/core/ - Types, config, main engine
- engine/backtest/ - Backtesting framework
- data/ - Historical data (SQLite, parquet)

When developing formulas, always:
1. Check existing formula IDs
2. Implement BaseEngine interface
3. Register in FormulaRegistry
4. Write unit tests`,
  options: {},
  permission: agentPermission,
  mode: "all",
  builtIn: false,
  color: "#FFD700",  // Gold color for trading
},
```

---

### Phase 3: Custom Local Model Provider

Add support for your preferred open source model via Ollama.

**Configuration (`opencode.json`):**
```json
{
  "provider": {
    "ollama": {
      "name": "Ollama Local",
      "api": "http://localhost:11434/v1",
      "npm": "@ai-sdk/openai-compatible",
      "env": [],
      "models": {
        "qwen3-coder": {
          "id": "qwen3:32b",
          "name": "Qwen 3 32B Coder",
          "limit": {
            "context": 131072,
            "output": 8192
          },
          "tool_call": true,
          "temperature": true,
          "cost": {
            "input": 0,
            "output": 0
          }
        },
        "deepseek-coder-v3": {
          "id": "deepseek-coder-v3:latest",
          "name": "DeepSeek Coder V3",
          "limit": {
            "context": 128000,
            "output": 8192
          },
          "tool_call": true
        },
        "codestral": {
          "id": "codestral:latest",
          "name": "Codestral 25.01",
          "limit": {
            "context": 32768,
            "output": 8192
          },
          "tool_call": true
        }
      }
    }
  },
  "model": "ollama/qwen3-coder",
  "agent": {
    "sovereign": {
      "model": "ollama/deepseek-coder-v3"
    }
  }
}
```

---

### Phase 4: Extended Context via Vector Search

Integrate vector search for million-token effective context.

**`packages/opencode/src/tool/vectorsearch.ts`:**
```typescript
import { z } from "zod"
import { Tool } from "./tool"
import Anthropic from "@anthropic-ai/sdk"

export const VectorSearchTool: Tool.Info = {
  id: "vectorsearch",
  init: async () => ({
    description: "Semantic search over Sovereign Engine codebase using embeddings",
    parameters: z.object({
      query: z.string(),
      limit: z.number().default(10),
      file_type: z.string().optional(),
    }),
    execute: async (args, ctx) => {
      // Use ChromaDB or similar for vector search
      // Pre-indexed embeddings of all 900+ formulas
      const results = await searchEmbeddings(args.query, args.limit)
      return {
        title: `Found ${results.length} relevant code sections`,
        output: results.map(r => `${r.file}:${r.line}\n${r.snippet}`).join("\n\n"),
        metadata: { query: args.query },
      }
    },
  }),
}

async function searchEmbeddings(query: string, limit: number) {
  // Connect to local ChromaDB
  // Return most relevant code snippets
  return []
}
```

**Pre-index script (`scripts/index_codebase.py`):**
```python
"""
Index Sovereign Engine codebase for semantic search.
Run once, update incrementally.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./data/code_vectors")
collection = client.get_or_create_collection("sovereign_engine")

def index_file(filepath: Path):
    code = filepath.read_text()
    # Chunk into functions/classes
    chunks = split_into_chunks(code)
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[f"{filepath.stem}_{i}"],
            metadatas=[{"file": str(filepath), "chunk": i}]
        )

# Index all Python files
for py_file in Path("engine").rglob("*.py"):
    index_file(py_file)
```

---

### Phase 5: Speed Optimizations

**1. Response Caching:**
```typescript
// packages/opencode/src/cache/response.ts
const responseCache = new Map<string, { response: string; timestamp: number }>()

export function getCached(key: string, maxAge: number = 3600000) {
  const cached = responseCache.get(key)
  if (cached && Date.now() - cached.timestamp < maxAge) {
    return cached.response
  }
  return null
}

export function setCache(key: string, response: string) {
  responseCache.set(key, { response, timestamp: Date.now() })
}
```

**2. Parallel Tool Execution:**
```typescript
// Already supported - use Promise.all for independent tools
const results = await Promise.all([
  formulaTool.execute({ action: "search", query: "hmm" }),
  backtestTool.execute({ engines: "rentech" }),
])
```

**3. Streaming Responses:**
```typescript
// OpenCode already supports streaming via AI SDK
// Configure in provider settings for lower latency
```

---

### Phase 6: Custom System Prompt

**`packages/opencode/src/session/sovereign-system.txt`:**
```
You are an elite quantitative trading AI built on the Sovereign Engine platform.

## Core Identity
- You understand 900+ mathematical trading formulas
- You are expert in HMM, GARCH, Kalman filters, and ensemble methods
- You know Bitcoin on-chain analytics inside and out

## Project Context
The Sovereign Engine is a Bitcoin trading system with:
- Adaptive formulas (10001-10005): Kelly betting, regime-adaptive sizing
- Pattern formulas (20001-20012): Chart patterns, momentum signals
- RenTech formulas (72001-72099): HMM regime detection, GARCH volatility, ensemble voting

## Your Tools
- `formula`: Search and analyze trading formulas
- `backtest`: Execute backtests with specific parameters
- `signal`: Analyze live and historical trading signals
- `vectorsearch`: Semantic search over the codebase

## Code Standards
1. Always use type hints
2. Implement BaseEngine interface for new formulas
3. Use NumPy vectorization over loops
4. Never hardcode API keys
5. Write tests for all new formulas

## Response Style
- Be precise and quantitative
- Include formula IDs when discussing strategies
- Reference specific code files and line numbers
- Think like a Renaissance Technologies quant
```

---

## Build & Deploy

```bash
# Install dependencies
cd tools/opencode
bun install

# Build the custom version
bun run build

# Link globally
npm link

# Run your custom OpenCode
opencode --model ollama/qwen3-coder
```

---

## Configuration Files

**`opencode.json` (full config):**
```json
{
  "model": "anthropic/claude-sonnet-4-5-20250514",
  "small_model": "anthropic/claude-haiku-4-5",

  "provider": {
    "ollama": {
      "name": "Ollama Local",
      "api": "http://localhost:11434/v1",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "qwen3-coder": {
          "id": "qwen3:32b",
          "limit": { "context": 131072, "output": 8192 },
          "tool_call": true
        }
      }
    }
  },

  "agent": {
    "sovereign": {
      "description": "Sovereign Engine trading specialist",
      "model": "anthropic/claude-sonnet-4-5-20250514",
      "prompt": "You are a quantitative trading specialist...",
      "tools": {
        "formula": true,
        "backtest": true,
        "signal": true
      }
    }
  },

  "tools": {
    "formula": true,
    "backtest": true,
    "signal": true,
    "vectorsearch": true
  },

  "experimental": {
    "batch_tool": true
  }
}
```

---

## Implementation Roadmap

### Week 1: Core Tools
- [ ] Create `formula.ts` tool
- [ ] Create `backtest.ts` tool
- [ ] Create `signal.ts` tool
- [ ] Register in `registry.ts`
- [ ] Test tool execution

### Week 2: Custom Agent
- [ ] Add "sovereign" agent to `agent.ts`
- [ ] Create system prompt
- [ ] Configure tool permissions
- [ ] Test agent delegation

### Week 3: Local Model Integration
- [ ] Configure Ollama provider
- [ ] Test with Qwen3/DeepSeek/Codestral
- [ ] Benchmark speed vs Claude
- [ ] Fine-tune prompts for local model

### Week 4: Vector Search
- [ ] Set up ChromaDB
- [ ] Index codebase
- [ ] Create `vectorsearch.ts` tool
- [ ] Test semantic search

### Week 5: Optimization
- [ ] Add response caching
- [ ] Optimize streaming
- [ ] Parallel tool execution
- [ ] Final benchmarks

---

## Expected Outcomes

| Metric | Before (Claude Code) | After (Custom OpenCode) |
|--------|---------------------|------------------------|
| Context Window | 200K tokens | 1M+ effective (vector search) |
| Formula Lookup | Manual grep | Instant tool call |
| Backtest Execution | Manual CLI | Single tool call |
| Model Flexibility | Claude only | 75+ providers + local |
| Cost (monthly) | ~$500+ | $0 with local models |
| Latency | ~2-5s | <1s with local models |

---

## Key Files to Modify

| File | Action | Priority |
|------|--------|----------|
| `tool/formula.ts` | CREATE | P0 |
| `tool/backtest.ts` | CREATE | P0 |
| `tool/signal.ts` | CREATE | P0 |
| `tool/registry.ts` | MODIFY | P0 |
| `agent/agent.ts` | MODIFY | P1 |
| `tool/vectorsearch.ts` | CREATE | P1 |
| `opencode.json` | CREATE | P1 |
| `session/sovereign-system.txt` | CREATE | P2 |

---

## Sources

- [OpenCode GitHub](https://github.com/sst/opencode)
- [OpenCode Docs - Models](https://opencode.ai/docs/models/)
- [OpenCode Docs - Agents](https://opencode.ai/docs/agents/)
- [AI SDK Providers](https://sdk.vercel.ai/providers)
