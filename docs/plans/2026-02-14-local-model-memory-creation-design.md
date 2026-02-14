# Local Model Memory Creation

**Date:** 2026-02-14
**Status:** Approved
**Branch:** `feat/embedding-experiment`

## Goal

Replace the Claude SDK agent and ChromaDB with in-process ONNX models via Transformers.js. Zero external dependencies — everything runs inside the Bun worker process.

**Motivations:** Reduce cost (no API calls), reduce latency (in-process inference), enable offline support (no internet required after model download).

## Architecture

```
Tool execution (from hooks)
        ↓
  Simplified prompt builder
        ↓
  Generative model (Qwen3-0.6B, ONNX/WASM via Transformers.js)
        ↓
  Tool call: create_observation({type, title, narrative, ...})
        ↓
  Parse tool call JSON → ParsedObservation
        ↓
  SQLite (observations table + embedding BLOB column)
        ↑
  Embedding model (all-MiniLM-L6-v2, ONNX/WASM via Transformers.js)
```

## Key Design Decisions

### Tool Calling Over Field Extraction

Rather than having the model produce free-form text and extracting structure with heuristics, we define a `create_observation` tool with a JSON Schema. Qwen3 is specifically trained for tool calling — it reasons about the input (thinking mode), then calls the tool with structured arguments.

If the model determines a tool execution is trivial, it simply doesn't call the tool (no "skip" handling needed).

File paths (`filesRead`, `filesModified`) are extracted deterministically from the tool input — no model involvement for data we already have.

### Prompt Reuse

The observation quality guidelines (what to record, what to skip, good vs bad examples) from the current Claude SDK system prompt are reused. The XML output format instructions are removed since tool calling handles structuring. Prompts are simplified for small model consumption.

### Two Models

| Role | Default Model | Size (Q4) | Purpose |
|------|--------------|-----------|---------|
| Generative | `onnx-community/Qwen3-0.6B-ONNX` | ~350MB | Semantic analysis + tool calling |
| Embedding | `Xenova/all-MiniLM-L6-v2` | ~23MB | Dedup, retrieval, similarity |

Both run via Transformers.js in the same Bun worker process. Models are auto-downloaded on first use and cached in `~/.claude-mem/models/`.

### ChromaDB Replacement

Embeddings are stored directly in SQLite as a BLOB column on the observations table. Context retrieval uses cosine similarity computed in code, replacing the external ChromaDB vector database.

## Components

### 1. Model Manager (`src/models/manager.ts`)

Handles model lifecycle — download, cache, and load.

- `loadGenerativeModel(modelId, dtype)` → Transformers.js `pipeline("text-generation", ...)`
- `loadEmbeddingModel(modelId)` → Transformers.js `pipeline("feature-extraction", ...)`
- Models loaded lazily on first inference call, kept in memory for worker lifetime
- Cache directory: `~/.claude-mem/models/` (configurable via `CLAUDE_MEM_MODEL_DIR`)

### 2. Observation Tool Schema

Single tool definition passed to the generative model:

```typescript
{
  name: "create_observation",
  parameters: {
    type: { enum: ["bugfix", "feature", "refactor", "change", "discovery", "decision"] },
    title: { type: "string" },          // ~80 chars
    subtitle: { type: "string" },        // max 24 words
    narrative: { type: "string" },       // full context
    facts: { type: "array<string>" },    // concise factual statements
    concepts: { type: "array", enum: ["how-it-works", "why-it-exists", ...] }
  },
  required: ["type", "title", "narrative"]
}
```

### 3. Local Agent (`src/worker/local-agent.ts`)

Replaces `sdk-agent.ts`. Implements the same `SDKAgent` interface (`{ processMessages }`).

- Receives `PendingInputMessage` from BackgroundProcessor (unchanged)
- Builds simplified prompt with tool definition
- Uses `apply_chat_template` with `tools` option for proper Qwen3 tool-calling format
- Calls generative model
- Parses `<tool_call>` JSON from output
- Merges deterministic file paths from tool input
- Computes embedding via embedding model
- Stores observation + embedding in SQLite
- Yields same `SDKAgentMessage` types (unchanged interface)

### 4. Prompt Builder (`src/models/prompts.ts`)

Simplified prompts for small model consumption. Reuses observation guidelines, removes format instructions.

Per-tool prompt:
```
Tool: Edit
File: src/auth/token.ts
Change: `const token = getToken()` → `const token = await getToken()`
Result: Applied successfully
```

System prompt retains: observer role, what to record, what to skip, quality examples.

### 5. Embedding Store (SQLite migration)

```sql
ALTER TABLE observations ADD COLUMN embedding BLOB;
```

Context retrieval (`handleGetContext`) uses cosine similarity on embeddings for relevance scoring, replacing FTS5 + heuristic scoring.

### 6. Tool Call Parser (`src/models/tool-call-parser.ts`)

Parses Qwen3's tool call format:
```
<tool_call>
{"name": "create_observation", "arguments": {...}}
</tool_call>
```

Simple: find `<tool_call>` tag, `JSON.parse` the content, validate against schema.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLAUDE_MEM_GEN_MODEL` | `onnx-community/Qwen3-0.6B-ONNX` | Generative model HF ID |
| `CLAUDE_MEM_EMBED_MODEL` | `Xenova/all-MiniLM-L6-v2` | Embedding model HF ID |
| `CLAUDE_MEM_GEN_DTYPE` | `q4` | Generative model quantization |
| `CLAUDE_MEM_MODEL_DIR` | `~/.claude-mem/models` | Local model cache directory |

Users can swap models (e.g., `Qwen2.5-0.5B-Instruct`, `Qwen2.5-Coder-0.5B-Instruct`, `Qwen3-Embedding-0.6B`) via env vars.

## What Changes

| Component | Change |
|-----------|--------|
| `src/worker/sdk-agent.ts` | Replaced by `src/worker/local-agent.ts` |
| `src/sdk/prompts.ts` | Replaced by `src/models/prompts.ts` |
| `src/sdk/parser.ts` | Replaced by `src/models/tool-call-parser.ts` |
| `src/services/chroma-sync.ts` | Removed entirely |
| `src/models/manager.ts` | New — model lifecycle |
| `src/db/migrations.ts` | New migration — embedding BLOB column |
| `src/worker/handlers.ts` | `handleGetContext` uses embedding similarity |
| `package.json` | Add `@huggingface/transformers`, remove `@anthropic-ai/claude-agent-sdk` |

## What Stays the Same

- `SDKAgent` interface (`{ processMessages }`)
- `BackgroundProcessor` polling and dispatch
- `SessionManager` queues and TTL eviction
- All hooks (fire-and-forget HTTP to worker)
- `ParsedObservation` and `ParsedSummary` domain types
- Overall data flow: hooks → worker → agent → SQLite → context injection
- CLI subcommands

## Dependencies

| Package | Action | Purpose |
|---------|--------|---------|
| `@huggingface/transformers` | Add | ONNX/WASM inference runtime |
| `@anthropic-ai/claude-agent-sdk` | Remove | No longer needed |

Models (~373MB total) are downloaded at runtime on first use, not bundled.

## Migration

Existing observations in SQLite are preserved. The `embedding` column is nullable — old observations without embeddings fall back to FTS5 matching. New observations get embeddings on creation. A backfill command could be added later to compute embeddings for existing data.
