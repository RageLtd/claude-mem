# Local Model Memory Creation - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Claude SDK agent and ChromaDB with in-process ONNX models (Qwen3-0.6B + all-MiniLM-L6-v2) via Transformers.js, using tool calling for structured observation extraction.

**Architecture:** Two ONNX models run in-process via Transformers.js WASM. The generative model (Qwen3-0.6B) analyzes tool executions and calls a `create_observation` tool with structured arguments. The embedding model (all-MiniLM-L6-v2) computes vectors stored in SQLite, replacing ChromaDB for similarity search.

**Tech Stack:** `@huggingface/transformers` (ONNX/WASM), Bun, SQLite, Qwen3-0.6B-ONNX, all-MiniLM-L6-v2

**Design doc:** `docs/plans/2026-02-14-local-model-memory-creation-design.md`

---

### Task 1: Add Transformers.js dependency and remove Claude SDK

**Files:**
- Modify: `package.json`

**Step 1: Install @huggingface/transformers**

Run: `bun add @huggingface/transformers`

**Step 2: Remove @anthropic-ai/claude-agent-sdk**

Run: `bun remove @anthropic-ai/claude-agent-sdk`

**Step 3: Verify installation**

Run: `bun install && bun test tests/unit/database.test.ts`
Expected: PASS (database tests don't depend on SDK)

**Step 4: Commit**

```bash
git add package.json bun.lock
git commit -m "feat: swap claude-agent-sdk for @huggingface/transformers"
```

---

### Task 2: Add embedding BLOB column to observations table

**Files:**
- Modify: `src/db/migrations.ts`
- Test: `tests/unit/database.test.ts`

**Step 1: Write the failing test**

Add a test to `tests/unit/database.test.ts` that verifies the `embedding` column exists on observations:

```typescript
it("stores and retrieves embedding blob on observations", () => {
  // Create session first
  createSession(db, {
    claudeSessionId: "embed-test",
    project: "test",
    userPrompt: "test",
  });

  // Store observation
  const obsResult = storeObservation(db, {
    claudeSessionId: "embed-test",
    project: "test",
    observation: {
      type: "discovery",
      title: "Test embedding",
      subtitle: null,
      narrative: "test",
      facts: [],
      concepts: [],
      filesRead: [],
      filesModified: [],
    },
    promptNumber: 1,
  });
  expect(obsResult.ok).toBe(true);

  // Verify we can store an embedding for this observation
  const embedding = new Float32Array([0.1, 0.2, 0.3]);
  const buffer = Buffer.from(embedding.buffer);
  db.run("UPDATE observations SET embedding = ? WHERE id = ?", [
    buffer,
    obsResult.ok ? obsResult.value : -1,
  ]);

  // Retrieve and verify
  const row = db
    .query<{ embedding: Buffer | null }, [number]>(
      "SELECT embedding FROM observations WHERE id = ?"
    )
    .get(obsResult.ok ? obsResult.value : -1);

  expect(row).not.toBeNull();
  expect(row!.embedding).not.toBeNull();
  const retrieved = new Float32Array(
    row!.embedding!.buffer,
    row!.embedding!.byteOffset,
    row!.embedding!.byteLength / 4
  );
  expect(retrieved[0]).toBeCloseTo(0.1);
  expect(retrieved[1]).toBeCloseTo(0.2);
  expect(retrieved[2]).toBeCloseTo(0.3);
});
```

**Step 2: Run test to verify it fails**

Run: `bun test tests/unit/database.test.ts -t "stores and retrieves embedding"`
Expected: FAIL — `embedding` column does not exist

**Step 3: Add migration**

In `src/db/migrations.ts`, add migration version 6 to the `migrations` array:

```typescript
{
  version: 6,
  description: "Add embedding column to observations",
  up: (db) => {
    db.run("ALTER TABLE observations ADD COLUMN embedding BLOB");
  },
},
```

**Step 4: Run test to verify it passes**

Run: `bun test tests/unit/database.test.ts -t "stores and retrieves embedding"`
Expected: PASS

**Step 5: Run full database test suite**

Run: `bun test tests/unit/database.test.ts`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/db/migrations.ts tests/unit/database.test.ts
git commit -m "feat: add embedding BLOB column to observations table"
```

---

### Task 3: Create model manager

**Files:**
- Create: `src/models/manager.ts`
- Test: `tests/unit/model-manager.test.ts`

**Step 1: Write the failing test**

Create `tests/unit/model-manager.test.ts`:

```typescript
import { describe, expect, it, mock } from "bun:test";
import { createModelManager, type ModelManager } from "../../src/models/manager";

describe("ModelManager", () => {
  it("creates with default configuration", () => {
    const manager = createModelManager({});
    expect(manager).toBeDefined();
    expect(manager.getConfig().generativeModelId).toBe(
      "onnx-community/Qwen3-0.6B-ONNX"
    );
    expect(manager.getConfig().embeddingModelId).toBe(
      "Xenova/all-MiniLM-L6-v2"
    );
  });

  it("accepts custom model IDs via config", () => {
    const manager = createModelManager({
      generativeModelId: "onnx-community/Qwen2.5-0.5B-Instruct",
      embeddingModelId: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
    });
    expect(manager.getConfig().generativeModelId).toBe(
      "onnx-community/Qwen2.5-0.5B-Instruct"
    );
    expect(manager.getConfig().embeddingModelId).toBe(
      "onnx-community/Qwen3-Embedding-0.6B-ONNX"
    );
  });

  it("accepts dtype configuration", () => {
    const manager = createModelManager({ dtype: "q4f16" });
    expect(manager.getConfig().dtype).toBe("q4f16");
  });
});
```

**Step 2: Run test to verify it fails**

Run: `bun test tests/unit/model-manager.test.ts`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `src/models/manager.ts`:

```typescript
/**
 * Model lifecycle manager for local ONNX inference.
 * Handles model download, caching, and lazy loading via Transformers.js.
 */

import { join } from "node:path";

// ============================================================================
// Types
// ============================================================================

export interface ModelManagerConfig {
  readonly generativeModelId: string;
  readonly embeddingModelId: string;
  readonly dtype: string;
  readonly cacheDir: string;
}

export interface ModelManagerDeps {
  readonly generativeModelId?: string;
  readonly embeddingModelId?: string;
  readonly dtype?: string;
  readonly cacheDir?: string;
}

export interface ModelManager {
  readonly getConfig: () => ModelManagerConfig;
  readonly generateText: (
    messages: readonly ChatMessage[],
    tools?: readonly ToolDefinition[],
  ) => Promise<string>;
  readonly computeEmbedding: (text: string) => Promise<Float32Array>;
  readonly dispose: () => Promise<void>;
}

export interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

export interface ToolDefinition {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly description: string;
    readonly parameters: Record<string, unknown>;
  };
}

// ============================================================================
// Defaults
// ============================================================================

const DEFAULT_GEN_MODEL = "onnx-community/Qwen3-0.6B-ONNX";
const DEFAULT_EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const DEFAULT_DTYPE = "q4";
const DEFAULT_CACHE_DIR = join(
  process.env.HOME || "",
  ".claude-mem",
  "models",
);

// ============================================================================
// Factory
// ============================================================================

export const createModelManager = (deps: ModelManagerDeps): ModelManager => {
  const config: ModelManagerConfig = {
    generativeModelId:
      deps.generativeModelId ||
      process.env.CLAUDE_MEM_GEN_MODEL ||
      DEFAULT_GEN_MODEL,
    embeddingModelId:
      deps.embeddingModelId ||
      process.env.CLAUDE_MEM_EMBED_MODEL ||
      DEFAULT_EMBED_MODEL,
    dtype: deps.dtype || process.env.CLAUDE_MEM_GEN_DTYPE || DEFAULT_DTYPE,
    cacheDir:
      deps.cacheDir || process.env.CLAUDE_MEM_MODEL_DIR || DEFAULT_CACHE_DIR,
  };

  // Lazy-loaded pipeline references
  let generativePipeline: unknown = null;
  let embeddingPipeline: unknown = null;

  const getGenerativePipeline = async () => {
    if (!generativePipeline) {
      const { pipeline } = await import("@huggingface/transformers");
      generativePipeline = await pipeline("text-generation", config.generativeModelId, {
        dtype: config.dtype,
        cache_dir: config.cacheDir,
      });
    }
    return generativePipeline;
  };

  const getEmbeddingPipeline = async () => {
    if (!embeddingPipeline) {
      const { pipeline } = await import("@huggingface/transformers");
      embeddingPipeline = await pipeline(
        "feature-extraction",
        config.embeddingModelId,
        { cache_dir: config.cacheDir },
      );
    }
    return embeddingPipeline;
  };

  const generateText = async (
    messages: readonly ChatMessage[],
    tools?: readonly ToolDefinition[],
  ): Promise<string> => {
    const gen = await getGenerativePipeline() as any;
    const output = await gen(messages as any, {
      max_new_tokens: 512,
      temperature: 0.1,
      do_sample: true,
      tools: tools as any,
    });
    // Extract generated text from pipeline output
    const generated = output[0]?.generated_text;
    if (Array.isArray(generated)) {
      // Chat format: array of messages — get the last assistant message
      const last = generated[generated.length - 1];
      return typeof last === "string" ? last : last?.content ?? "";
    }
    return typeof generated === "string" ? generated : "";
  };

  const computeEmbedding = async (text: string): Promise<Float32Array> => {
    const embed = await getEmbeddingPipeline() as any;
    const output = await embed(text, { pooling: "mean", normalize: true });
    return new Float32Array(output.data);
  };

  const dispose = async (): Promise<void> => {
    if (generativePipeline && typeof (generativePipeline as any).dispose === "function") {
      await (generativePipeline as any).dispose();
    }
    if (embeddingPipeline && typeof (embeddingPipeline as any).dispose === "function") {
      await (embeddingPipeline as any).dispose();
    }
    generativePipeline = null;
    embeddingPipeline = null;
  };

  return {
    getConfig: () => config,
    generateText,
    computeEmbedding,
    dispose,
  };
};
```

**Step 4: Run test to verify it passes**

Run: `bun test tests/unit/model-manager.test.ts`
Expected: PASS (config tests don't require actual model downloads)

**Step 5: Commit**

```bash
git add src/models/manager.ts tests/unit/model-manager.test.ts
git commit -m "feat: add model manager for local ONNX inference"
```

---

### Task 4: Create tool call parser

**Files:**
- Create: `src/models/tool-call-parser.ts`
- Test: `tests/unit/tool-call-parser.test.ts`

**Step 1: Write the failing tests**

Create `tests/unit/tool-call-parser.test.ts`:

```typescript
import { describe, expect, it } from "bun:test";
import {
  parseToolCall,
  type ToolCallResult,
} from "../../src/models/tool-call-parser";

describe("parseToolCall", () => {
  it("parses a valid tool call with all fields", () => {
    const input = `Let me analyze this.
<tool_call>
{"name": "create_observation", "arguments": {"type": "bugfix", "title": "Fixed async token bug", "subtitle": "Added missing await", "narrative": "The getToken call was missing await", "facts": ["getToken is async"], "concepts": ["problem-solution"]}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    expect(result!.name).toBe("create_observation");
    expect(result!.arguments.type).toBe("bugfix");
    expect(result!.arguments.title).toBe("Fixed async token bug");
    expect(result!.arguments.narrative).toBe(
      "The getToken call was missing await"
    );
    expect(result!.arguments.facts).toEqual(["getToken is async"]);
    expect(result!.arguments.concepts).toEqual(["problem-solution"]);
  });

  it("parses minimal required fields", () => {
    const input = `<tool_call>
{"name": "create_observation", "arguments": {"type": "discovery", "title": "Found config pattern", "narrative": "The config uses a factory function"}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    expect(result!.arguments.type).toBe("discovery");
    expect(result!.arguments.subtitle).toBeUndefined();
    expect(result!.arguments.facts).toBeUndefined();
  });

  it("returns null when no tool call present (trivial skip)", () => {
    const input = "This tool execution is routine and does not need recording.";
    const result = parseToolCall(input);
    expect(result).toBeNull();
  });

  it("returns null for malformed JSON inside tool_call tags", () => {
    const input = `<tool_call>
{not valid json}
</tool_call>`;
    const result = parseToolCall(input);
    expect(result).toBeNull();
  });

  it("handles thinking tags before tool call", () => {
    const input = `<think>
This is a significant bug fix that should be recorded.
</think>
<tool_call>
{"name": "create_observation", "arguments": {"type": "bugfix", "title": "Fixed race condition", "narrative": "Concurrent requests caused data corruption"}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    expect(result!.arguments.type).toBe("bugfix");
  });

  it("validates observation type is a known enum value", () => {
    const input = `<tool_call>
{"name": "create_observation", "arguments": {"type": "invalid_type", "title": "Test", "narrative": "Test"}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    // Should coerce to "change" for unknown types
    expect(result!.arguments.type).toBe("change");
  });
});
```

**Step 2: Run test to verify it fails**

Run: `bun test tests/unit/tool-call-parser.test.ts`
Expected: FAIL — module not found

**Step 3: Write implementation**

Create `src/models/tool-call-parser.ts`:

```typescript
/**
 * Parser for Qwen3 tool call output.
 * Extracts structured observation data from <tool_call> blocks.
 */

import { type ObservationType, isObservationType } from "../types/domain";

// ============================================================================
// Types
// ============================================================================

export interface ToolCallArguments {
  readonly type: ObservationType;
  readonly title: string;
  readonly subtitle?: string;
  readonly narrative: string;
  readonly facts?: readonly string[];
  readonly concepts?: readonly string[];
}

export interface ToolCallResult {
  readonly name: string;
  readonly arguments: ToolCallArguments;
}

// ============================================================================
// Parser
// ============================================================================

/**
 * Parses a tool call from model output.
 * Returns null if no tool call is present (model decided to skip)
 * or if the tool call is malformed.
 */
export const parseToolCall = (text: string): ToolCallResult | null => {
  const match = text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/);
  if (!match) return null;

  const jsonStr = match[1].trim();

  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonStr);
  } catch {
    return null;
  }

  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !("name" in parsed) ||
    !("arguments" in parsed)
  ) {
    return null;
  }

  const { name, arguments: args } = parsed as {
    name: unknown;
    arguments: unknown;
  };

  if (typeof name !== "string" || typeof args !== "object" || args === null) {
    return null;
  }

  const rawArgs = args as Record<string, unknown>;

  // Validate required fields
  if (typeof rawArgs.title !== "string" || typeof rawArgs.narrative !== "string") {
    return null;
  }

  // Coerce type to valid ObservationType
  const rawType = typeof rawArgs.type === "string" ? rawArgs.type : "change";
  const type: ObservationType = isObservationType(rawType) ? rawType : "change";

  return {
    name: String(name),
    arguments: {
      type,
      title: rawArgs.title,
      subtitle: typeof rawArgs.subtitle === "string" ? rawArgs.subtitle : undefined,
      narrative: rawArgs.narrative,
      facts: Array.isArray(rawArgs.facts)
        ? rawArgs.facts.filter((f): f is string => typeof f === "string")
        : undefined,
      concepts: Array.isArray(rawArgs.concepts)
        ? rawArgs.concepts.filter((c): c is string => typeof c === "string")
        : undefined,
    },
  };
};
```

**Step 4: Run test to verify it passes**

Run: `bun test tests/unit/tool-call-parser.test.ts`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/models/tool-call-parser.ts tests/unit/tool-call-parser.test.ts
git commit -m "feat: add tool call parser for Qwen3 structured output"
```

---

### Task 5: Create local model prompts

**Files:**
- Create: `src/models/prompts.ts`
- Test: `tests/unit/local-prompts.test.ts`

**Step 1: Write the failing tests**

Create `tests/unit/local-prompts.test.ts`:

```typescript
import { describe, expect, it } from "bun:test";
import {
  buildLocalSystemPrompt,
  buildLocalObservationPrompt,
  buildLocalSummaryPrompt,
  OBSERVATION_TOOL,
} from "../../src/models/prompts";

describe("local model prompts", () => {
  it("builds a system prompt with observer guidelines", () => {
    const prompt = buildLocalSystemPrompt();
    expect(prompt).toContain("observer");
    expect(prompt).toContain("bugfix");
    expect(prompt).toContain("discovery");
    // Should NOT contain XML format instructions
    expect(prompt).not.toContain("<observation>");
    expect(prompt).not.toContain("</observation>");
  });

  it("builds observation prompt from tool execution", () => {
    const prompt = buildLocalObservationPrompt({
      toolName: "Edit",
      toolInput: { file_path: "src/auth.ts", old_string: "foo", new_string: "bar" },
      toolResponse: "Applied edit",
      cwd: "/projects/app",
      occurredAt: "2026-02-14T12:00:00Z",
    });
    expect(prompt).toContain("Edit");
    expect(prompt).toContain("src/auth.ts");
    expect(prompt).toContain("Applied edit");
  });

  it("builds summary prompt", () => {
    const prompt = buildLocalSummaryPrompt({
      lastUserMessage: "Fix the auth bug",
      lastAssistantMessage: "I fixed it",
    });
    expect(prompt).toContain("Fix the auth bug");
    expect(prompt).toContain("summary");
  });

  it("exports observation tool definition with correct schema", () => {
    expect(OBSERVATION_TOOL.type).toBe("function");
    expect(OBSERVATION_TOOL.function.name).toBe("create_observation");
    const params = OBSERVATION_TOOL.function.parameters as any;
    expect(params.properties.type.enum).toContain("bugfix");
    expect(params.properties.type.enum).toContain("discovery");
    expect(params.required).toContain("type");
    expect(params.required).toContain("title");
    expect(params.required).toContain("narrative");
  });
});
```

**Step 2: Run test to verify it fails**

Run: `bun test tests/unit/local-prompts.test.ts`
Expected: FAIL — module not found

**Step 3: Write implementation**

Create `src/models/prompts.ts`:

```typescript
/**
 * Prompts for local model inference.
 * Reuses observation quality guidelines from the SDK agent,
 * adapted for small model consumption with tool calling.
 */

import type { ToolObservation } from "../types/domain";
import type { ToolDefinition } from "./manager";

// ============================================================================
// Tool Definition
// ============================================================================

export const OBSERVATION_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: "create_observation",
    description:
      "Record a meaningful observation from a tool execution. Only call this for non-trivial work.",
    parameters: {
      type: "object",
      properties: {
        type: {
          type: "string",
          enum: [
            "bugfix",
            "feature",
            "refactor",
            "change",
            "discovery",
            "decision",
          ],
          description:
            "bugfix: something broken now fixed. feature: new capability. refactor: restructured, behavior unchanged. change: generic modification. discovery: learning about existing system. decision: architectural choice.",
        },
        title: {
          type: "string",
          description: "Short title capturing the core action (~80 chars)",
        },
        subtitle: {
          type: "string",
          description: "One sentence explanation (max 24 words)",
        },
        narrative: {
          type: "string",
          description:
            "Full context: what was done, how it works, why it matters",
        },
        facts: {
          type: "array",
          items: { type: "string" },
          description: "Concise, self-contained factual statements",
        },
        concepts: {
          type: "array",
          items: {
            type: "string",
            enum: [
              "how-it-works",
              "why-it-exists",
              "what-changed",
              "problem-solution",
              "gotcha",
              "pattern",
              "trade-off",
            ],
          },
          description: "Concept tags categorizing this observation",
        },
      },
      required: ["type", "title", "narrative"],
    },
  },
};

// ============================================================================
// System Prompt
// ============================================================================

export const buildLocalSystemPrompt = (): string => {
  return `You are an observer that records what happens during a developer session.

When you receive a tool execution notification, decide if it represents meaningful work. If yes, call the create_observation tool. If the operation is trivial (empty file checks, basic listings, simple installs), do NOT call the tool.

Record OUTCOMES and INSIGHTS, not just actions:
- Bug investigations: root cause, what was found
- Discoveries: how code works, why something behaves a certain way
- Fixes: what was broken and how it was fixed
- Features: new functionality added
- Decisions: architectural choices, trade-offs

Use past tense: discovered, fixed, implemented, learned.

Good: "Fixed missing await on getToken() causing auth failures downstream"
Bad: "Analyzed the code and recorded findings"`;
};

// ============================================================================
// Per-message Prompts
// ============================================================================

export const buildLocalObservationPrompt = (
  observation: ToolObservation,
): string => {
  const { toolName, toolInput, toolResponse } = observation;

  // Extract key details based on tool type
  const inputSummary =
    typeof toolInput === "object" && toolInput !== null
      ? JSON.stringify(toolInput, null, 2).slice(0, 1000)
      : String(toolInput).slice(0, 1000);

  const responseSummary =
    typeof toolResponse === "string"
      ? toolResponse.slice(0, 500)
      : JSON.stringify(toolResponse, null, 2).slice(0, 500);

  return `Tool: ${toolName}
Input: ${inputSummary}
Result: ${responseSummary}`;
};

export interface SummaryPromptInput {
  readonly lastUserMessage: string;
  readonly lastAssistantMessage?: string;
}

export const buildLocalSummaryPrompt = (input: SummaryPromptInput): string => {
  return `Generate a progress summary of what was accomplished.

User request: ${input.lastUserMessage}
${input.lastAssistantMessage ? `Assistant response: ${input.lastAssistantMessage}` : ""}

Provide a brief summary covering: what was requested, what was completed, what was learned, and suggested next steps.`;
};
```

**Step 4: Run test to verify it passes**

Run: `bun test tests/unit/local-prompts.test.ts`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/models/prompts.ts tests/unit/local-prompts.test.ts
git commit -m "feat: add local model prompts with tool calling schema"
```

---

### Task 6: Create local agent (replaces sdk-agent)

**Files:**
- Create: `src/worker/local-agent.ts`
- Test: `tests/unit/local-agent.test.ts`

This is the core component. It implements the same `SDKAgent` interface as `sdk-agent.ts` but uses local models via the ModelManager.

**Step 1: Write the failing tests**

Create `tests/unit/local-agent.test.ts`. The tests should mock the ModelManager to avoid downloading actual models:

```typescript
import { describe, expect, it, mock, beforeEach } from "bun:test";
import { Database } from "bun:sqlite";
import { createDatabase, runMigrations, createSession } from "../../src/db/index";
import { createLocalAgent, type LocalAgentDeps } from "../../src/worker/local-agent";
import type { ActiveSession } from "../../src/worker/session-manager";
import type { PendingInputMessage, SDKAgentMessage } from "../../src/worker/sdk-agent";
import type { ModelManager } from "../../src/models/manager";

const createMockModelManager = (
  generateResponse = `<tool_call>
{"name": "create_observation", "arguments": {"type": "bugfix", "title": "Fixed auth bug", "narrative": "The authentication was broken due to missing await"}}
</tool_call>`,
): ModelManager => ({
  getConfig: () => ({
    generativeModelId: "test-model",
    embeddingModelId: "test-embed",
    dtype: "q4",
    cacheDir: "/tmp/test-models",
  }),
  generateText: mock(async () => generateResponse),
  computeEmbedding: mock(async () => new Float32Array([0.1, 0.2, 0.3])),
  dispose: mock(async () => {}),
});

const createTestSession = (): ActiveSession => ({
  sessionDbId: 1,
  claudeSessionId: "test-session-123",
  project: "test-project",
  userPrompt: "Fix the auth bug",
  abortController: new AbortController(),
});

async function* createInputMessages(
  messages: PendingInputMessage[],
): AsyncIterable<PendingInputMessage> {
  for (const msg of messages) {
    yield msg;
  }
}

const collectMessages = async (
  iterable: AsyncIterable<SDKAgentMessage>,
): Promise<SDKAgentMessage[]> => {
  const results: SDKAgentMessage[] = [];
  for await (const msg of iterable) {
    results.push(msg);
  }
  return results;
};

describe("LocalAgent", () => {
  let db: Database;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);
    createSession(db, {
      claudeSessionId: "test-session-123",
      project: "test-project",
      userPrompt: "Fix the auth bug",
    });
  });

  it("processes an observation and stores it", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const input = createInputMessages([
      {
        type: "observation",
        data: {
          observation: {
            toolName: "Edit",
            toolInput: { file_path: "src/auth.ts" },
            toolResponse: "Applied edit",
            cwd: "/projects/app",
            occurredAt: "2026-02-14T12:00:00Z",
          },
        },
      },
    ]);

    const messages = await collectMessages(
      agent.processMessages(session, input),
    );

    expect(messages.length).toBeGreaterThanOrEqual(1);
    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();
  });

  it("yields acknowledged when model skips trivial execution", async () => {
    const modelManager = createMockModelManager(
      "This is a routine file listing, nothing notable.",
    );
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const input = createInputMessages([
      {
        type: "observation",
        data: {
          observation: {
            toolName: "LS",
            toolInput: { path: "." },
            toolResponse: "file1.ts\nfile2.ts",
            cwd: "/projects/app",
            occurredAt: "2026-02-14T12:00:00Z",
          },
        },
      },
    ]);

    const messages = await collectMessages(
      agent.processMessages(session, input),
    );

    const ack = messages.find((m) => m.type === "acknowledged");
    expect(ack).toBeDefined();
  });

  it("extracts file paths from tool input for Edit tool", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const input = createInputMessages([
      {
        type: "observation",
        data: {
          observation: {
            toolName: "Edit",
            toolInput: { file_path: "src/auth.ts", old_string: "a", new_string: "b" },
            toolResponse: "Applied edit to src/auth.ts",
            cwd: "/projects/app",
            occurredAt: "2026-02-14T12:00:00Z",
          },
        },
      },
    ]);

    const messages = await collectMessages(
      agent.processMessages(session, input),
    );

    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();
    const data = stored!.data as { observation: { filesModified: string[] } };
    expect(data.observation.filesModified).toContain("src/auth.ts");
  });

  it("yields aborted when session is already aborted", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();
    session.abortController.abort();

    const input = createInputMessages([]);
    const messages = await collectMessages(
      agent.processMessages(session, input),
    );

    expect(messages).toEqual([{ type: "aborted" }]);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `bun test tests/unit/local-agent.test.ts`
Expected: FAIL — module not found

**Step 3: Write implementation**

Create `src/worker/local-agent.ts`:

```typescript
/**
 * LocalAgent - Local model processing for observations.
 * Uses Transformers.js ONNX models instead of Claude SDK.
 * Implements the same SDKAgent interface for drop-in replacement.
 */

import type { Database } from "bun:sqlite";
import {
  findSimilarObservation,
  storeObservation,
  storeSummary,
} from "../db/index";
import type { ModelManager, ChatMessage } from "../models/manager";
import {
  buildLocalSystemPrompt,
  buildLocalObservationPrompt,
  buildLocalSummaryPrompt,
  OBSERVATION_TOOL,
} from "../models/prompts";
import { parseToolCall } from "../models/tool-call-parser";
import type { ToolObservation } from "../types/domain";
import type {
  PendingInputMessage,
  SDKAgent,
  SDKAgentMessage,
} from "./sdk-agent";
import type { ActiveSession } from "./session-manager";

// ============================================================================
// Types
// ============================================================================

export interface LocalAgentDeps {
  readonly db: Database;
  readonly modelManager: ModelManager;
}

// ============================================================================
// File Path Extraction
// ============================================================================

/**
 * Extracts file paths from tool input deterministically.
 * No model involvement — uses tool name to determine read vs modified.
 */
const extractFilePaths = (
  toolName: string,
  toolInput: unknown,
): { readonly filesRead: readonly string[]; readonly filesModified: readonly string[] } => {
  const input = toolInput as Record<string, unknown> | null;
  if (!input) return { filesRead: [], filesModified: [] };

  const filePath =
    typeof input.file_path === "string" ? input.file_path : null;
  const path = typeof input.path === "string" ? input.path : null;
  const resolved = filePath || path;

  if (!resolved) return { filesRead: [], filesModified: [] };

  const readTools = ["Read", "Grep", "Glob", "LS"];
  const writeTools = ["Edit", "MultiEdit", "Write", "NotebookEdit"];

  if (readTools.includes(toolName)) {
    return { filesRead: [resolved], filesModified: [] };
  }
  if (writeTools.includes(toolName)) {
    return { filesRead: [], filesModified: [resolved] };
  }

  return { filesRead: [resolved], filesModified: [] };
};

// ============================================================================
// Factory
// ============================================================================

const log = (msg: string) => console.log(`[local-agent] ${msg}`);

export const createLocalAgent = (deps: LocalAgentDeps): SDKAgent => {
  const { db, modelManager } = deps;

  const processMessages = async function* (
    session: ActiveSession,
    inputMessages: AsyncIterable<PendingInputMessage>,
  ): AsyncIterable<SDKAgentMessage> {
    if (session.abortController.signal.aborted) {
      yield { type: "aborted" };
      return;
    }

    let promptNumber = 1;
    const systemPrompt = buildLocalSystemPrompt();

    for await (const msg of inputMessages) {
      if (session.abortController.signal.aborted) {
        yield { type: "aborted" };
        return;
      }

      if (msg.type === "observation" && msg.data.observation) {
        const observation = msg.data.observation;
        log(`Processing observation for tool=${observation.toolName}`);

        const messages: ChatMessage[] = [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: buildLocalObservationPrompt(observation),
          },
        ];

        const response = await modelManager.generateText(
          messages,
          [OBSERVATION_TOOL],
        );

        const toolCall = parseToolCall(response);

        if (!toolCall) {
          log("Model skipped (no tool call) — trivial operation");
          yield { type: "acknowledged" };
          continue;
        }

        // Merge deterministic file paths
        const { filesRead, filesModified } = extractFilePaths(
          observation.toolName,
          observation.toolInput,
        );

        const parsedObservation = {
          type: toolCall.arguments.type,
          title: toolCall.arguments.title,
          subtitle: toolCall.arguments.subtitle ?? null,
          narrative: toolCall.arguments.narrative,
          facts: toolCall.arguments.facts ?? [],
          concepts: toolCall.arguments.concepts ?? [],
          filesRead,
          filesModified,
        };

        // Deduplication check
        const dupCheck = findSimilarObservation(db, {
          project: session.project,
          title: parsedObservation.title,
          withinMs: 3600000,
        });

        if (dupCheck.ok && dupCheck.value) {
          log(`Skipping duplicate: "${parsedObservation.title}"`);
          yield { type: "acknowledged" };
          continue;
        }

        // Store observation
        const result = storeObservation(db, {
          claudeSessionId: session.claudeSessionId,
          project: session.project,
          observation: parsedObservation,
          promptNumber,
          discoveryTokens: 0,
        });

        if (result.ok) {
          // Compute and store embedding
          const embeddingText = [
            parsedObservation.title,
            parsedObservation.narrative,
          ]
            .filter(Boolean)
            .join(" ");

          const embedding = await modelManager.computeEmbedding(embeddingText);
          const buffer = Buffer.from(embedding.buffer);
          db.run("UPDATE observations SET embedding = ? WHERE id = ?", [
            buffer,
            result.value,
          ]);

          log(`Observation stored id=${result.value}`);
          yield {
            type: "observation_stored",
            data: { id: result.value, observation: parsedObservation },
          };
        } else {
          log(`Failed to store: ${result.error.message}`);
          yield {
            type: "error",
            data: `Failed to store observation: ${result.error.message}`,
          };
        }
      } else if (msg.type === "summarize") {
        log("Processing summary request");

        const messages: ChatMessage[] = [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: buildLocalSummaryPrompt({
              lastUserMessage: msg.data.lastUserMessage || "",
              lastAssistantMessage: msg.data.lastAssistantMessage,
            }),
          },
        ];

        const response = await modelManager.generateText(messages);

        // For summaries, parse the free-text response into summary fields
        const summary = {
          request: msg.data.lastUserMessage || null,
          investigated: null,
          learned: null,
          completed: response.slice(0, 500) || null,
          nextSteps: null,
          notes: null,
        };

        const result = storeSummary(db, {
          claudeSessionId: session.claudeSessionId,
          project: session.project,
          summary,
          promptNumber,
          discoveryTokens: 0,
        });

        if (result.ok) {
          log(`Summary stored id=${result.value}`);
          yield {
            type: "summary_stored",
            data: { id: result.value, summary },
          };
        } else {
          yield {
            type: "error",
            data: `Failed to store summary: ${result.error.message}`,
          };
        }
      } else if (msg.type === "continuation" && msg.data.userPrompt) {
        promptNumber = msg.data.promptNumber || promptNumber + 1;
        log(`Continuation prompt #${promptNumber}`);
      }
    }
  };

  return { processMessages };
};
```

**Step 4: Run test to verify it passes**

Run: `bun test tests/unit/local-agent.test.ts`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/worker/local-agent.ts tests/unit/local-agent.test.ts
git commit -m "feat: add local agent using Transformers.js for observation extraction"
```

---

### Task 7: Wire local agent into worker main

**Files:**
- Modify: `src/worker/main.ts`

**Step 1: Read current main.ts** (already read above)

**Step 2: Update main.ts to use local agent**

Replace the SDK agent creation in `src/worker/main.ts`:

Change the import from:
```typescript
import { createSDKAgent } from "./sdk-agent";
```
To:
```typescript
import { createLocalAgent } from "./local-agent";
import { createModelManager } from "../models/manager";
```

Change the agent creation from:
```typescript
const sdkAgent = createSDKAgent({ db });
log("SDKAgent initialized");
```
To:
```typescript
const modelManager = createModelManager({});
log(`ModelManager initialized (gen=${modelManager.getConfig().generativeModelId}, embed=${modelManager.getConfig().embeddingModelId})`);

const sdkAgent = createLocalAgent({ db, modelManager });
log("LocalAgent initialized");
```

Add model cleanup to shutdown handler, before `db.close()`:
```typescript
await modelManager.dispose();
```

**Step 3: Run existing worker-main tests**

Run: `bun test tests/unit/worker-main.test.ts`
Expected: May need test updates if they mock SDK agent — update mocks to match new imports.

**Step 4: Commit**

```bash
git add src/worker/main.ts
git commit -m "feat: wire local agent into worker service"
```

---

### Task 8: Add cosine similarity to relevance scoring

**Files:**
- Modify: `src/utils/relevance.ts`
- Modify: `src/worker/handlers.ts`
- Test: `tests/unit/relevance.test.ts`

**Step 1: Write the failing test**

Add to `tests/unit/relevance.test.ts`:

```typescript
import { cosineSimilarity } from "../../src/utils/relevance";

describe("cosineSimilarity", () => {
  it("returns 1.0 for identical vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0);
  });

  it("returns 0.0 for orthogonal vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([0, 1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });

  it("returns -1.0 for opposite vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([-1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0);
  });

  it("handles normalized vectors", () => {
    const a = new Float32Array([0.6, 0.8]);
    const b = new Float32Array([0.8, 0.6]);
    // dot product = 0.48 + 0.48 = 0.96
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.96);
  });

  it("returns 0 for zero-length vectors", () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBe(0);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `bun test tests/unit/relevance.test.ts -t "cosineSimilarity"`
Expected: FAIL — function not exported

**Step 3: Add cosineSimilarity to relevance.ts**

Add at the end of `src/utils/relevance.ts`:

```typescript
/**
 * Cosine similarity between two vectors.
 * Returns value in [-1, 1] where 1 = identical direction.
 */
export const cosineSimilarity = (
  a: Float32Array,
  b: Float32Array,
): number => {
  if (a.length !== b.length || a.length === 0) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
};
```

**Step 4: Run test to verify it passes**

Run: `bun test tests/unit/relevance.test.ts`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/utils/relevance.ts tests/unit/relevance.test.ts
git commit -m "feat: add cosine similarity function for embedding-based retrieval"
```

---

### Task 9: Remove ChromaDB sync service

**Files:**
- Delete: `src/services/chroma-sync.ts`
- Delete: `tests/unit/chroma-sync.test.ts`
- Modify: `src/worker/sdk-agent.ts` — remove ChromaDB references (this file is still referenced by background-processor imports; keep it for backward compat or update imports)

**Step 1: Remove ChromaDB references from sdk-agent.ts**

Since `local-agent.ts` replaces `sdk-agent.ts`, and `background-processor.ts` imports types from `sdk-agent.ts`, verify which imports are still needed.

Check that `background-processor.ts` only imports types (`PendingInputMessage`, `SDKAgent`, `SDKAgentMessage`) from `sdk-agent.ts`. These types should be extracted to a shared types file or re-exported from `local-agent.ts`.

**Step 2: Move shared types to a types file**

Create `src/worker/agent-types.ts` with the shared interfaces (`SDKAgent`, `SDKAgentMessage`, `PendingInputMessage`, etc.) extracted from `sdk-agent.ts`. Update imports in `background-processor.ts` and `local-agent.ts` to use this new file.

**Step 3: Delete ChromaDB files**

```bash
rm src/services/chroma-sync.ts tests/unit/chroma-sync.test.ts
```

**Step 4: Run full test suite**

Run: `bun test`
Expected: All PASS (chroma-sync tests removed, other tests unaffected)

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove ChromaDB sync service, extract shared agent types"
```

---

### Task 10: Format, lint, and final validation

**Files:**
- All modified/created files

**Step 1: Run Biome**

Run: `bunx biome check --write .`
Expected: All files formatted and linted

**Step 2: Run full test suite**

Run: `bun test`
Expected: All PASS

**Step 3: Verify build**

Run: `bun run build`
Expected: Build succeeds (may need adjustments for new Transformers.js dependency in the standalone binary)

**Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "chore: format and lint all files"
```

---

### Summary of deliverables

| Task | Creates/Modifies | Purpose |
|------|-----------------|---------|
| 1 | `package.json` | Swap dependencies |
| 2 | `src/db/migrations.ts` | Add embedding column |
| 3 | `src/models/manager.ts` | Model lifecycle management |
| 4 | `src/models/tool-call-parser.ts` | Parse Qwen3 tool calls |
| 5 | `src/models/prompts.ts` | Simplified prompts with tool schema |
| 6 | `src/worker/local-agent.ts` | Core agent replacement |
| 7 | `src/worker/main.ts` | Wire into worker |
| 8 | `src/utils/relevance.ts` | Cosine similarity for embeddings |
| 9 | Remove ChromaDB, extract types | Cleanup |
| 10 | All files | Format, lint, validate |
