# Event-Driven Message Router Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SessionManager + BackgroundProcessor (~624 lines) with a single message-router module (~80 lines) that processes messages through a FIFO queue with no timers, polling, or per-session state.

**Architecture:** HTTP handlers enqueue typed messages into a global FIFO queue. A drain function processes them sequentially through the local ONNX model. Session context is read from the database, not held in memory. Single-session design — no routing or multiplexing.

**Tech Stack:** Bun, TypeScript, SQLite (bun:sqlite), Transformers.js (local ONNX model)

---

### Task 1: Create message-router with queue/drain/shutdown

Create the core queue mechanism with a mock processor for isolated testing.

**Files:**
- Create: `src/worker/message-router.ts`
- Create: `tests/unit/message-router.test.ts`

**Step 1: Write the failing tests**

In `tests/unit/message-router.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it, mock } from "bun:test";
import {
  createMessageRouter,
  type MessageRouter,
  type RouterMessage,
} from "../../src/worker/message-router";

describe("message-router", () => {
  it("processes enqueued messages in FIFO order", async () => {
    const processed: string[] = [];
    const processor = mock(async (msg: RouterMessage) => {
      processed.push(msg.claudeSessionId);
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue({
      type: "observation",
      claudeSessionId: "a",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });
    router.enqueue({
      type: "observation",
      claudeSessionId: "b",
      data: { toolName: "Read", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });

    await router.shutdown();

    expect(processed).toEqual(["a", "b"]);
    expect(processor).toHaveBeenCalledTimes(2);
  });

  it("processes messages sequentially (not in parallel)", async () => {
    const timeline: string[] = [];
    const processor = mock(async (msg: RouterMessage) => {
      timeline.push(`start-${msg.claudeSessionId}`);
      await new Promise((r) => setTimeout(r, 10));
      timeline.push(`end-${msg.claudeSessionId}`);
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue({
      type: "observation",
      claudeSessionId: "first",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });
    router.enqueue({
      type: "observation",
      claudeSessionId: "second",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });

    await router.shutdown();

    expect(timeline).toEqual([
      "start-first",
      "end-first",
      "start-second",
      "end-second",
    ]);
  });

  it("shutdown resolves immediately when queue is empty", async () => {
    const processor = mock(async () => {});
    const router = createMessageRouter({ processMessage: processor });

    await router.shutdown(); // Should not hang

    expect(processor).not.toHaveBeenCalled();
  });

  it("reports pending count", async () => {
    let resolveFirst: () => void;
    const blockingPromise = new Promise<void>((r) => {
      resolveFirst = r;
    });
    const processor = mock(async () => {
      await blockingPromise;
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue({
      type: "observation",
      claudeSessionId: "a",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });
    router.enqueue({
      type: "observation",
      claudeSessionId: "b",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });

    // First message is being processed, second is pending
    expect(router.pending()).toBe(1);

    resolveFirst!();
    await router.shutdown();

    expect(router.pending()).toBe(0);
  });

  it("continues processing after a message handler throws", async () => {
    const processed: string[] = [];
    let callCount = 0;
    const processor = mock(async (msg: RouterMessage) => {
      callCount++;
      if (callCount === 1) throw new Error("boom");
      processed.push(msg.claudeSessionId);
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue({
      type: "observation",
      claudeSessionId: "fails",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });
    router.enqueue({
      type: "observation",
      claudeSessionId: "succeeds",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });

    await router.shutdown();

    expect(processed).toEqual(["succeeds"]);
  });

  it("handles messages enqueued during processing", async () => {
    const processed: string[] = [];
    let enqueueMore: (() => void) | null = null;

    const processor = mock(async (msg: RouterMessage) => {
      processed.push(msg.claudeSessionId);
      if (enqueueMore) {
        const fn = enqueueMore;
        enqueueMore = null;
        fn();
      }
    });

    const router = createMessageRouter({ processMessage: processor });

    enqueueMore = () => {
      router.enqueue({
        type: "observation",
        claudeSessionId: "dynamic",
        data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
      });
    };

    router.enqueue({
      type: "observation",
      claudeSessionId: "initial",
      data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
    });

    await router.shutdown();

    expect(processed).toEqual(["initial", "dynamic"]);
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `bun test tests/unit/message-router.test.ts`
Expected: FAIL — module `../../src/worker/message-router` does not exist.

**Step 3: Write minimal implementation**

In `src/worker/message-router.ts`:

```typescript
/**
 * Message router — sequential FIFO queue for processing hook messages.
 * Replaces SessionManager + BackgroundProcessor with ~50 lines.
 *
 * Messages are processed one at a time through the local ONNX model.
 * No timers, no polling, no per-session state. Drain triggered at enqueue time.
 */

import type { ToolObservation } from "../types/domain";

// ============================================================================
// Types
// ============================================================================

export interface ObservationData {
  readonly toolName: string;
  readonly toolInput: unknown;
  readonly toolResponse: unknown;
  readonly cwd: string;
}

export interface SummarizeData {
  readonly lastUserMessage: string;
  readonly lastAssistantMessage?: string;
}

export interface CompleteData {
  readonly reason: string;
}

export type RouterMessageType = "observation" | "summarize" | "complete";

export interface RouterMessage {
  readonly type: RouterMessageType;
  readonly claudeSessionId: string;
  readonly data: ObservationData | SummarizeData | CompleteData;
}

export interface MessageRouterDeps {
  readonly processMessage: (msg: RouterMessage) => Promise<void>;
}

export interface MessageRouter {
  readonly enqueue: (msg: RouterMessage) => void;
  readonly shutdown: () => Promise<void>;
  readonly pending: () => number;
}

// ============================================================================
// Factory
// ============================================================================

const log = (msg: string) => console.log(`[router] ${msg}`);

export const createMessageRouter = (
  deps: MessageRouterDeps,
): MessageRouter => {
  const queue: RouterMessage[] = [];
  let drainPromise: Promise<void> | null = null;

  const drain = async () => {
    while (queue.length > 0) {
      const msg = queue.shift()!;
      try {
        await deps.processMessage(msg);
      } catch (e) {
        log(
          `Error processing ${msg.type} for ${msg.claudeSessionId}: ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
    }
    drainPromise = null;
  };

  return {
    enqueue: (msg: RouterMessage) => {
      queue.push(msg);
      if (!drainPromise) {
        drainPromise = drain();
      }
    },
    shutdown: () => drainPromise ?? Promise.resolve(),
    pending: () => queue.length,
  };
};
```

**Step 4: Run tests to verify they pass**

Run: `bun test tests/unit/message-router.test.ts`
Expected: All 6 tests PASS.

**Step 5: Lint and commit**

```bash
bunx biome check --write src/worker/message-router.ts tests/unit/message-router.test.ts
git add src/worker/message-router.ts tests/unit/message-router.test.ts
git commit -m "feat: add message-router with FIFO queue and drain"
```

---

### Task 2: Extract standalone functions from local-agent

Refactor the local agent's async generator into two standalone pure functions (`processObservation`, `processSummary`) that take explicit dependencies. Keep the existing `createLocalAgent` temporarily for backwards compatibility until handlers are updated.

**Files:**
- Modify: `src/worker/local-agent.ts`
- Modify: `tests/unit/local-agent.test.ts`

**Step 1: Write failing tests for standalone functions**

Add to `tests/unit/local-agent.test.ts` — a new `describe("standalone processing functions")` block:

```typescript
import {
  createLocalAgent,
  processObservation,
  processSummary,
  type SessionContext,
} from "../../src/worker/local-agent";

// Add this describe block after the existing "local-agent" describe:

describe("standalone processing functions", () => {
  let db: Database;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);
    createSession(db, {
      claudeSessionId: "test-session-standalone",
      project: "test-project",
      userPrompt: "Help me build something",
    });
  });

  afterEach(() => {
    db.close();
  });

  const testContext: SessionContext = {
    claudeSessionId: "test-session-standalone",
    project: "test-project",
    promptNumber: 1,
  };

  it("processObservation stores observation from model output", async () => {
    const modelManager = createMockModelManager();

    const result = await processObservation(
      { db, modelManager },
      testContext,
      {
        toolName: "Edit",
        toolInput: { file_path: "/src/auth.ts", new_string: "..." },
        toolResponse: "File edited",
        cwd: "/project",
        occurredAt: new Date().toISOString(),
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeGreaterThan(0);
      const dbResult = getObservationById(db, result.value);
      expect(dbResult.ok).toBe(true);
      if (dbResult.ok && dbResult.value) {
        expect(dbResult.value.title).toBe("Added user authentication");
      }
    }
  });

  it("processObservation returns ok with null when model skips", async () => {
    const modelManager = createMockModelManager({
      generateTextResponse: "Trivial operation, skipping.",
    });

    const result = await processObservation(
      { db, modelManager },
      testContext,
      {
        toolName: "LS",
        toolInput: { path: "/tmp" },
        toolResponse: "file1.ts",
        cwd: "/project",
        occurredAt: new Date().toISOString(),
      },
    );

    expect(result.ok).toBe(true);
    expect(result.value).toBeNull();
  });

  it("processObservation deduplicates similar observations", async () => {
    const modelManager = createMockModelManager();

    // Store first
    const first = await processObservation(
      { db, modelManager },
      testContext,
      {
        toolName: "Edit",
        toolInput: { file_path: "/src/auth.ts" },
        toolResponse: "Edited",
        cwd: "/project",
        occurredAt: new Date().toISOString(),
      },
    );
    expect(first.ok).toBe(true);

    // Store duplicate
    const second = await processObservation(
      { db, modelManager },
      testContext,
      {
        toolName: "Edit",
        toolInput: { file_path: "/src/auth.ts" },
        toolResponse: "Edited again",
        cwd: "/project",
        occurredAt: new Date().toISOString(),
      },
    );

    expect(second.ok).toBe(true);
    expect(second.value).toBeNull(); // Deduplicated
  });

  it("processSummary stores summary from tool call", async () => {
    const summaryToolCall = `<tool_call>
{"name": "create_summary", "arguments": {"request": "Add auth", "completed": "Implemented OAuth2"}}
</tool_call>`;
    const modelManager = createMockModelManager({
      generateTextResponse: summaryToolCall,
    });

    const result = await processSummary(
      { db, modelManager },
      testContext,
      {
        lastUserMessage: "Add auth",
        lastAssistantMessage: "Done",
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeGreaterThan(0);
    }
  });

  it("processSummary falls back when model skips tool call", async () => {
    const modelManager = createMockModelManager({
      generateTextResponse: "Implemented OAuth2 authentication.",
    });

    const result = await processSummary(
      { db, modelManager },
      testContext,
      {
        lastUserMessage: "Add auth",
      },
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeGreaterThan(0);
    }
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `bun test tests/unit/local-agent.test.ts`
Expected: FAIL — `processObservation` and `processSummary` are not exported from `local-agent`.

**Step 3: Extract standalone functions**

In `src/worker/local-agent.ts`, add the `SessionContext` type and export the two standalone functions. These are extracted from the logic inside the `processMessages` generator — same code, just a different entry point.

Add the `SessionContext` interface:

```typescript
export interface SessionContext {
  readonly claudeSessionId: string;
  readonly project: string;
  readonly promptNumber: number;
}
```

Add `processObservation` function (extracted from lines 184-271 of the current generator):

```typescript
export const processObservation = async (
  deps: LocalAgentDeps,
  context: SessionContext,
  observation: ToolObservation,
): Promise<Result<number | null, Error>> => {
  const { db, modelManager } = deps;
  const systemPrompt = buildLocalSystemPrompt();
  const userPrompt = buildLocalObservationPrompt(observation);

  log(`Processing observation for tool=${observation.toolName}`);

  const response = await modelManager.generateText(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    [OBSERVATION_TOOL],
  );

  const toolCall = parseToolCall(response);
  if (!toolCall) {
    log("Model skipped observation (no tool call)");
    return ok(null);
  }

  const args = toolCall.arguments;
  const { filesRead, filesModified } = extractFilePaths(
    observation.toolName,
    observation.toolInput,
  );

  const parsed: ParsedObservation = {
    type: args.type,
    title: args.title,
    subtitle: args.subtitle ?? null,
    narrative: args.narrative,
    facts: args.facts ?? [],
    concepts: args.concepts ?? [],
    filesRead,
    filesModified,
  };

  // Deduplication check
  const dupCheck = findSimilarObservation(db, {
    project: context.project,
    title: parsed.title || "",
    withinMs: 3600000,
  });

  if (dupCheck.ok && dupCheck.value) {
    log(
      `Skipping duplicate: "${parsed.title}" (similar to #${dupCheck.value.id})`,
    );
    return ok(null);
  }

  const result = storeObservation(db, {
    claudeSessionId: context.claudeSessionId,
    project: context.project,
    observation: parsed,
    promptNumber: context.promptNumber,
    discoveryTokens: 0,
  });

  if (!result.ok) {
    return err(new Error(result.error.message));
  }

  log(`Observation stored with id=${result.value}`);

  // Store embedding asynchronously
  storeEmbedding(db, modelManager, result.value, parsed.title || "", parsed.narrative || "");

  return ok(result.value);
};
```

Add `processSummary` function (extracted from lines 273-326 of the current generator):

```typescript
export const processSummary = async (
  deps: LocalAgentDeps,
  context: SessionContext,
  input: { readonly lastUserMessage: string; readonly lastAssistantMessage?: string },
): Promise<Result<number, Error>> => {
  const { db, modelManager } = deps;
  const systemPrompt = buildLocalSystemPrompt();
  const userPrompt = buildLocalSummaryPrompt({
    lastUserMessage: input.lastUserMessage,
    lastAssistantMessage: input.lastAssistantMessage,
  });

  log("Processing summarize request");

  const response = await modelManager.generateText(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    [SUMMARY_TOOL],
  );

  const toolCall = parseSummaryToolCall(response);
  const summary: ParsedSummary = toolCall
    ? {
        request: toolCall.arguments.request ?? null,
        investigated: toolCall.arguments.investigated ?? null,
        learned: toolCall.arguments.learned ?? null,
        completed: toolCall.arguments.completed ?? null,
        nextSteps: toolCall.arguments.nextSteps ?? null,
        notes: toolCall.arguments.notes ?? null,
      }
    : buildSummaryFromResponse(input.lastUserMessage || null, response);

  const result = storeSummary(db, {
    claudeSessionId: context.claudeSessionId,
    project: context.project,
    summary,
    promptNumber: context.promptNumber,
    discoveryTokens: 0,
  });

  if (!result.ok) {
    return err(new Error(result.error.message));
  }

  log(`Summary stored with id=${result.value}`);
  return ok(result.value);
};
```

Add imports for `ok`, `err`, `Result` from `../types/result` and `ToolObservation` from `../types/domain`.

**Step 4: Run tests to verify they pass**

Run: `bun test tests/unit/local-agent.test.ts`
Expected: All tests PASS (both old and new).

**Step 5: Lint and commit**

```bash
bunx biome check --write src/worker/local-agent.ts tests/unit/local-agent.test.ts
git add src/worker/local-agent.ts tests/unit/local-agent.test.ts
git commit -m "refactor: extract standalone processObservation and processSummary"
```

---

### Task 3: Wire processMessage dispatcher into message-router

Create the `createProcessMessage` function that dispatches `RouterMessage` to the standalone local-agent functions. This connects the router to actual processing.

**Files:**
- Modify: `src/worker/message-router.ts`
- Modify: `tests/unit/message-router.test.ts`

**Step 1: Write failing integration test**

Add to `tests/unit/message-router.test.ts`:

```typescript
import type { Database } from "bun:sqlite";
import {
  createDatabase,
  createSession,
  getObservationById,
  runMigrations,
  updateSessionStatus,
} from "../../src/db/index";
import type {
  ModelManager,
  ModelManagerConfig,
} from "../../src/models/manager";
import {
  createMessageRouter,
  createProcessMessage,
  type MessageRouter,
  type RouterMessage,
} from "../../src/worker/message-router";

// Copy createMockModelManager and VALID_TOOL_CALL from local-agent.test.ts

describe("message-router integration", () => {
  let db: Database;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);
    createSession(db, {
      claudeSessionId: "test-session-int",
      project: "test-project",
      userPrompt: "Build something",
    });
  });

  afterEach(() => {
    db.close();
  });

  it("processes observation message end-to-end", async () => {
    const modelManager = createMockModelManager();
    const processMessage = createProcessMessage({ db, modelManager });
    const router = createMessageRouter({ processMessage });

    router.enqueue({
      type: "observation",
      claudeSessionId: "test-session-int",
      data: {
        toolName: "Edit",
        toolInput: { file_path: "/src/app.ts" },
        toolResponse: "File edited",
        cwd: "/test-project",
      },
    });

    await router.shutdown();

    // Verify observation was stored in DB
    const obs = getObservationById(db, 1);
    expect(obs.ok).toBe(true);
    if (obs.ok && obs.value) {
      expect(obs.value.title).toBe("Added user authentication");
    }
  });

  it("processes summarize message end-to-end", async () => {
    const summaryResponse = `<tool_call>
{"name": "create_summary", "arguments": {"request": "Build something", "completed": "Built it"}}
</tool_call>`;
    const modelManager = createMockModelManager({
      generateTextResponse: summaryResponse,
    });
    const processMessage = createProcessMessage({ db, modelManager });
    const router = createMessageRouter({ processMessage });

    router.enqueue({
      type: "summarize",
      claudeSessionId: "test-session-int",
      data: {
        lastUserMessage: "Build something",
        lastAssistantMessage: "Done",
      },
    });

    await router.shutdown();

    // Verify summary was stored (check via DB query)
    const row = db
      .query("SELECT * FROM session_summaries WHERE claude_session_id = ?")
      .get("test-session-int");
    expect(row).toBeDefined();
  });

  it("processes complete message by updating session status", async () => {
    const modelManager = createMockModelManager();
    const processMessage = createProcessMessage({ db, modelManager });
    const router = createMessageRouter({ processMessage });

    router.enqueue({
      type: "complete",
      claudeSessionId: "test-session-int",
      data: { reason: "exit" },
    });

    await router.shutdown();

    // Verify session status was updated
    const session = db
      .query("SELECT status FROM sdk_sessions WHERE claude_session_id = ?")
      .get("test-session-int") as { status: string } | null;
    expect(session?.status).toBe("completed");
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `bun test tests/unit/message-router.test.ts`
Expected: FAIL — `createProcessMessage` is not exported.

**Step 3: Implement createProcessMessage**

Add to `src/worker/message-router.ts`:

```typescript
import type { Database } from "bun:sqlite";
import {
  getSessionByClaudeId,
  updateSessionStatus,
} from "../db/index";
import type { ModelManager } from "../models/manager";
import {
  processObservation,
  processSummary,
  type SessionContext,
} from "./local-agent";

export interface ProcessMessageDeps {
  readonly db: Database;
  readonly modelManager: ModelManager;
}

export const createProcessMessage = (
  deps: ProcessMessageDeps,
): ((msg: RouterMessage) => Promise<void>) => {
  return async (msg: RouterMessage): Promise<void> => {
    const { db } = deps;

    // Look up session context from DB
    const sessionResult = getSessionByClaudeId(db, msg.claudeSessionId);
    if (!sessionResult.ok || !sessionResult.value) {
      log(`Session not found for ${msg.claudeSessionId}, skipping`);
      return;
    }

    const session = sessionResult.value;
    const context: SessionContext = {
      claudeSessionId: msg.claudeSessionId,
      project: session.project,
      promptNumber: session.promptCounter || 1,
    };

    if (msg.type === "observation") {
      const data = msg.data as ObservationData;
      await processObservation(deps, context, {
        toolName: data.toolName,
        toolInput: data.toolInput,
        toolResponse: data.toolResponse,
        cwd: data.cwd,
        occurredAt: new Date().toISOString(),
      });
      return;
    }

    if (msg.type === "summarize") {
      const data = msg.data as SummarizeData;
      await processSummary(deps, context, {
        lastUserMessage: data.lastUserMessage,
        lastAssistantMessage: data.lastAssistantMessage,
      });
      return;
    }

    if (msg.type === "complete") {
      updateSessionStatus(db, session.id, "completed");
      return;
    }
  };
};
```

Note: Check that `getSessionByClaudeId` returns an object with `project`, `promptCounter`, and `id` fields. Read the DB types and adjust field names as needed.

**Step 4: Run tests to verify they pass**

Run: `bun test tests/unit/message-router.test.ts`
Expected: All tests PASS.

**Step 5: Lint and commit**

```bash
bunx biome check --write src/worker/message-router.ts tests/unit/message-router.test.ts
git add src/worker/message-router.ts tests/unit/message-router.test.ts
git commit -m "feat: wire processMessage dispatcher to local-agent functions"
```

---

### Task 4: Update handlers to use MessageRouter

Replace all `SessionManager` usage in handlers with `MessageRouter.enqueue()`.

**Files:**
- Modify: `src/worker/handlers.ts`
- Modify: `tests/unit/worker-handlers.test.ts`

**Step 1: Update WorkerDeps type**

In `src/worker/handlers.ts`, replace:

```typescript
import type { SessionManager } from "./session-manager";
```

With:

```typescript
import type { MessageRouter } from "./message-router";
```

Change `WorkerDeps`:

```typescript
export interface WorkerDeps {
  readonly db: Database;
  readonly router?: MessageRouter;
  readonly startedAt?: number;
  readonly version?: string;
}
```

**Step 2: Update handleQueueObservation**

Replace all `deps.sessionManager` logic (lines 195-227) with a single enqueue:

```typescript
  // Enqueue for background processing
  if (deps.router) {
    deps.router.enqueue({
      type: "observation",
      claudeSessionId,
      data: { toolName, toolInput, toolResponse, cwd },
    });
  }
```

Remove the session initialization logic (the SessionManager `initializeSession` calls). Keep the DB session creation logic (lines 168-216) since we still need the session record in the DB.

**Step 3: Update handleQueueSummary**

Replace `deps.sessionManager.queueSummarize()` (lines 387-393) with:

```typescript
  if (deps.router) {
    deps.router.enqueue({
      type: "summarize",
      claudeSessionId,
      data: { lastUserMessage, lastAssistantMessage },
    });
  }
```

**Step 4: Update handleCompleteSession**

Replace `deps.sessionManager.closeSession()` (lines 431-433) with:

```typescript
  if (deps.router) {
    deps.router.enqueue({
      type: "complete",
      claudeSessionId,
      data: { reason },
    });
  }
```

**Step 5: Update handleQueuePrompt**

Remove all `deps.sessionManager` calls (initializeSession, queueContinuation). The DB operations (`createSession`, `incrementPromptCounter`, `saveUserPrompt`) remain — they're the source of truth the router reads from.

**Step 6: Update handleHealth**

Replace `deps.sessionManager?.getActiveSessions().length ?? 0` with `deps.router?.pending() ?? 0`.

**Step 7: Update tests**

In `tests/unit/worker-handlers.test.ts`:

- Remove `import { createSessionManager, type SessionManager } from "../../src/worker/session-manager"`
- Remove the entire `describe("worker handlers with SessionManager integration")` block (lines 649-791) — these tests verified SessionManager queueing which no longer exists
- Add new integration tests that verify `router.enqueue` is called:

```typescript
import { createMessageRouter, type MessageRouter, type RouterMessage } from "../../src/worker/message-router";

describe("worker handlers with MessageRouter", () => {
  let db: Database;
  let enqueued: RouterMessage[];
  let router: MessageRouter;
  let deps: WorkerDeps;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);
    enqueued = [];
    router = createMessageRouter({
      processMessage: async (msg) => { enqueued.push(msg); },
    });
    deps = { db, router };
  });

  afterEach(() => {
    db.close();
  });

  it("handleQueueObservation enqueues observation message", async () => {
    createSession(db, {
      claudeSessionId: "claude-123",
      project: "test",
      userPrompt: "Test",
    });

    const result = await handleQueueObservation(deps, {
      claudeSessionId: "claude-123",
      toolName: "Edit",
      toolInput: { file_path: "/src/app.ts" },
      toolResponse: "Edited",
      cwd: "/project",
    });

    expect(result.status).toBe(200);
    await router.shutdown();
    expect(enqueued.length).toBe(1);
    expect(enqueued[0].type).toBe("observation");
    expect(enqueued[0].claudeSessionId).toBe("claude-123");
  });

  it("handleQueueSummary enqueues summarize message", async () => {
    createSession(db, {
      claudeSessionId: "claude-123",
      project: "test",
      userPrompt: "Test",
    });

    const result = await handleQueueSummary(deps, {
      claudeSessionId: "claude-123",
      lastUserMessage: "Fix bug",
      lastAssistantMessage: "Fixed",
    });

    expect(result.status).toBe(200);
    await router.shutdown();
    expect(enqueued.length).toBe(1);
    expect(enqueued[0].type).toBe("summarize");
  });

  it("handleCompleteSession enqueues complete message", async () => {
    createSession(db, {
      claudeSessionId: "claude-123",
      project: "test",
      userPrompt: "Test",
    });

    const result = await handleCompleteSession(deps, {
      claudeSessionId: "claude-123",
      reason: "exit",
    });

    expect(result.status).toBe(200);
    await router.shutdown();
    expect(enqueued.length).toBe(1);
    expect(enqueued[0].type).toBe("complete");
  });
});
```

**Step 8: Run all tests**

Run: `bun test`
Expected: All tests PASS.

**Step 9: Lint and commit**

```bash
bunx biome check --write src/worker/handlers.ts tests/unit/worker-handlers.test.ts
git add src/worker/handlers.ts tests/unit/worker-handlers.test.ts
git commit -m "refactor: replace SessionManager with MessageRouter in handlers"
```

---

### Task 5: Update main.ts and service.ts wiring

Wire the new MessageRouter into the worker startup and shutdown.

**Files:**
- Modify: `src/worker/main.ts`
- Modify: `src/worker/service.ts`

**Step 1: Update main.ts**

Replace the session manager, background processor, and eviction sweep setup with a single router creation.

Replace lines 12-16 (imports):
```typescript
import { createMessageRouter, createProcessMessage } from "./message-router";
import { createWorkerRouter } from "./service";
```

Remove imports of `createBackgroundProcessor`, `createLocalAgent`, `createSessionManager`.

Replace lines 46-96 (session manager, local agent, background processor, eviction) with:

```typescript
      // Create message router for processing hook messages
      const processMessage = createProcessMessage({ db, modelManager });
      const messageRouter = createMessageRouter({ processMessage });
      log("MessageRouter initialized");

      // Create HTTP router
      const router = createWorkerRouter({
        db,
        router: messageRouter,
        startedAt,
        version: VERSION,
      });
```

Replace shutdown (lines 99-118) with:

```typescript
      const shutdown = async () => {
        log("Shutting down...");
        log(`Waiting for ${messageRouter.pending()} pending messages...`);
        await messageRouter.shutdown();
        await modelManager.dispose();
        db.close();
        server.stop();
        process.exit(0);
      };
```

**Step 2: Update service.ts WorkerDeps import**

In `src/worker/service.ts`, the `WorkerDeps` is imported from `./handlers`. Since we updated `WorkerDeps` in Task 4, no additional changes needed in service.ts unless it imports `SessionManager` directly. Check and remove any stale imports.

**Step 3: Run all tests**

Run: `bun test`
Expected: All tests PASS.

**Step 4: Lint and commit**

```bash
bunx biome check --write src/worker/main.ts src/worker/service.ts
git add src/worker/main.ts src/worker/service.ts
git commit -m "refactor: wire MessageRouter in worker startup"
```

---

### Task 6: Delete old modules and clean up

Remove the replaced modules and all stale references.

**Files:**
- Delete: `src/worker/session-manager.ts`
- Delete: `src/worker/background-processor.ts`
- Delete: `src/worker/agent-types.ts`
- Modify: `src/worker/local-agent.ts` (remove `createLocalAgent`, `ActiveSession` import, `SDKAgent` interface)
- Modify: any files that still import from deleted modules

**Step 1: Delete the three files**

```bash
rm src/worker/session-manager.ts src/worker/background-processor.ts src/worker/agent-types.ts
```

**Step 2: Clean up local-agent.ts**

Remove the `createLocalAgent` factory function and the old `SDKAgent`/`ActiveSession` imports. The file should only export:
- `processObservation`
- `processSummary`
- `SessionContext` type
- `LocalAgentDeps` type

Remove the `import type { ... } from "./agent-types"` and `import type { ActiveSession } from "./session-manager"` lines.

**Step 3: Update local-agent.test.ts**

Remove the old `describe("local-agent")` block that tests `createLocalAgent` with the async generator pattern (lines 98-462). Keep only the `describe("standalone processing functions")` block added in Task 2.

Remove imports of `createLocalAgent`, `ActiveSession`, `PendingInputMessage`, `SDKAgentMessage`.

**Step 4: Search for stale imports**

Run: `grep -r "session-manager\|background-processor\|agent-types\|createLocalAgent\|ActiveSession\|SDKAgent\|SessionManager" src/ tests/ --include="*.ts" -l`

Fix any remaining references.

**Step 5: Run all tests**

Run: `bun test`
Expected: All tests PASS.

**Step 6: Lint and commit**

```bash
bunx biome check --write .
git add -A
git commit -m "refactor: delete SessionManager, BackgroundProcessor, and agent-types"
```

---

### Task 7: Build and verify end-to-end

Build the binary and test with actual HTTP requests to verify the new architecture works.

**Files:**
- No file changes — verification only

**Step 1: Run full test suite**

Run: `bun test`
Expected: All tests PASS.

**Step 2: Build binary**

Run: `bun run build`
Expected: Binary compiled successfully to `plugin/bin/claude-mem`.

**Step 3: Start worker and send test observation**

```bash
# Start worker (will use local ONNX model)
CLAUDE_MEM_PORT=4567 plugin/bin/claude-mem worker &
sleep 3

# Send a test observation
curl -s -X POST http://127.0.0.1:4567/observation \
  -H "Content-Type: application/json" \
  -d '{"claudeSessionId":"test-router","toolName":"Edit","toolInput":{"file_path":"src/app.ts","old_string":"a","new_string":"b"},"toolResponse":"File edited","cwd":"/tmp/test"}'

# Wait for processing
sleep 15

# Check health
curl -s http://127.0.0.1:4567/health | jq .

# Kill worker
kill %1
```

Expected: Observation is queued (200 response), health shows `pending: 0` after processing.

**Step 4: Commit verification notes (optional)**

No commit needed — this is verification only.
