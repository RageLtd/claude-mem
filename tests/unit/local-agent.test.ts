/**
 * Tests for the local agent that uses Transformers.js for observation extraction.
 * Mocks ModelManager to avoid downloading actual models.
 */

import type { Database } from "bun:sqlite";
import { afterEach, beforeEach, describe, expect, it, mock } from "bun:test";
import {
  createDatabase,
  createSession,
  getObservationById,
  runMigrations,
} from "../../src/db/index";
import type {
  ModelManager,
  ModelManagerConfig,
} from "../../src/models/manager";
import type { ToolObservation } from "../../src/types/domain";
import type {
  PendingInputMessage,
  SDKAgentMessage,
} from "../../src/worker/agent-types";
import {
  createLocalAgent,
  processObservation,
  processSummary,
  type SessionContext,
} from "../../src/worker/local-agent";
import type { ActiveSession } from "../../src/worker/session-manager";

// ============================================================================
// Mock ModelManager factory
// ============================================================================

const VALID_TOOL_CALL = `<tool_call>
{"name": "create_observation", "arguments": {"type": "feature", "title": "Added user authentication", "subtitle": "OAuth2 flow implemented", "narrative": "Implemented OAuth2 PKCE flow for user authentication", "facts": ["Uses PKCE flow", "Supports refresh tokens"], "concepts": ["what-changed"]}}
</tool_call>`;

const createMockModelManager = (
  overrides: { generateTextResponse?: string } = {},
): ModelManager => {
  const generateText = mock(
    async () => overrides.generateTextResponse ?? VALID_TOOL_CALL,
  );
  const computeEmbedding = mock(async () => new Float32Array([0.1, 0.2, 0.3]));
  const dispose = mock(async () => {});
  const getConfig = mock(
    (): ModelManagerConfig => ({
      generativeModelId: "test-model",
      embeddingModelId: "test-embed-model",
      dtype: "q4",
      cacheDir: "/tmp/test-models",
    }),
  );

  return { generateText, computeEmbedding, dispose, getConfig };
};

// ============================================================================
// Helper: create a test ActiveSession
// ============================================================================

const createTestSession = (
  overrides: Partial<ActiveSession> = {},
): ActiveSession => ({
  sessionDbId: 1,
  claudeSessionId: "test-session-123",
  project: "test-project",
  userPrompt: "Help me build something",
  startedAt: Date.now(),
  abortController: new AbortController(),
  ...overrides,
});

// ============================================================================
// Helper: async iterable from array
// ============================================================================

async function* fromArray<T>(items: readonly T[]): AsyncIterable<T> {
  for (const item of items) {
    yield item;
  }
}

// ============================================================================
// Helper: collect async iterable
// ============================================================================

const collectMessages = async (
  iterable: AsyncIterable<SDKAgentMessage>,
): Promise<readonly SDKAgentMessage[]> => {
  const messages: SDKAgentMessage[] = [];
  for await (const msg of iterable) {
    messages.push(msg);
  }
  return messages;
};

// ============================================================================
// Tests
// ============================================================================

describe("local-agent", () => {
  let db: Database;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);

    // Create session record so FK constraint is satisfied
    createSession(db, {
      claudeSessionId: "test-session-123",
      project: "test-project",
      userPrompt: "Help me build something",
    });
  });

  afterEach(() => {
    db.close();
  });

  it("processes an observation and stores it", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const observation: ToolObservation = {
      toolName: "Edit",
      toolInput: { file_path: "/src/auth.ts", new_string: "..." },
      toolResponse: "File edited",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    };

    const input: PendingInputMessage = {
      type: "observation",
      data: { observation },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    expect(messages.length).toBeGreaterThanOrEqual(1);
    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();
    expect(stored?.data).toBeDefined();

    // Verify the observation was stored in DB
    const obsData = stored?.data as { id: number };
    const dbResult = getObservationById(db, obsData.id);
    expect(dbResult.ok).toBe(true);
    if (dbResult.ok && dbResult.value) {
      expect(dbResult.value.title).toBe("Added user authentication");
      expect(dbResult.value.type).toBe("feature");
    }
  });

  it("yields acknowledged when model skips (no tool call)", async () => {
    const modelManager = createMockModelManager({
      generateTextResponse: "This is a trivial operation, skipping.",
    });
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const observation: ToolObservation = {
      toolName: "LS",
      toolInput: { path: "/tmp" },
      toolResponse: "file1.ts\nfile2.ts",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    };

    const input: PendingInputMessage = {
      type: "observation",
      data: { observation },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    expect(messages.length).toBe(1);
    expect(messages[0].type).toBe("acknowledged");
  });

  it("extracts file paths from Edit tool input", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const observation: ToolObservation = {
      toolName: "Edit",
      toolInput: {
        file_path: "/src/components/Button.tsx",
        old_string: "a",
        new_string: "b",
      },
      toolResponse: "File edited",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    };

    const input: PendingInputMessage = {
      type: "observation",
      data: { observation },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();

    const obsData = stored?.data as { id: number };
    const dbResult = getObservationById(db, obsData.id);
    expect(dbResult.ok).toBe(true);
    if (dbResult.ok && dbResult.value) {
      expect(dbResult.value.filesModified).toContain(
        "/src/components/Button.tsx",
      );
    }
  });

  it("yields aborted when session is already aborted", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    // Abort before processing
    session.abortController.abort();

    const observation: ToolObservation = {
      toolName: "Edit",
      toolInput: { file_path: "/src/app.ts" },
      toolResponse: "done",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    };

    const input: PendingInputMessage = {
      type: "observation",
      data: { observation },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    expect(messages.length).toBe(1);
    expect(messages[0].type).toBe("aborted");
  });

  it("stores embedding alongside observation", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const observation: ToolObservation = {
      toolName: "Write",
      toolInput: {
        file_path: "/src/utils.ts",
        content: "export const foo = 1;",
      },
      toolResponse: "File written",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    };

    const input: PendingInputMessage = {
      type: "observation",
      data: { observation },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();

    // Verify computeEmbedding was called
    expect(modelManager.computeEmbedding).toHaveBeenCalled();

    // Verify embedding was stored in DB
    const obsData = stored?.data as { id: number };
    const row = db
      .query<{ embedding: Buffer | null }, [number]>(
        "SELECT embedding FROM observations WHERE id = ?",
      )
      .get(obsData.id);

    expect(row).toBeDefined();
    expect(row?.embedding).not.toBeNull();

    // Verify the stored embedding matches
    if (row?.embedding) {
      const restored = new Float32Array(
        row.embedding.buffer,
        row.embedding.byteOffset,
        row.embedding.byteLength / Float32Array.BYTES_PER_ELEMENT,
      );
      expect(restored[0]).toBeCloseTo(0.1);
      expect(restored[1]).toBeCloseTo(0.2);
      expect(restored[2]).toBeCloseTo(0.3);
    }
  });

  it("extracts filesRead from Read tool input", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const observation: ToolObservation = {
      toolName: "Read",
      toolInput: { file_path: "/src/config.ts" },
      toolResponse: "export const config = {}",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    };

    const input: PendingInputMessage = {
      type: "observation",
      data: { observation },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();

    const obsData = stored?.data as { id: number };
    const dbResult = getObservationById(db, obsData.id);
    expect(dbResult.ok).toBe(true);
    if (dbResult.ok && dbResult.value) {
      expect(dbResult.value.filesRead).toContain("/src/config.ts");
    }
  });

  it("handles summarize messages with tool calling", async () => {
    const summaryToolCall = `<tool_call>
{"name": "create_summary", "arguments": {"request": "Help me add auth", "investigated": "OAuth2 providers", "learned": "PKCE flow is required for SPAs", "completed": "Implemented OAuth2 PKCE flow", "nextSteps": "Add refresh token rotation"}}
</tool_call>`;
    const modelManager = createMockModelManager({
      generateTextResponse: summaryToolCall,
    });
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const input: PendingInputMessage = {
      type: "summarize",
      data: {
        lastUserMessage: "Help me add auth",
        lastAssistantMessage: "I implemented OAuth2",
      },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    expect(messages.length).toBeGreaterThanOrEqual(1);
    const stored = messages.find((m) => m.type === "summary_stored");
    expect(stored).toBeDefined();

    const summaryData = stored?.data as {
      id: number;
      summary: {
        request: string | null;
        investigated: string | null;
        learned: string | null;
        completed: string | null;
        nextSteps: string | null;
      };
    };
    expect(summaryData.summary.request).toBe("Help me add auth");
    expect(summaryData.summary.investigated).toBe("OAuth2 providers");
    expect(summaryData.summary.learned).toBe("PKCE flow is required for SPAs");
    expect(summaryData.summary.completed).toBe("Implemented OAuth2 PKCE flow");
    expect(summaryData.summary.nextSteps).toBe("Add refresh token rotation");
  });

  it("falls back to buildSummaryFromResponse when model skips tool call", async () => {
    const plainResponse =
      "Implemented OAuth2 authentication with PKCE flow for the application.";
    const modelManager = createMockModelManager({
      generateTextResponse: plainResponse,
    });
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const input: PendingInputMessage = {
      type: "summarize",
      data: {
        lastUserMessage: "Help me add auth",
        lastAssistantMessage: "I implemented OAuth2",
      },
    };

    const messages = await collectMessages(
      agent.processMessages(session, fromArray([input])),
    );

    const stored = messages.find((m) => m.type === "summary_stored");
    expect(stored).toBeDefined();

    const summaryData = stored?.data as {
      id: number;
      summary: {
        request: string | null;
        completed: string | null;
        investigated: string | null;
      };
    };
    expect(summaryData.summary.request).toBe("Help me add auth");
    expect(summaryData.summary.completed).toBe(plainResponse);
    expect(summaryData.summary.investigated).toBeNull();
  });

  it("handles continuation messages by updating promptNumber", async () => {
    const modelManager = createMockModelManager();
    const agent = createLocalAgent({ db, modelManager });
    const session = createTestSession();

    const continuationInput: PendingInputMessage = {
      type: "continuation",
      data: {
        userPrompt: "Now add tests",
        promptNumber: 2,
      },
    };

    const observationInput: PendingInputMessage = {
      type: "observation",
      data: {
        observation: {
          toolName: "Write",
          toolInput: { file_path: "/tests/auth.test.ts", content: "test" },
          toolResponse: "File written",
          cwd: "/project",
          occurredAt: new Date().toISOString(),
        },
      },
    };

    const messages = await collectMessages(
      agent.processMessages(
        session,
        fromArray([continuationInput, observationInput]),
      ),
    );

    const stored = messages.find((m) => m.type === "observation_stored");
    expect(stored).toBeDefined();

    // Verify the observation was stored with promptNumber 2
    const obsData = stored?.data as { id: number };
    const dbResult = getObservationById(db, obsData.id);
    expect(dbResult.ok).toBe(true);
    if (dbResult.ok && dbResult.value) {
      expect(dbResult.value.promptNumber).toBe(2);
    }
  });
});

// ============================================================================
// Standalone processing functions
// ============================================================================

describe("standalone processing functions", () => {
  let db: Database;

  const testContext: SessionContext = {
    claudeSessionId: "test-session-standalone",
    project: "test-project",
    promptNumber: 1,
  };

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

  it("processObservation stores observation from model output", async () => {
    const modelManager = createMockModelManager();

    const result = await processObservation({ db, modelManager }, testContext, {
      toolName: "Edit",
      toolInput: { file_path: "/src/auth.ts", new_string: "..." },
      toolResponse: "File edited",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    });

    expect(result.ok).toBe(true);
    if (result.ok && result.value) {
      const dbResult = getObservationById(db, result.value);
      expect(dbResult.ok).toBe(true);
      if (dbResult.ok && dbResult.value) {
        expect(dbResult.value.title).toBe("Added user authentication");
      }
    }
  });

  it("processObservation returns ok(null) when model skips", async () => {
    const modelManager = createMockModelManager({
      generateTextResponse: "Trivial operation, skipping.",
    });

    const result = await processObservation({ db, modelManager }, testContext, {
      toolName: "LS",
      toolInput: { path: "/tmp" },
      toolResponse: "file1.ts",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeNull();
    }
  });

  it("processObservation deduplicates similar observations", async () => {
    const modelManager = createMockModelManager();

    const first = await processObservation({ db, modelManager }, testContext, {
      toolName: "Edit",
      toolInput: { file_path: "/src/auth.ts" },
      toolResponse: "Edited",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    });
    expect(first.ok).toBe(true);

    const second = await processObservation({ db, modelManager }, testContext, {
      toolName: "Edit",
      toolInput: { file_path: "/src/auth.ts" },
      toolResponse: "Edited again",
      cwd: "/project",
      occurredAt: new Date().toISOString(),
    });

    expect(second.ok).toBe(true);
    if (second.ok) {
      expect(second.value).toBeNull();
    }
  });

  it("processSummary stores summary from tool call", async () => {
    const summaryToolCall = `<tool_call>
{"name": "create_summary", "arguments": {"request": "Add auth", "completed": "Implemented OAuth2"}}
</tool_call>`;
    const modelManager = createMockModelManager({
      generateTextResponse: summaryToolCall,
    });

    const result = await processSummary({ db, modelManager }, testContext, {
      lastUserMessage: "Add auth",
      lastAssistantMessage: "Done",
    });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeGreaterThan(0);
    }
  });

  it("processSummary falls back when model skips tool call", async () => {
    const modelManager = createMockModelManager({
      generateTextResponse: "Implemented OAuth2 authentication.",
    });

    const result = await processSummary({ db, modelManager }, testContext, {
      lastUserMessage: "Add auth",
    });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeGreaterThan(0);
    }
  });
});
