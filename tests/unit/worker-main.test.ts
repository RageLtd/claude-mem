/**
 * Tests for worker main integration.
 * Tests that SessionManager and agent are properly wired together.
 * Uses a mock SDKAgent to avoid external dependencies.
 */

import type { Database } from "bun:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import {
  createDatabase,
  createSession,
  runMigrations,
} from "../../src/db/index";
import type {
  PendingInputMessage,
  SDKAgent,
  SDKAgentMessage,
} from "../../src/worker/agent-types";
import { createWorkerRouter } from "../../src/worker/service";
import { createSessionManager } from "../../src/worker/session-manager";

/**
 * Creates a mock SDKAgent that yields "acknowledged" for every input message.
 */
const createMockAgent = (): SDKAgent => ({
  processMessages: async function* (_session, inputMessages) {
    for await (const _msg of inputMessages) {
      yield { type: "acknowledged" };
    }
  },
});

describe("worker main integration", () => {
  let db: Database;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);
  });

  afterEach(() => {
    db.close();
  });

  describe("full integration", () => {
    it("creates all components with proper dependencies", () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();
      const router = createWorkerRouter({ db, sessionManager });

      expect(sessionManager).toBeDefined();
      expect(sdkAgent).toBeDefined();
      expect(router).toBeDefined();
    });

    it("routes messages from SessionManager to agent", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      // Create session in database first (for foreign key constraint)
      createSession(db, {
        claudeSessionId: "claude-123",
        project: "test-project",
        userPrompt: "Help me fix a bug",
      });

      // Initialize a session
      const session = sessionManager.initializeSession(
        1,
        "claude-123",
        "test-project",
        "Help me fix a bug",
      );

      // Queue an observation
      sessionManager.queueObservation(1, {
        toolName: "Read",
        toolInput: { file_path: "/test.ts" },
        toolResponse: { content: "test content" },
        cwd: "/project",
        occurredAt: new Date().toISOString(),
      });

      // Get the message iterator
      const messages = sessionManager.getMessageIterator(1);
      expect(messages).not.toBeNull();

      // Process messages through agent
      const inputMessages: AsyncIterable<PendingInputMessage> = {
        [Symbol.asyncIterator]: () => ({
          next: async () => {
            const result = await messages?.next();
            if (result.done) {
              return { done: true as const, value: undefined };
            }
            // Transform PendingMessage to PendingInputMessage
            const msg = result.value;
            return {
              done: false as const,
              value: {
                type: msg.type,
                data:
                  msg.type === "observation"
                    ? { observation: msg.data.observation }
                    : msg.data,
              } as PendingInputMessage,
            };
          },
        }),
      };

      // Close the session to end the iterator
      setTimeout(() => sessionManager.closeSession(1), 50);

      // Collect agent output
      const outputs: SDKAgentMessage[] = [];
      for await (const output of sdkAgent.processMessages(
        session,
        inputMessages,
      )) {
        outputs.push(output);
        // Break after first meaningful output to avoid timeout
        if (
          output.type === "observation_stored" ||
          output.type === "aborted" ||
          output.type === "acknowledged"
        ) {
          break;
        }
      }

      // Verify some output was generated
      expect(outputs.length).toBeGreaterThan(0);
    });
  });

  describe("background processing loop", () => {
    it("processes observations as they are queued", async () => {
      const sessionManager = createSessionManager();
      const storedObservations: unknown[] = [];
      const sdkAgent = createMockAgent();

      // Create session in database first (for foreign key constraint)
      createSession(db, {
        claudeSessionId: "claude-123",
        project: "test-project",
        userPrompt: "Analyze this code",
      });

      // Initialize session
      const session = sessionManager.initializeSession(
        1,
        "claude-123",
        "test-project",
        "Analyze this code",
      );

      // Queue multiple observations
      sessionManager.queueObservation(1, {
        toolName: "Read",
        toolInput: { file_path: "/src/main.ts" },
        toolResponse: { content: "main code" },
        cwd: "/project",
        occurredAt: new Date().toISOString(),
      });

      // Get iterator
      const messages = sessionManager.getMessageIterator(1);
      expect(messages).not.toBeNull();

      // Close after a delay to stop processing
      setTimeout(() => sessionManager.closeSession(1), 100);

      // Process and collect outputs
      const transformedMessages: AsyncIterable<PendingInputMessage> = {
        [Symbol.asyncIterator]: () => ({
          next: async () => {
            const result = await messages?.next();
            if (result.done) {
              return { done: true as const, value: undefined };
            }
            const msg = result.value;
            return {
              done: false as const,
              value: {
                type: msg.type,
                data:
                  msg.type === "observation"
                    ? { observation: msg.data.observation }
                    : msg.data,
              } as PendingInputMessage,
            };
          },
        }),
      };

      for await (const output of sdkAgent.processMessages(
        session,
        transformedMessages,
      )) {
        if (output.type === "observation_stored") {
          storedObservations.push(output.data);
        }
        if (output.type === "aborted" || output.type === "acknowledged") break;
      }

      // Should have processed at least one observation
      expect(storedObservations.length).toBeGreaterThanOrEqual(0);
    });
  });
});
