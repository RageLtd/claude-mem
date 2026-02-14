/**
 * Tests for BackgroundProcessor - manages session processing with proper cleanup.
 * Uses a mock SDKAgent to avoid external dependencies.
 */

import type { Database } from "bun:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import {
  createDatabase,
  createSession,
  runMigrations,
} from "../../src/db/index";
import type { SDKAgent, SDKAgentMessage } from "../../src/worker/agent-types";
import {
  type BackgroundProcessorDeps,
  createBackgroundProcessor,
} from "../../src/worker/background-processor";
import { createSessionManager } from "../../src/worker/session-manager";

/**
 * Creates a mock SDKAgent that yields "acknowledged" for every input message.
 */
const createMockAgent = (
  overrideMessages?: readonly SDKAgentMessage[],
): SDKAgent => ({
  processMessages: async function* (_session, inputMessages) {
    if (overrideMessages) {
      for (const msg of overrideMessages) {
        yield msg;
      }
      return;
    }
    for await (const _msg of inputMessages) {
      yield { type: "acknowledged" };
    }
  },
});

describe("BackgroundProcessor", () => {
  let db: Database;

  beforeEach(() => {
    db = createDatabase(":memory:");
    runMigrations(db);
  });

  afterEach(() => {
    db.close();
  });

  describe("createBackgroundProcessor", () => {
    it("creates a processor with required methods", () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const deps: BackgroundProcessorDeps = {
        sessionManager,
        sdkAgent,
        pollIntervalMs: 1000,
      };

      const processor = createBackgroundProcessor(deps);

      expect(processor.start).toBeDefined();
      expect(processor.stop).toBeDefined();
      expect(processor.getActiveProcessingCount).toBeDefined();
      expect(processor.awaitCompletion).toBeDefined();
    });
  });

  describe("start/stop", () => {
    it("starts polling and can be stopped", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50, // Fast polling for test
      });

      processor.start();

      // Let it run briefly
      await new Promise((resolve) => setTimeout(resolve, 100));

      processor.stop();

      // Should be stopped without error
      expect(processor.getActiveProcessingCount()).toBe(0);
    });

    it("does not start multiple times", () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 1000,
      });

      processor.start();
      processor.start(); // Should be idempotent
      processor.stop();
    });
  });

  describe("session processing tracking", () => {
    it("tracks active processing count", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
      });

      // Initialize a session
      sessionManager.initializeSession(1, "claude-123", "test-project", "test");

      processor.start();

      // Wait for processing to start
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Close session to end processing
      sessionManager.closeSession(1);

      // Wait for completion
      await processor.awaitCompletion(500);

      expect(processor.getActiveProcessingCount()).toBe(0);

      processor.stop();
    });

    it("does not start duplicate processing for same session", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
      });

      // Initialize a session
      sessionManager.initializeSession(1, "claude-123", "test-project", "test");

      processor.start();

      // Let multiple poll cycles run
      await new Promise((resolve) => setTimeout(resolve, 200));

      // Close session to end processing
      sessionManager.closeSession(1);
      await processor.awaitCompletion(500);

      processor.stop();

      // Should complete without duplicates (verified by no errors)
      expect(processor.getActiveProcessingCount()).toBe(0);
    });
  });

  describe("awaitCompletion", () => {
    it("waits for all processing to complete", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
      });

      // Initialize session and queue work
      sessionManager.initializeSession(1, "claude-123", "test-project", "test");

      processor.start();

      // Wait for processing to start
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Close session
      sessionManager.closeSession(1);

      // Wait for completion
      await processor.awaitCompletion(1000);

      processor.stop();

      // Processing should have completed
      expect(processor.getActiveProcessingCount()).toBe(0);
    });

    it("times out if processing takes too long", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
      });

      // Initialize session
      sessionManager.initializeSession(1, "claude-123", "test-project", "test");

      processor.start();

      // Wait for processing to start
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Should timeout and not hang forever
      const startTime = Date.now();
      await processor.awaitCompletion(100);
      const elapsed = Date.now() - startTime;

      expect(elapsed).toBeLessThan(200);

      // Force cleanup
      sessionManager.closeSession(1);
      processor.stop();
    });
  });

  describe("error handling", () => {
    it("handles empty sessions gracefully", async () => {
      const sessionManager = createSessionManager();
      const sdkAgent = createMockAgent();

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
      });

      processor.start();

      // Let it run with no sessions
      await new Promise((resolve) => setTimeout(resolve, 100));

      processor.stop();

      // Should be stopped without error
      expect(processor.getActiveProcessingCount()).toBe(0);
    });
  });

  describe("event callbacks", () => {
    it("calls onObservationStored callback", async () => {
      const sessionManager = createSessionManager();
      const storedEvents: Array<{ sessionId: string; observationId: number }> =
        [];

      // Mock agent that yields observation_stored
      const sdkAgent = createMockAgent([
        {
          type: "observation_stored",
          data: { id: 1, observation: { type: "feature", title: "Test" } },
        },
      ]);

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
        onObservationStored: (sessionId, observationId) => {
          storedEvents.push({ sessionId, observationId });
        },
      });

      // Create session in database (required for foreign key constraint)
      createSession(db, {
        claudeSessionId: "claude-123",
        project: "test-project",
        userPrompt: "test",
      });

      // Initialize session in SessionManager
      sessionManager.initializeSession(1, "claude-123", "test-project", "test");

      processor.start();

      // Wait for processing to start and get first response
      await new Promise((resolve) => setTimeout(resolve, 200));

      // Close to end processing
      sessionManager.closeSession(1);
      await processor.awaitCompletion(500);

      processor.stop();

      // Should have received callback
      expect(storedEvents.length).toBeGreaterThan(0);
      expect(storedEvents[0].sessionId).toBe("claude-123");
    });

    it("calls onSummaryStored callback", async () => {
      const sessionManager = createSessionManager();
      const storedEvents: Array<{ sessionId: string; summaryId: number }> = [];

      // Mock agent that yields summary_stored
      const sdkAgent = createMockAgent([
        {
          type: "summary_stored",
          data: { id: 1, summary: { request: "Test request" } },
        },
      ]);

      const processor = createBackgroundProcessor({
        sessionManager,
        sdkAgent,
        pollIntervalMs: 50,
        onSummaryStored: (sessionId, summaryId) => {
          storedEvents.push({ sessionId, summaryId });
        },
      });

      // Create session in database (required for foreign key constraint)
      createSession(db, {
        claudeSessionId: "claude-123",
        project: "test-project",
        userPrompt: "test",
      });

      // Initialize session in SessionManager
      sessionManager.initializeSession(1, "claude-123", "test-project", "test");

      // Queue a summarize request
      sessionManager.queueSummarize(1, "final message", "assistant response");

      processor.start();

      // Wait for processing
      await new Promise((resolve) => setTimeout(resolve, 200));

      // Close to end processing
      sessionManager.closeSession(1);
      await processor.awaitCompletion(500);

      processor.stop();

      // Should have received callback
      expect(storedEvents.length).toBeGreaterThan(0);
      expect(storedEvents[0].sessionId).toBe("claude-123");
    });
  });
});
