/**
 * Message router — sequential FIFO queue for processing hook messages.
 * Replaces SessionManager + BackgroundProcessor with ~80 lines.
 *
 * Messages are processed one at a time through the local ONNX model.
 * No timers, no polling, no per-session state. Drain triggered at enqueue time.
 */

import type { Database } from "bun:sqlite";
import { getSessionByClaudeId, updateSessionStatus } from "../db/index";
import type { ModelManager } from "../models/manager";
import {
  processObservation,
  processSummary,
  type SessionContext,
} from "./local-agent";

// ============================================================================
// Types
// ============================================================================

export interface ProcessMessageDeps {
  readonly db: Database;
  readonly modelManager: ModelManager;
}

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

export const createMessageRouter = (deps: MessageRouterDeps): MessageRouter => {
  const queue: RouterMessage[] = [];
  let drainPromise: Promise<void> | null = null;

  const drain = async () => {
    for (let msg = queue.shift(); msg; msg = queue.shift()) {
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

// ============================================================================
// Process message dispatcher
// ============================================================================

export const createProcessMessage = (
  deps: ProcessMessageDeps,
): ((msg: RouterMessage) => Promise<void>) => {
  return async (msg: RouterMessage): Promise<void> => {
    const { db } = deps;

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
    }
  };
};
