/**
 * Message router — sequential FIFO queue for processing hook messages.
 * Replaces SessionManager + BackgroundProcessor with ~50 lines.
 *
 * Messages are processed one at a time through the local ONNX model.
 * No timers, no polling, no per-session state. Drain triggered at enqueue time.
 */

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
