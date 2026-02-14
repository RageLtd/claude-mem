/**
 * Shared agent types used by BackgroundProcessor and agent implementations.
 */

import type { ToolObservation } from "../types/domain";
import type { ActiveSession } from "./session-manager";

export type SDKAgentMessageType =
  | "observation_stored"
  | "summary_stored"
  | "aborted"
  | "error"
  | "acknowledged";

export interface SDKAgentMessage {
  readonly type: SDKAgentMessageType;
  readonly data?: unknown;
}

export interface PendingInputMessage {
  readonly type: "observation" | "summarize" | "continuation";
  readonly data: {
    readonly observation?: ToolObservation;
    readonly lastUserMessage?: string;
    readonly lastAssistantMessage?: string;
    readonly userPrompt?: string;
    readonly promptNumber?: number;
  };
}

export interface SDKAgent {
  readonly processMessages: (
    session: ActiveSession,
    inputMessages: AsyncIterable<PendingInputMessage>,
  ) => AsyncIterable<SDKAgentMessage>;
}
