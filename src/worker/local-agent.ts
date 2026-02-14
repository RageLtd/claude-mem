/**
 * Local agent for observation extraction using Transformers.js.
 * Replaces the SDK agent with local ONNX model inference.
 * Implements the same SDKAgent interface for drop-in compatibility.
 */

import type { Database } from "bun:sqlite";
import {
  findSimilarObservation,
  storeObservation,
  storeSummary,
} from "../db/index";
import type { ModelManager } from "../models/manager";
import {
  buildLocalObservationPrompt,
  buildLocalSummaryPrompt,
  buildLocalSystemPrompt,
  OBSERVATION_TOOL,
} from "../models/prompts";
import { parseToolCall } from "../models/tool-call-parser";
import type { ParsedObservation, ParsedSummary } from "../types/domain";
import { fromTry } from "../types/result";
import type {
  PendingInputMessage,
  SDKAgent,
  SDKAgentMessage,
} from "./agent-types";
import type { ActiveSession } from "./session-manager";

// ============================================================================
// Types
// ============================================================================

export interface LocalAgentDeps {
  readonly db: Database;
  readonly modelManager: ModelManager;
}

// ============================================================================
// Constants
// ============================================================================

/** Tools whose file_path or path field indicates a modified file */
const WRITE_TOOLS = new Set(["Edit", "Write", "MultiEdit", "NotebookEdit"]);

/** Tools whose file_path or path field indicates a read file */
const READ_TOOLS = new Set(["Read", "Grep", "Glob", "LS"]);

// ============================================================================
// Internal helpers
// ============================================================================

const log = (msg: string) => console.log(`[local-agent] ${msg}`);

/**
 * Extracts file paths deterministically from tool input.
 * Returns { filesRead, filesModified } based on tool name.
 */
const extractFilePaths = (
  toolName: string,
  toolInput: unknown,
): {
  readonly filesRead: readonly string[];
  readonly filesModified: readonly string[];
} => {
  const input =
    typeof toolInput === "object" && toolInput !== null
      ? (toolInput as Record<string, unknown>)
      : {};

  const filePath =
    typeof input.file_path === "string"
      ? input.file_path
      : typeof input.path === "string"
        ? input.path
        : null;

  if (!filePath) {
    return { filesRead: [], filesModified: [] };
  }

  if (WRITE_TOOLS.has(toolName)) {
    return { filesRead: [], filesModified: [filePath] };
  }

  if (READ_TOOLS.has(toolName)) {
    return { filesRead: [filePath], filesModified: [] };
  }

  // Unknown tool — default to filesRead
  return { filesRead: [filePath], filesModified: [] };
};

/**
 * Parses a simple summary from model text output.
 * Extracts lines matching key summary fields.
 */
const parseSummaryFromText = (text: string): ParsedSummary => ({
  request: text.slice(0, 500) || null,
  investigated: null,
  learned: null,
  completed: null,
  nextSteps: null,
  notes: null,
});

/**
 * Stores the embedding for an observation asynchronously.
 */
const storeEmbedding = (
  db: Database,
  modelManager: ModelManager,
  observationId: number,
  title: string,
  narrative: string,
): void => {
  const embeddingText = `${title} ${narrative}`;
  modelManager
    .computeEmbedding(embeddingText)
    .then((embedding) => {
      const result = fromTry(() =>
        db.run("UPDATE observations SET embedding = ? WHERE id = ?", [
          Buffer.from(embedding.buffer),
          observationId,
        ]),
      );
      if (!result.ok) {
        log(`Failed to store embedding: ${result.error.message}`);
      }
    })
    .catch((e) => {
      log(
        `Embedding computation failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    });
};

// ============================================================================
// Factory
// ============================================================================

export const createLocalAgent = (deps: LocalAgentDeps): SDKAgent => {
  const { db, modelManager } = deps;
  const systemPrompt = buildLocalSystemPrompt();

  const processMessages = async function* (
    session: ActiveSession,
    inputMessages: AsyncIterable<PendingInputMessage>,
  ): AsyncIterable<SDKAgentMessage> {
    log(`Starting processMessages for session ${session.claudeSessionId}`);

    // Check if already aborted
    if (session.abortController.signal.aborted) {
      log("Session already aborted");
      yield { type: "aborted" };
      return;
    }

    let promptNumber = 1;

    for await (const msg of inputMessages) {
      // Check abort between messages
      if (session.abortController.signal.aborted) {
        log("Session aborted during processing");
        yield { type: "aborted" };
        return;
      }

      log(`Received message type=${msg.type}`);

      if (msg.type === "continuation") {
        promptNumber = msg.data.promptNumber ?? promptNumber + 1;
        log(`Updated promptNumber to ${promptNumber}`);
        continue;
      }

      if (msg.type === "observation" && msg.data.observation) {
        const observation = msg.data.observation;
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
          yield { type: "acknowledged" };
          continue;
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
          project: session.project,
          title: parsed.title || "",
          withinMs: 3600000,
        });

        if (dupCheck.ok && dupCheck.value) {
          log(
            `Skipping duplicate: "${parsed.title}" (similar to #${dupCheck.value.id})`,
          );
          yield { type: "acknowledged" };
          continue;
        }

        const result = storeObservation(db, {
          claudeSessionId: session.claudeSessionId,
          project: session.project,
          observation: parsed,
          promptNumber,
          discoveryTokens: 0,
        });

        if (result.ok) {
          log(`Observation stored with id=${result.value}`);

          // Store embedding asynchronously
          storeEmbedding(
            db,
            modelManager,
            result.value,
            parsed.title || "",
            parsed.narrative || "",
          );

          yield {
            type: "observation_stored",
            data: { id: result.value, observation: parsed },
          };
        } else {
          log(`Failed to store observation: ${result.error.message}`);
          yield {
            type: "error",
            data: `Failed to store observation: ${result.error.message}`,
          };
        }

        continue;
      }

      if (msg.type === "summarize") {
        const userPrompt = buildLocalSummaryPrompt({
          lastUserMessage: msg.data.lastUserMessage || "",
          lastAssistantMessage: msg.data.lastAssistantMessage,
        });

        log("Processing summarize request");

        const response = await modelManager.generateText([
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ]);

        const summary = parseSummaryFromText(response);

        const result = storeSummary(db, {
          claudeSessionId: session.claudeSessionId,
          project: session.project,
          summary,
          promptNumber,
          discoveryTokens: 0,
        });

        if (result.ok) {
          log(`Summary stored with id=${result.value}`);
          yield {
            type: "summary_stored",
            data: { id: result.value, summary },
          };
        } else {
          log(`Failed to store summary: ${result.error.message}`);
          yield {
            type: "error",
            data: `Failed to store summary: ${result.error.message}`,
          };
        }
      }
    }

    log("processMessages complete");
  };

  return { processMessages };
};
