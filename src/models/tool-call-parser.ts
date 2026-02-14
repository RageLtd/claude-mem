/**
 * Parser for Qwen3 tool call output.
 * Extracts structured observation data from <tool_call> blocks.
 */

import { isObservationType, type ObservationType } from "../types/domain";

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
  if (
    typeof rawArgs.title !== "string" ||
    typeof rawArgs.narrative !== "string"
  ) {
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
      subtitle:
        typeof rawArgs.subtitle === "string" ? rawArgs.subtitle : undefined,
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
