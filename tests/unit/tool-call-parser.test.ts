import { describe, expect, it } from "bun:test";
import { parseToolCall } from "../../src/models/tool-call-parser";

describe("parseToolCall", () => {
  it("parses a valid tool call with all fields", () => {
    const input = `Let me analyze this.
<tool_call>
{"name": "create_observation", "arguments": {"type": "bugfix", "title": "Fixed async token bug", "subtitle": "Added missing await", "narrative": "The getToken call was missing await", "facts": ["getToken is async"], "concepts": ["problem-solution"]}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    expect(result?.name).toBe("create_observation");
    expect(result?.arguments.type).toBe("bugfix");
    expect(result?.arguments.title).toBe("Fixed async token bug");
    expect(result?.arguments.narrative).toBe(
      "The getToken call was missing await",
    );
    expect(result?.arguments.facts).toEqual(["getToken is async"]);
    expect(result?.arguments.concepts).toEqual(["problem-solution"]);
  });

  it("parses minimal required fields", () => {
    const input = `<tool_call>
{"name": "create_observation", "arguments": {"type": "discovery", "title": "Found config pattern", "narrative": "The config uses a factory function"}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    expect(result?.arguments.type).toBe("discovery");
    expect(result?.arguments.subtitle).toBeUndefined();
    expect(result?.arguments.facts).toBeUndefined();
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
    expect(result?.arguments.type).toBe("bugfix");
  });

  it("validates observation type is a known enum value", () => {
    const input = `<tool_call>
{"name": "create_observation", "arguments": {"type": "invalid_type", "title": "Test", "narrative": "Test"}}
</tool_call>`;

    const result = parseToolCall(input);
    expect(result).not.toBeNull();
    expect(result?.arguments.type).toBe("change");
  });
});
