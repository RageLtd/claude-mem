import { describe, expect, it } from "bun:test";
import { createModelManager } from "../../src/models/manager";

describe("ModelManager", () => {
  it("creates with default configuration", () => {
    const manager = createModelManager({});
    expect(manager).toBeDefined();
    expect(manager.getConfig().generativeModelId).toBe(
      "onnx-community/Qwen3-0.6B-ONNX",
    );
    expect(manager.getConfig().embeddingModelId).toBe(
      "Xenova/all-MiniLM-L6-v2",
    );
  });

  it("accepts custom model IDs via config", () => {
    const manager = createModelManager({
      generativeModelId: "onnx-community/Qwen2.5-0.5B-Instruct",
      embeddingModelId: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
    });
    expect(manager.getConfig().generativeModelId).toBe(
      "onnx-community/Qwen2.5-0.5B-Instruct",
    );
    expect(manager.getConfig().embeddingModelId).toBe(
      "onnx-community/Qwen3-Embedding-0.6B-ONNX",
    );
  });

  it("accepts dtype configuration", () => {
    const manager = createModelManager({ dtype: "q4f16" });
    expect(manager.getConfig().dtype).toBe("q4f16");
  });
});
