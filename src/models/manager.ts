/**
 * Model lifecycle manager for local ONNX inference.
 * Handles model download, caching, and lazy loading via Transformers.js.
 */

import { join } from "node:path";
import type {
  DataType,
  FeatureExtractionPipeline,
  Tensor,
  TextGenerationOutput,
  TextGenerationPipeline,
} from "@huggingface/transformers";

// ============================================================================
// Types
// ============================================================================

export interface ModelManagerConfig {
  readonly generativeModelId: string;
  readonly embeddingModelId: string;
  readonly dtype: string;
  readonly cacheDir: string;
}

export interface ModelManagerDeps {
  readonly generativeModelId?: string;
  readonly embeddingModelId?: string;
  readonly dtype?: string;
  readonly cacheDir?: string;
}

export interface ModelManager {
  readonly getConfig: () => ModelManagerConfig;
  readonly generateText: (
    messages: readonly ChatMessage[],
    tools?: readonly ToolDefinition[],
  ) => Promise<string>;
  readonly computeEmbedding: (text: string) => Promise<Float32Array>;
  readonly dispose: () => Promise<void>;
}

export interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

export interface ToolDefinition {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly description: string;
    readonly parameters: Record<string, unknown>;
  };
}

// ============================================================================
// Defaults
// ============================================================================

const DEFAULT_GEN_MODEL = "onnx-community/Qwen3-0.6B-ONNX";
const DEFAULT_EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const DEFAULT_DTYPE: DataType = "q4";
const DEFAULT_CACHE_DIR = join(process.env.HOME || "", ".claude-mem", "models");

// ============================================================================
// Internal helpers
// ============================================================================

/** Type guard for objects with a dispose method */
const hasDispose = (
  value: unknown,
): value is { dispose: () => Promise<void> } =>
  typeof value === "object" &&
  value !== null &&
  "dispose" in value &&
  typeof (value as Record<string, unknown>).dispose === "function";

/** Extract generated text from a TextGenerationOutput entry */
const extractGeneratedText = (output: TextGenerationOutput): string => {
  const generated = output[0]?.generated_text;
  if (Array.isArray(generated)) {
    const last = generated[generated.length - 1];
    return typeof last === "string" ? last : (last?.content ?? "");
  }
  return typeof generated === "string" ? generated : "";
};

// ============================================================================
// Factory
// ============================================================================

export const createModelManager = (deps: ModelManagerDeps): ModelManager => {
  const config: ModelManagerConfig = {
    generativeModelId:
      deps.generativeModelId ||
      process.env.CLAUDE_MEM_GEN_MODEL ||
      DEFAULT_GEN_MODEL,
    embeddingModelId:
      deps.embeddingModelId ||
      process.env.CLAUDE_MEM_EMBED_MODEL ||
      DEFAULT_EMBED_MODEL,
    dtype: deps.dtype || process.env.CLAUDE_MEM_GEN_DTYPE || DEFAULT_DTYPE,
    cacheDir:
      deps.cacheDir || process.env.CLAUDE_MEM_MODEL_DIR || DEFAULT_CACHE_DIR,
  };

  // Lazy-loaded pipeline references
  let generativePipeline: TextGenerationPipeline | null = null;
  let embeddingPipeline: FeatureExtractionPipeline | null = null;

  const getGenerativePipeline = async (): Promise<TextGenerationPipeline> => {
    if (!generativePipeline) {
      const { pipeline } = await import("@huggingface/transformers");
      generativePipeline = (await pipeline(
        "text-generation" as "text-generation",
        config.generativeModelId,
        {
          dtype: config.dtype as DataType,
          cache_dir: config.cacheDir,
        },
      )) as TextGenerationPipeline;
    }
    return generativePipeline;
  };

  const getEmbeddingPipeline = async (): Promise<FeatureExtractionPipeline> => {
    if (!embeddingPipeline) {
      const { pipeline } = await import("@huggingface/transformers");
      embeddingPipeline = await pipeline(
        "feature-extraction",
        config.embeddingModelId,
        { cache_dir: config.cacheDir },
      );
    }
    return embeddingPipeline;
  };

  const generateText = async (
    messages: readonly ChatMessage[],
    tools?: readonly ToolDefinition[],
  ): Promise<string> => {
    const gen = await getGenerativePipeline();
    const chatInput = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));
    const genOptions: Record<string, unknown> = {
      max_new_tokens: 512,
      temperature: 0.1,
      do_sample: true,
    };
    if (tools && tools.length > 0) {
      genOptions.tools = tools;
    }
    const output = (await gen(chatInput, genOptions)) as TextGenerationOutput;
    return extractGeneratedText(output);
  };

  const computeEmbedding = async (text: string): Promise<Float32Array> => {
    const embed = await getEmbeddingPipeline();
    const output: Tensor = await embed(text, {
      pooling: "mean",
      normalize: true,
    });
    return new Float32Array(output.data as Float32Array);
  };

  const dispose = async (): Promise<void> => {
    if (hasDispose(generativePipeline)) {
      await generativePipeline.dispose();
    }
    if (hasDispose(embeddingPipeline)) {
      await embeddingPipeline.dispose();
    }
    generativePipeline = null;
    embeddingPipeline = null;
  };

  return {
    getConfig: () => config,
    generateText,
    computeEmbedding,
    dispose,
  };
};
