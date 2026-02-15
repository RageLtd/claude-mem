/**
 * Model lifecycle manager for local ONNX inference.
 * Handles model download, caching, and lazy loading via Transformers.js.
 *
 * Uses AutoTokenizer + AutoModelForCausalLM directly for text generation
 * (rather than the pipeline API) because the pipeline's apply_chat_template
 * call doesn't pass tools or enable_thinking through to the Jinja template.
 * See: node_modules/@huggingface/transformers/src/pipelines.js:1024
 */

import { join } from "node:path";
import type {
  DataType,
  FeatureExtractionPipeline,
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
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

  // Lazy-loaded model references
  let tokenizer: PreTrainedTokenizer | null = null;
  let generativeModel: PreTrainedModel | null = null;
  let embeddingPipeline: FeatureExtractionPipeline | null = null;

  const getGenerativeComponents = async (): Promise<{
    tokenizer: PreTrainedTokenizer;
    model: PreTrainedModel;
  }> => {
    if (!tokenizer || !generativeModel) {
      const { AutoTokenizer, AutoModelForCausalLM } = await import(
        "@huggingface/transformers"
      );
      tokenizer = await AutoTokenizer.from_pretrained(
        config.generativeModelId,
        { cache_dir: config.cacheDir },
      );
      generativeModel = await AutoModelForCausalLM.from_pretrained(
        config.generativeModelId,
        {
          dtype: config.dtype as DataType,
          cache_dir: config.cacheDir,
        },
      );
    }
    return {
      tokenizer: tokenizer as PreTrainedTokenizer,
      model: generativeModel as PreTrainedModel,
    };
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
    const gen = await getGenerativeComponents();
    const chatInput = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Apply chat template with tools and enable_thinking control.
    // The pipeline API doesn't pass these through to apply_chat_template,
    // which is why we use AutoTokenizer + AutoModelForCausalLM directly.
    const templateOptions: Record<string, unknown> = {
      tokenize: false,
      add_generation_prompt: true,
    };
    if (tools && tools.length > 0) {
      templateOptions.tools = tools;
      // Disable Qwen3 thinking mode when tool calling — thinking mode
      // causes the model to generate free-form text instead of <tool_call> blocks
      templateOptions.enable_thinking = false;
    }

    const prompt = gen.tokenizer.apply_chat_template(
      chatInput,
      templateOptions,
    ) as string;

    const inputs = gen.tokenizer(prompt, { return_tensor: true });
    const inputLength = (inputs.input_ids as Tensor).dims[1];

    const outputIds = await gen.model.generate({
      ...inputs,
      max_new_tokens: 512,
      temperature: 0.1,
      do_sample: true,
    });

    // Decode only the generated tokens (skip the prompt)
    const allIds = (outputIds as Tensor).tolist()[0] as number[];
    const generatedIds = allIds.slice(inputLength);
    const decoded = gen.tokenizer.decode(generatedIds, {
      skip_special_tokens: true,
    });

    return decoded;
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
    if (hasDispose(generativeModel)) {
      await generativeModel.dispose();
    }
    if (hasDispose(embeddingPipeline)) {
      await embeddingPipeline.dispose();
    }
    tokenizer = null;
    generativeModel = null;
    embeddingPipeline = null;
  };

  return {
    getConfig: () => config,
    generateText,
    computeEmbedding,
    dispose,
  };
};
