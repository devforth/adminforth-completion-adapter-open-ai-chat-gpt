declare module "@adminforth/completion-adapter-openai-responses" {
  export interface AdapterOptions {
    openAiApiKey: string;
    model?: string;
    extraRequestBodyParameters?: Record<string, unknown>;
  }

  export default class CompletionAdapterOpenAIResponses {
    constructor(options: AdapterOptions);
    validate(): void;
    measureTokensCount(content: string): number;
    complete(
      content: string,
      maxTokens?: number,
      outputSchema?: any,
      reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh",
      toolsOrOnChunk?: unknown,
      onChunk?: unknown,
    ): Promise<{
      content?: string;
      finishReason?: string;
      error?: string;
    }>;
  }
}