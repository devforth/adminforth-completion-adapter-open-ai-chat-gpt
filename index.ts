import type { AdapterOptions } from "./types.js";
import type { CompletionAdapter } from "adminforth";
import { encoding_for_model, type TiktokenModel } from "tiktoken";

type StreamChunkCallback = (chunk: string) => void | Promise<void>;

export default class CompletionAdapterOpenAIChatGPT
  implements CompletionAdapter
{
  options: AdapterOptions;
  private encoding: ReturnType<typeof encoding_for_model>;

  constructor(options: AdapterOptions) {
    this.options = options;
    this.encoding = encoding_for_model(
      (this.options.model || "gpt-5-nano") as TiktokenModel
    );
  }

  validate() {
    if (!this.options.openAiApiKey) {
      throw new Error("openAiApiKey is required");
    }
  }

  measureTokensCount(content: string): number {
    //TODO: Implement token counting logic
    const tokens = this.encoding.encode(content);
    return tokens.length;
  }

  complete = async (
    content: string,
    maxTokens: number = 50,
    outputSchema?: any,
    onChunk?: StreamChunkCallback,
  ): Promise<{
    content?: string;
    finishReason?: string;
    error?: string;
  }> => {
    // stop parameter is alredy not supported
    // adapter users should explicitely ask model to stop at dot if needed (or "Complete only up to the end of sentence")
    const model = this.options.model || "gpt-5-nano";
    const isStreaming = typeof onChunk === "function";

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.options.openAiApiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "user",
            content,
          },
        ],
        max_completion_tokens: maxTokens,
        response_format: outputSchema
          ? {
              type: "json_schema",
              ...outputSchema,
            }
          : undefined,
        stream: isStreaming,
        ...this.options.extraRequestBodyParameters,
      }),
    });

    if (!resp.ok) {
      let errorMessage = `OpenAI request failed with status ${resp.status}`;
      try {
        const errorData = await resp.json();
        if (errorData?.error?.message) {
          errorMessage = errorData.error.message;
        }
      } catch {}
      return { error: errorMessage };
    }

    if (!isStreaming) {
      const data = await resp.json();
      if (data.error) {
        return { error: data.error.message };
      }

      return {
        content: data.choices?.[0]?.message?.content,
        finishReason: data.choices?.[0]?.finish_reason,
      };
    }

    if (!resp.body) {
      return { error: "Response body is empty" };
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";
    let fullContent = "";
    let finishReason: string | undefined;

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          const line = rawLine.trim();

          if (!line || !line.startsWith("data:")) {
            continue;
          }

          const dataStr = line.slice(5).trim();

          if (dataStr === "[DONE]") {
            return {
              content: fullContent,
              finishReason,
            };
          }

          let parsed: any;
          try {
            parsed = JSON.parse(dataStr);
          } catch {
            continue;
          }

          if (parsed.error?.message) {
            return { error: parsed.error.message };
          }

          const choice = parsed.choices?.[0];
          if (!choice) {
            continue;
          }

          if (choice.finish_reason) {
            finishReason = choice.finish_reason;
          }

          const chunk = choice.delta?.content ?? "";
          if (!chunk) {
            continue;
          }

          fullContent += chunk;
          await onChunk(chunk);
        }
      }

      if (buffer.trim().startsWith("data:")) {
        const dataStr = buffer.trim().slice(5).trim();
        if (dataStr && dataStr !== "[DONE]") {
          try {
            const parsed = JSON.parse(dataStr);
            const choice = parsed.choices?.[0];
            const chunk = choice?.delta?.content ?? "";
            if (chunk) {
              fullContent += chunk;
              await onChunk(chunk);
            }
            if (choice?.finish_reason) {
              finishReason = choice.finish_reason;
            }
          } catch {}
        }
      }

      return {
        content: fullContent,
        finishReason,
      };
    } catch (error: any) {
      return {
        error: error?.message || "Streaming failed",
      };
    } finally {
      reader.releaseLock();
    }
  };
}