const replacementPackage = "@adminforth/completion-adapter-openai-responses";
const replacementUrl = "https://www.npmjs.com/package/@adminforth/completion-adapter-openai-responses";

console.warn(
  [
    "",
    "[adminforth] @adminforth/completion-adapter-open-ai-chat-gpt is deprecated.",
    `Install ${replacementPackage} instead.`,
    `${replacementPackage} is fully compatible and replaces this package.`,
    `More info: ${replacementUrl}`,
    "",
  ].join("\n"),
);