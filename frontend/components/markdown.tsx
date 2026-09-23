"use client";

import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { isValidElement } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "@/components/agents/code-block";
import { mapLanguage } from "@/lib/chat-api";

function flatten(node: ReactNode): string {
  if (node == null || node === false) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(flatten).join("");
  if (isValidElement(node)) {
    return flatten((node.props as { children?: ReactNode }).children);
  }
  return "";
}

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Fenced code blocks arrive wrapped in <pre><code>. Render those with
        // the rich CodeBlock; leave inline <code> to the surrounding styles.
        pre({ children }: ComponentPropsWithoutRef<"pre">) {
          const child = Array.isArray(children) ? children[0] : children;
          if (!isValidElement(child)) {
            return <pre>{children}</pre>;
          }
          const props = child.props as {
            className?: string;
            children?: ReactNode;
          };
          const match = /language-([\w+#-]+)/.exec(props.className ?? "");
          const raw = flatten(props.children).replace(/\n$/, "");
          const lineCount = raw.split("\n").length;
          return (
            <CodeBlock
              code={raw}
              language={mapLanguage(match?.[1])}
              filename={match?.[1] ? match[1].toLowerCase() : undefined}
              showLineNumbers={lineCount > 1}
              maxHeight={460}
              className="my-3"
            />
          );
        },
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
