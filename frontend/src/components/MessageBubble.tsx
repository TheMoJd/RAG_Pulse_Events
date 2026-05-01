import ReactMarkdown from "react-markdown";
import type { Source } from "../api";
import EventCard from "./EventCard";

type Props = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

export default function MessageBubble({ role, content, sources }: Props) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-2xl ${isUser ? "items-end" : "items-start"} flex flex-col gap-3`}>
        <div
          className={`px-5 py-3 rounded-2xl shadow-sm leading-relaxed ${
            isUser
              ? "bg-brand-600 text-white rounded-br-sm whitespace-pre-wrap"
              : "bg-white text-slate-800 rounded-bl-sm border border-slate-200"
          }`}
        >
          {isUser ? (
            content
          ) : (
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
                ul: ({ children }) => <ul className="list-disc pl-5 my-2 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 my-2 space-y-1">{children}</ol>,
                li: ({ children }) => <li>{children}</li>,
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-brand-600 hover:underline break-all"
                  >
                    {children}
                  </a>
                ),
                code: ({ children }) => (
                  <code className="bg-slate-100 text-slate-800 px-1 py-0.5 rounded text-sm">
                    {children}
                  </code>
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          )}
        </div>
        {!isUser && sources && sources.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
            {sources.map((s, i) => (
              <EventCard key={i} source={s} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
