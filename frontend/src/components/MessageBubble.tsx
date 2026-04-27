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
          className={`px-5 py-3 rounded-2xl shadow-sm whitespace-pre-wrap leading-relaxed ${
            isUser
              ? "bg-brand-600 text-white rounded-br-sm"
              : "bg-white text-slate-800 rounded-bl-sm border border-slate-200"
          }`}
        >
          {content}
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
