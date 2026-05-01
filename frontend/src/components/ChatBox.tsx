import { useEffect, useRef, useState } from "react";
import type { Source } from "../api";
import { ask } from "../api";
import MessageBubble from "./MessageBubble";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const SUGGESTIONS = [
  "Quels concerts à voir prochainement à Brest ?",
  "Quelles expositions à voir prochainement à Brest ?",
  "Un spectacle pour enfants à venir ?",
  "Festivals à venir à Brest ?",
];

const WELCOME: Message = {
  role: "assistant",
  content:
    "Bonjour ! Je suis l'assistant culturel de Puls-Events pour la ville de Brest 👋\n\nPose-moi une question sur les événements à venir : concerts, expositions, spectacles, festivals…",
};

export default function ChatBox() {
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(question: string) {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setMessages((m) => [...m, { role: "user", content: trimmed }]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await ask(trimmed);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, sources: res.sources },
      ]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Erreur inconnue";
      setError(msg);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Désolé, une erreur s'est produite : ${msg}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        <div className="max-w-3xl mx-auto">
          {messages.map((m, i) => (
            <MessageBubble key={i} {...m} />
          ))}

          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-5 py-3 shadow-sm">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-brand-400 rounded-full animate-bounce" />
                  <span
                    className="w-2 h-2 bg-brand-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.15s" }}
                  />
                  <span
                    className="w-2 h-2 bg-brand-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.3s" }}
                  />
                </div>
              </div>
            </div>
          )}

          {messages.length === 1 && !loading && (
            <div className="mt-6">
              <p className="text-xs uppercase font-semibold text-slate-500 tracking-wider mb-3">
                Suggestions
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="text-left text-sm px-4 py-3 bg-white border border-slate-200 rounded-xl hover:border-brand-400 hover:bg-brand-50 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-slate-200 bg-white/70 backdrop-blur px-4 sm:px-6 py-4">
        <form
          className="max-w-3xl mx-auto flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Posez votre question..."
            disabled={loading}
            className="flex-1 px-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent disabled:opacity-50 bg-white"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-6 py-3 bg-brand-600 hover:bg-brand-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors shadow-sm"
          >
            {loading ? "..." : "Envoyer"}
          </button>
        </form>
        {error && (
          <p className="max-w-3xl mx-auto mt-2 text-xs text-red-600">{error}</p>
        )}
      </div>
    </div>
  );
}
