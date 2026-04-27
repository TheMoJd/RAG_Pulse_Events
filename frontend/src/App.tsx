import { useEffect, useState } from "react";
import ChatBox from "./components/ChatBox";
import { getHealth } from "./api";

export default function App() {
  const [indexSize, setIndexSize] = useState<number | null>(null);
  const [model, setModel] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setIndexSize(h.index_size);
        setModel(h.model);
      })
      .catch(() => {
        setIndexSize(0);
      });
  }, []);

  return (
    <div className="h-full flex flex-col">
      <header className="bg-white/70 backdrop-blur border-b border-slate-200 px-4 sm:px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white text-lg font-bold shadow-sm">
              P
            </div>
            <div>
              <h1 className="font-bold text-lg text-slate-900 leading-tight">
                Puls-Events
              </h1>
              <p className="text-xs text-slate-500">Assistant culturel — Brest</p>
            </div>
          </div>
          <div className="text-right text-xs text-slate-500">
            {indexSize !== null && (
              <span className="inline-flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    indexSize > 0 ? "bg-green-500" : "bg-red-500"
                  }`}
                />
                {indexSize > 0 ? `${indexSize} chunks indexés` : "Index vide"}
              </span>
            )}
            {model && <p className="mt-0.5 hidden sm:block">{model}</p>}
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <ChatBox />
      </main>
    </div>
  );
}
