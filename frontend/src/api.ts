export type Source = {
  title: string | null;
  url: string | null;
  daterange: string | null;
  location_name: string | null;
  image: string | null;
  score: number;
};

export type AskResponse = {
  answer: string;
  sources: Source[];
};

export type HealthResponse = {
  status: string;
  index_size: number;
  model: string;
};

const API_BASE = "/api";

export async function ask(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
