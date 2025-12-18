const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.trim() || "http://127.0.0.1:8000";

export async function apiPost<T = any>(path: string, body: any): Promise<T> {
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      msg = j?.detail || j?.error || JSON.stringify(j);
    } catch {
      try {
        msg = await res.text();
      } catch {}
    }
    throw new Error(msg);
  }

  return (await res.json()) as T;
}

export async function apiGet<T = any>(path: string): Promise<T> {
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url);

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(txt || `Request failed (${res.status})`);
  }

  return (await res.json()) as T;
}
