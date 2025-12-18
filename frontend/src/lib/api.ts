export function getApiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
}

export async function apiPost<T = any>(
  path: string,
  body?: unknown
): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }

  return res.json();
}
