const BASE_URL = import.meta.env.VITE_API_URL || "";

async function request<T>(
  path: string,
  options: RequestInit = {},
  sessionId?: string
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (sessionId) {
    headers["X-Session-Id"] = sessionId;
  }
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = headers["Content-Type"] || "application/json";
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  createSession: () =>
    request<{ session_id: string }>("/api/session/", { method: "POST" }),

  uploadFile: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Record<string, unknown>>(
      "/api/layers/upload",
      { method: "POST", body: form },
      sessionId
    );
  },

  getLayers: (sessionId: string) =>
    request<Record<string, unknown>[]>("/api/layers/", {}, sessionId),

  deleteLayer: (sessionId: string, layerId: number) =>
    request<{ deleted: boolean }>(
      `/api/layers/${layerId}`,
      { method: "DELETE" },
      sessionId
    ),

  getLayerData: (layerId: number) =>
    request<{ columns: string[]; data: Record<string, unknown>[] }>(
      `/api/layers/${layerId}/data`
    ),

  getLayerGeoJSON: (layerId: number) =>
    request<GeoJSON.FeatureCollection>(`/api/layers/${layerId}/geojson`),

  dispatch: (sessionId: string, prompt: string) =>
    request<{
      intent: string;
      metadata: Record<string, unknown>;
      outputs: Array<{
        title: string;
        output_type: string;
        content: unknown;
        figure_url?: string;
        timestamp: string;
      }>;
    }>(
      "/api/analysis/dispatch",
      {
        method: "POST",
        body: JSON.stringify({ prompt, session_id: sessionId }),
      }
    ),

  getChatHistory: (sessionId: string) =>
    request<
      Array<{
        id: number;
        role: string;
        content: string;
        created_at: string;
      }>
    >("/api/chat/", {}, sessionId),

  sendMessage: (sessionId: string, role: string, content: string) =>
    request<Record<string, unknown>>("/api/chat/", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, role, content }),
    }),
};
