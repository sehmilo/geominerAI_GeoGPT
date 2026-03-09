import { create } from "zustand";

interface SessionState {
  sessionId: string;
  settings: {
    hfModel: string;
    provider: string;
    mapCenter: [number, number];
    mapZoom: number;
  };
  updateSettings: (partial: Partial<SessionState["settings"]>) => void;
}

function getOrCreateSessionId(): string {
  const stored = localStorage.getItem("geominerai_session_id");
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem("geominerai_session_id", id);
  return id;
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: getOrCreateSessionId(),
  settings: {
    hfModel: "HuggingFaceH4/zephyr-7b-beta",
    provider: "auto",
    mapCenter: [9.9, 8.9],
    mapZoom: 8,
  },
  updateSettings: (partial) =>
    set((state) => ({
      settings: { ...state.settings, ...partial },
    })),
}));
