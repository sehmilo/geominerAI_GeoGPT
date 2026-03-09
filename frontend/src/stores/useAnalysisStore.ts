import { create } from "zustand";

interface AnalysisState {
  isLoading: boolean;
  activeIntent: string | null;
  taskId: string | null;
  setLoading: (loading: boolean) => void;
  setActiveIntent: (intent: string | null) => void;
  setTaskId: (id: string | null) => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  isLoading: false,
  activeIntent: null,
  taskId: null,
  setLoading: (loading) => set({ isLoading: loading }),
  setActiveIntent: (intent) => set({ activeIntent: intent }),
  setTaskId: (id) => set({ taskId: id }),
}));
