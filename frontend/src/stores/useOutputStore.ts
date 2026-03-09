import { create } from "zustand";
import type { AnalysisOutput } from "../types/analysis";

interface OutputState {
  outputs: AnalysisOutput[];
  prependOutput: (output: AnalysisOutput) => void;
  setOutputs: (outputs: AnalysisOutput[]) => void;
  clearOutputs: () => void;
}

export const useOutputStore = create<OutputState>((set) => ({
  outputs: [],
  prependOutput: (output) =>
    set((state) => ({ outputs: [output, ...state.outputs] })),
  setOutputs: (outputs) => set({ outputs }),
  clearOutputs: () => set({ outputs: [] }),
}));
