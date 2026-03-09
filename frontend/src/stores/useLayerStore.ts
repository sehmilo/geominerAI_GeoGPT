import { create } from "zustand";
import type { Layer } from "../types/layer";

interface LayerState {
  layers: Layer[];
  selectedLayerId: number | null;
  setLayers: (layers: Layer[]) => void;
  addLayer: (layer: Layer) => void;
  removeLayer: (id: number) => void;
  selectLayer: (id: number | null) => void;
}

export const useLayerStore = create<LayerState>((set) => ({
  layers: [],
  selectedLayerId: null,
  setLayers: (layers) => set({ layers }),
  addLayer: (layer) =>
    set((state) => ({ layers: [layer, ...state.layers] })),
  removeLayer: (id) =>
    set((state) => ({
      layers: state.layers.filter((l) => l.id !== id),
      selectedLayerId:
        state.selectedLayerId === id ? null : state.selectedLayerId,
    })),
  selectLayer: (id) => set({ selectedLayerId: id }),
}));
