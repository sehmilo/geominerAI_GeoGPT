import { create } from "zustand";

interface MapState {
  center: [number, number];
  zoom: number;
  drawnFeatures: GeoJSON.FeatureCollection;
  selectedHole: string | null;
  setCenter: (center: [number, number]) => void;
  setZoom: (zoom: number) => void;
  setDrawnFeatures: (fc: GeoJSON.FeatureCollection) => void;
  setSelectedHole: (hole: string | null) => void;
}

export const useMapStore = create<MapState>((set) => ({
  center: [9.9, 8.9],
  zoom: 8,
  drawnFeatures: { type: "FeatureCollection", features: [] },
  selectedHole: null,
  setCenter: (center) => set({ center }),
  setZoom: (zoom) => set({ zoom }),
  setDrawnFeatures: (fc) => set({ drawnFeatures: fc }),
  setSelectedHole: (hole) => set({ selectedHole: hole }),
}));
