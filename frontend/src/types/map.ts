export interface MapViewState {
  center: [number, number];
  zoom: number;
}

export interface DrawnFeature {
  type: string;
  geometry: any;
  properties: Record<string, any>;
}
