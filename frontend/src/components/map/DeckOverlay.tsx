/**
 * Deck.gl overlay for rendering large point datasets (1000+).
 * Used when CSV layers exceed the threshold for efficient GeoJSON circle rendering.
 *
 * Phase 2 implementation — integrated into MapContainer when point count > 1000.
 */

import { ScatterplotLayer } from "@deck.gl/layers";
import { getClusterRgba } from "../../lib/colorScales";

interface PointData {
  lat: number;
  lon: number;
  cluster?: string;
  hole_id?: string;
  [key: string]: unknown;
}

export function createScatterplotLayer(
  id: string,
  data: PointData[],
  options: { radiusPixels?: number; opacity?: number } = {}
) {
  return new ScatterplotLayer({
    id,
    data,
    pickable: true,
    getPosition: (d: PointData) => [d.lon, d.lat],
    getRadius: options.radiusPixels || 50,
    radiusUnits: "meters" as const,
    getFillColor: (d: PointData) => getClusterRgba(d.cluster || ""),
    opacity: options.opacity || 0.85,
    stroked: true,
    getLineColor: [255, 255, 255, 180],
    lineWidthMinPixels: 1,
  });
}
