import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import { osmStyle } from "../../lib/mapStyles";
import { useMapStore } from "../../stores/useMapStore";
import { useLayerStore } from "../../stores/useLayerStore";
import { getClusterColor } from "../../lib/colorScales";
import { useApi } from "../../hooks/useApi";

export function MapContainer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const drawRef = useRef<MapboxDraw | null>(null);

  const { center, zoom, setDrawnFeatures } = useMapStore();
  const layers = useLayerStore((s) => s.layers);
  const { getLayerData, getLayerGeoJSON } = useApi();

  // Initialize map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: osmStyle,
      center: [center[1], center[0]], // MapLibre uses [lng, lat]
      zoom,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(
      new maplibregl.ScaleControl({ maxWidth: 200 }),
      "bottom-right"
    );

    // Draw tools
    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: {
        polygon: true,
        line_string: true,
        point: true,
        trash: true,
      },
    }) as unknown as maplibregl.IControl & MapboxDraw;

    map.addControl(draw, "top-left");
    drawRef.current = draw;

    // Capture drawings
    const updateDrawn = () => {
      const all = draw.getAll();
      if (all && all.features.length > 0) {
        setDrawnFeatures(all as GeoJSON.FeatureCollection);
      }
    };
    map.on("draw.create", updateDrawn);
    map.on("draw.update", updateDrawn);
    map.on("draw.delete", updateDrawn);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update layers on the map
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    // Remove existing dynamic sources/layers
    const style = map.getStyle();
    if (style?.layers) {
      for (const layer of style.layers) {
        if (
          typeof layer.id === "string" &&
          layer.id.startsWith("geominerai-")
        ) {
          map.removeLayer(layer.id);
        }
      }
    }
    if (style?.sources) {
      for (const sourceId of Object.keys(style.sources)) {
        if (sourceId.startsWith("geominerai-")) {
          map.removeSource(sourceId);
        }
      }
    }

    // Add CSV point layers and GeoJSON layers
    for (const layer of layers) {
      const sourceId = `geominerai-${layer.id}`;

      if (layer.layer_type === "csv" && layer.row_count) {
        // Fetch data and add points
        getLayerData(layer.id)
          .then((data) => {
            if (!mapRef.current) return;
            const features = data.data
              .filter(
                (r: Record<string, unknown>) =>
                  r.lat != null && r.lon != null
              )
              .map((r: Record<string, unknown>) => ({
                type: "Feature" as const,
                geometry: {
                  type: "Point" as const,
                  coordinates: [Number(r.lon), Number(r.lat)],
                },
                properties: {
                  ...r,
                  _color: getClusterColor(String(r.cluster || "")),
                },
              }));

            const fc: GeoJSON.FeatureCollection = {
              type: "FeatureCollection",
              features,
            };

            if (!mapRef.current?.getSource(sourceId)) {
              mapRef.current?.addSource(sourceId, {
                type: "geojson",
                data: fc,
              });
              mapRef.current?.addLayer({
                id: `${sourceId}-points`,
                type: "circle",
                source: sourceId,
                paint: {
                  "circle-radius": 5,
                  "circle-color": ["get", "_color"],
                  "circle-stroke-width": 1,
                  "circle-stroke-color": "#ffffff",
                  "circle-opacity": 0.85,
                },
              });
            }
          })
          .catch(() => {});
      } else if (layer.has_geodata) {
        getLayerGeoJSON(layer.id)
          .then((geojson) => {
            if (!mapRef.current?.getSource(sourceId)) {
              mapRef.current?.addSource(sourceId, {
                type: "geojson",
                data: geojson,
              });
              mapRef.current?.addLayer({
                id: `${sourceId}-fill`,
                type: "fill",
                source: sourceId,
                paint: {
                  "fill-color": "#8b5cf6",
                  "fill-opacity": 0.15,
                },
              });
              mapRef.current?.addLayer({
                id: `${sourceId}-line`,
                type: "line",
                source: sourceId,
                paint: {
                  "line-color": "#8b5cf6",
                  "line-width": 2,
                },
              });
            }
          })
          .catch(() => {});
      }
    }
  }, [layers, getLayerData, getLayerGeoJSON]);

  return <div ref={containerRef} className="w-full h-full" />;
}
