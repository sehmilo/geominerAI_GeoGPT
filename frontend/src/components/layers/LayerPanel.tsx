import { useEffect } from "react";
import {
  FileText,
  FileSpreadsheet,
  Map,
  Image,
  Layers,
  Trash2,
  Globe,
  FileQuestion,
} from "lucide-react";
import { useLayerStore } from "../../stores/useLayerStore";
import { useApi } from "../../hooks/useApi";
import { LayerUpload } from "./LayerUpload";
import type { Layer } from "../../types/layer";

const ICONS: Record<string, typeof FileText> = {
  pdf: FileText,
  word: FileText,
  csv: FileSpreadsheet,
  geojson: Map,
  kml: Globe,
  raster: Globe,
  image: Image,
  vector: Layers,
  text: FileText,
  unknown: FileQuestion,
};

export function LayerPanel() {
  const layers = useLayerStore((s) => s.layers);
  const setLayers = useLayerStore((s) => s.setLayers);
  const removeLayer = useLayerStore((s) => s.removeLayer);
  const { getLayers, deleteLayer } = useApi();

  useEffect(() => {
    getLayers()
      .then((data) => setLayers(data as unknown as Layer[]))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleRemove(id: number) {
    try {
      await deleteLayer(id);
      removeLayer(id);
    } catch {
      // ignore
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-slate-700">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Layers
        </h2>
      </div>

      <div className="px-3 py-2 border-b border-slate-700">
        <LayerUpload />
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-1">
        {layers.length === 0 && (
          <p className="text-slate-500 text-xs px-1 py-4 text-center">
            No layers yet. Upload files above.
          </p>
        )}

        {layers.map((layer) => {
          const Icon = ICONS[layer.layer_type] || FileQuestion;
          return (
            <div
              key={layer.id}
              className="group flex items-start gap-2 px-2 py-2 rounded hover:bg-slate-800 text-sm"
            >
              <Icon size={16} className="text-slate-400 mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-slate-200 truncate text-xs font-medium">
                  {layer.name}
                </p>
                <p className="text-slate-500 text-xs">
                  {layer.layer_type}
                  {layer.row_count != null && ` - ${layer.row_count} rows`}
                  {layer.has_geodata && " - spatial"}
                </p>
              </div>
              <button
                onClick={() => handleRemove(layer.id)}
                className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 transition-opacity"
                title="Remove layer"
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
      </div>

      <div className="px-3 py-2 border-t border-slate-700 text-xs text-slate-500">
        {layers.length} layer(s)
      </div>
    </div>
  );
}
