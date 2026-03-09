import { useState } from "react";
import { useApi } from "./useApi";
import { useLayerStore } from "../stores/useLayerStore";
import type { Layer } from "../types/layer";

export function useFileUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const { uploadFile, getLayers } = useApi();
  const setLayers = useLayerStore((s) => s.setLayers);

  async function upload(files: FileList | File[]) {
    setIsUploading(true);
    setError(null);
    setProgress(0);

    const fileArray = Array.from(files);
    try {
      for (let i = 0; i < fileArray.length; i++) {
        await uploadFile(fileArray[i]);
        setProgress(((i + 1) / fileArray.length) * 100);
      }
      // Refresh layers
      const layers = await getLayers();
      setLayers(layers as unknown as Layer[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  return { upload, isUploading, progress, error };
}
