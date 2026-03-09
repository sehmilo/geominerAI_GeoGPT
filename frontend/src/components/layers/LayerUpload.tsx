import { useRef } from "react";
import { Upload, Loader } from "lucide-react";
import { useFileUpload } from "../../hooks/useFileUpload";

const ACCEPTED_TYPES = [
  ".pdf", ".docx", ".doc", ".csv",
  ".geojson", ".json", ".kml",
  ".tif", ".tiff", ".dem", ".asc",
  ".png", ".jpg", ".jpeg", ".bmp",
  ".shp", ".gpkg", ".txt", ".md",
].join(",");

export function LayerUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { upload, isUploading, progress, error } = useFileUpload();

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      upload(e.target.files);
      e.target.value = "";
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      upload(e.dataTransfer.files);
    }
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
      className="border border-dashed border-slate-600 rounded px-3 py-3 text-center cursor-pointer hover:border-slate-400 transition-colors"
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        multiple
        className="hidden"
        onChange={handleChange}
      />
      {isUploading ? (
        <div className="flex items-center justify-center gap-2 text-slate-400 text-xs">
          <Loader size={14} className="animate-spin" />
          <span>Uploading... {Math.round(progress)}%</span>
        </div>
      ) : (
        <div className="flex items-center justify-center gap-2 text-slate-400 text-xs">
          <Upload size={14} />
          <span>Drop files or click to upload</span>
        </div>
      )}
      {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
    </div>
  );
}
