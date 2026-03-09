import { Download } from "lucide-react";

interface Props {
  url: string;
  filename: string;
  label?: string;
}

export function DownloadButton({ url, filename, label = "Download" }: Props) {
  return (
    <a
      href={url}
      download={filename}
      className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-600 transition-colors"
    >
      <Download size={12} />
      {label}
    </a>
  );
}
