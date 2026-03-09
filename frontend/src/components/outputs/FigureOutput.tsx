import { Download } from "lucide-react";
import { DownloadButton } from "./DownloadButton";

interface Props {
  url: string;
  title: string;
}

export function FigureOutput({ url, title }: Props) {
  const fullUrl = url.startsWith("http") ? url : `${window.location.origin}${url}`;

  return (
    <div className="py-2">
      <img
        src={fullUrl}
        alt={title}
        className="max-w-full rounded border border-gray-200"
      />
      <div className="mt-2">
        <DownloadButton url={fullUrl} filename={`${title.replace(/\s+/g, "_")}.png`} label="Download PNG" />
      </div>
    </div>
  );
}
