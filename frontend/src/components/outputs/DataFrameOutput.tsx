import { useMemo } from "react";
import { Download } from "lucide-react";

interface Props {
  data: unknown;
  title: string;
}

export function DataFrameOutput({ data, title }: Props) {
  const rows = useMemo(() => {
    if (Array.isArray(data)) return data as Record<string, unknown>[];
    return [];
  }, [data]);

  const columns = useMemo(() => {
    if (rows.length === 0) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  function downloadCSV() {
    if (rows.length === 0) return;
    const header = columns.join(",");
    const body = rows
      .map((row) => columns.map((c) => String(row[c] ?? "")).join(","))
      .join("\n");
    const csv = `${header}\n${body}`;
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/\s+/g, "_")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (rows.length === 0) {
    return <p className="text-sm text-gray-400 py-2">No data to display</p>;
  }

  return (
    <div className="py-2">
      <div className="overflow-x-auto max-h-64 overflow-y-auto border border-gray-200 rounded">
        <table className="w-full text-xs">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-2 py-1.5 text-left font-medium text-gray-600 border-b border-gray-200"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 100).map((row, i) => (
              <tr key={i} className="hover:bg-gray-50">
                {columns.map((col) => (
                  <td key={col} className="px-2 py-1 text-gray-700 border-b border-gray-100">
                    {String(row[col] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > 100 && (
        <p className="text-xs text-gray-400 mt-1">
          Showing 100 of {rows.length} rows
        </p>
      )}
      <button
        onClick={downloadCSV}
        className="mt-2 flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-600 transition-colors"
      >
        <Download size={12} />
        Download CSV
      </button>
    </div>
  );
}
