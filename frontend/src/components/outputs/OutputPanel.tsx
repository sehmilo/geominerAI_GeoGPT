import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useOutputStore } from "../../stores/useOutputStore";
import { TextOutput } from "./TextOutput";
import { FigureOutput } from "./FigureOutput";
import { DataFrameOutput } from "./DataFrameOutput";
import type { AnalysisOutput } from "../../types/analysis";

export function OutputPanel() {
  const outputs = useOutputStore((s) => s.outputs);

  if (outputs.length === 0) {
    return (
      <div className="px-4 py-4 text-sm text-gray-400 text-center">
        Analysis outputs appear here. Try: run hotspot analysis | draw cross
        section | run prospectivity analysis
      </div>
    );
  }

  return (
    <div className="px-3 py-2 space-y-1">
      {outputs.map((output, i) => (
        <OutputCard key={i} output={output} defaultOpen={i === 0} />
      ))}
    </div>
  );
}

function OutputCard({
  output,
  defaultOpen,
}: {
  output: AnalysisOutput;
  defaultOpen: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-gray-200 rounded">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition-colors"
      >
        {isOpen ? (
          <ChevronDown size={14} className="text-gray-400" />
        ) : (
          <ChevronRight size={14} className="text-gray-400" />
        )}
        <span className="text-xs text-gray-500">[{output.timestamp}]</span>
        <span className="text-sm font-medium text-gray-700 truncate">
          {output.title}
        </span>
      </button>

      {isOpen && (
        <div className="px-3 pb-3 border-t border-gray-100">
          {output.output_type === "text" && (
            <TextOutput content={String(output.content || "")} />
          )}
          {output.output_type === "figure" && output.figure_url && (
            <FigureOutput url={output.figure_url} title={output.title} />
          )}
          {output.output_type === "dataframe" && (
            <DataFrameOutput data={output.content} title={output.title} />
          )}
          {output.output_type === "geojson" && (
            <TextOutput
              content={`\`\`\`json\n${JSON.stringify(output.content, null, 2).slice(0, 2000)}\n\`\`\``}
            />
          )}
        </div>
      )}
    </div>
  );
}
