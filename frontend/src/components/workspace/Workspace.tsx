import { LayerPanel } from "../layers/LayerPanel";
import { MapContainer } from "../map/MapContainer";
import { ChatPanel } from "../chat/ChatPanel";
import { OutputPanel } from "../outputs/OutputPanel";

export function Workspace() {
  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-slate-800 text-white px-4 py-2 flex items-center gap-3 shrink-0">
        <span className="text-xl font-bold">GeoMinerAI</span>
        <span className="text-slate-400 text-sm hidden sm:inline">
          RAG - Hotspot Analysis - ML Prospectivity - Cross-Section -
          Geoprocessing
        </span>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar — Layers */}
        <aside className="w-72 bg-slate-900 text-white flex flex-col shrink-0 border-r border-slate-700">
          <LayerPanel />
        </aside>

        {/* Right area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Output panel */}
          <div className="h-1/4 min-h-[140px] overflow-y-auto border-b border-gray-300 bg-white">
            <OutputPanel />
          </div>

          {/* Map */}
          <div className="flex-1 min-h-[200px]">
            <MapContainer />
          </div>

          {/* Chat */}
          <div className="h-1/4 min-h-[160px] border-t border-gray-300 bg-white">
            <ChatPanel />
          </div>
        </main>
      </div>
    </div>
  );
}
