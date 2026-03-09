import { useState } from "react";
import { Settings, X } from "lucide-react";
import { useSessionStore } from "../../stores/useSessionStore";

const MODELS = [
  "HuggingFaceH4/zephyr-7b-beta",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "microsoft/phi-2",
  "google/gemma-2b-it",
  "tiiuae/falcon-7b-instruct",
];

export function SettingsDrawer() {
  const [isOpen, setIsOpen] = useState(false);
  const { settings, updateSettings } = useSessionStore();

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="p-2 text-slate-400 hover:text-white transition-colors"
        title="Settings"
      >
        <Settings size={18} />
      </button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="bg-black/40 flex-1" onClick={() => setIsOpen(false)} />
      <div className="w-80 bg-white shadow-xl flex flex-col">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h3 className="font-semibold text-gray-800">Settings</h3>
          <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-gray-600">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              LLM Model
            </label>
            <select
              value={settings.hfModel}
              onChange={(e) => updateSettings({ hfModel: e.target.value })}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-400 focus:outline-none"
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>
                  {m.split("/")[1]}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Provider
            </label>
            <select
              value={settings.provider}
              onChange={(e) => updateSettings({ provider: e.target.value })}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-400 focus:outline-none"
            >
              <option value="auto">auto</option>
              <option value="hf-inference">hf-inference</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Map Center Latitude
            </label>
            <input
              type="number"
              step="0.1"
              value={settings.mapCenter[0]}
              onChange={(e) =>
                updateSettings({
                  mapCenter: [Number(e.target.value), settings.mapCenter[1]],
                })
              }
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-400 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Map Center Longitude
            </label>
            <input
              type="number"
              step="0.1"
              value={settings.mapCenter[1]}
              onChange={(e) =>
                updateSettings({
                  mapCenter: [settings.mapCenter[0], Number(e.target.value)],
                })
              }
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-400 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Default Zoom
            </label>
            <input
              type="range"
              min={2}
              max={18}
              value={settings.mapZoom}
              onChange={(e) =>
                updateSettings({ mapZoom: Number(e.target.value) })
              }
              className="w-full"
            />
            <span className="text-xs text-gray-500">{settings.mapZoom}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
