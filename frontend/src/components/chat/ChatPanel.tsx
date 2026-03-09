import { useState } from "react";
import { Send, Flame, Ruler, Gem, Maximize2, Loader } from "lucide-react";
import { useChatStore } from "../../stores/useChatStore";
import { useAnalysisStore } from "../../stores/useAnalysisStore";
import { useAnalysis } from "../../hooks/useAnalysis";
import { ChatMessage } from "./ChatMessage";

const QUICK_COMMANDS = [
  { label: "Hotspot", icon: Flame, prompt: "Run hotspot analysis on the uploaded CSV" },
  { label: "Cross-Section", icon: Ruler, prompt: "Draw a geological cross section A to A'" },
  { label: "Prospectivity", icon: Gem, prompt: "Run Sn-REE prospectivity analysis" },
  { label: "Buffer 500m", icon: Maximize2, prompt: "Buffer drawn polygon by 500 m" },
];

export function ChatPanel() {
  const [input, setInput] = useState("");
  const messages = useChatStore((s) => s.messages);
  const isLoading = useAnalysisStore((s) => s.isLoading);
  const { dispatch } = useAnalysis();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    dispatch(input.trim());
    setInput("");
  }

  function handleQuickCommand(prompt: string) {
    if (isLoading) return;
    dispatch(prompt);
  }

  const visibleMessages = messages.slice(-8);

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2">
        {visibleMessages.length === 0 && (
          <p className="text-gray-400 text-sm text-center py-4">
            Ask a question or use a quick command below
          </p>
        )}
        {visibleMessages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}
        {isLoading && (
          <div className="flex items-center gap-2 text-sm text-gray-400 px-2">
            <Loader size={14} className="animate-spin" />
            Analyzing...
          </div>
        )}
      </div>

      {/* Quick commands */}
      <div className="flex gap-1 px-4 py-1">
        {QUICK_COMMANDS.map(({ label, icon: Icon, prompt }) => (
          <button
            key={label}
            onClick={() => handleQuickCommand(prompt)}
            disabled={isLoading}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-600 disabled:opacity-50 transition-colors"
          >
            <Icon size={12} />
            {label}
          </button>
        ))}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2 px-4 py-2 border-t border-gray-200">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask or command... (e.g. run hotspot on SnO2, draw cross section A-A')"
          className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
