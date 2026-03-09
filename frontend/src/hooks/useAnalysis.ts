import { useApi } from "./useApi";
import { useChatStore } from "../stores/useChatStore";
import { useOutputStore } from "../stores/useOutputStore";
import { useAnalysisStore } from "../stores/useAnalysisStore";
import type { AnalysisOutput } from "../types/analysis";

export function useAnalysis() {
  const { dispatch: apiDispatch } = useApi();
  const addMessage = useChatStore((s) => s.addMessage);
  const prependOutput = useOutputStore((s) => s.prependOutput);
  const { setLoading, setActiveIntent } = useAnalysisStore();

  async function dispatch(prompt: string) {
    if (!prompt.trim()) return;

    addMessage({ role: "user", content: prompt });
    setLoading(true);
    setActiveIntent(null);

    try {
      const result = await apiDispatch(prompt);
      setActiveIntent(result.intent);

      // Add outputs
      for (const out of result.outputs) {
        prependOutput(out as AnalysisOutput);
      }

      addMessage({
        role: "assistant",
        content: `Done — **${result.intent}** analysis complete.`,
      });
    } catch (e) {
      addMessage({
        role: "assistant",
        content: `Error: ${e instanceof Error ? e.message : "Analysis failed"}`,
      });
    } finally {
      setLoading(false);
    }
  }

  return { dispatch };
}
