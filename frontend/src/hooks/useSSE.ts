import { useEffect, useRef, useState } from "react";
import { useChatStore } from "../stores/useChatStore";

interface UseSSEReturn {
  data: string;
  isConnected: boolean;
  error: string | null;
}

export function useSSE(url: string | null): UseSSEReturn {
  const [data, setData] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const updateLastMessage = useChatStore((s) => s.updateLastMessage);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!url) return;

    const source = new EventSource(url);
    sourceRef.current = source;
    let accumulated = "";

    source.onopen = () => setIsConnected(true);

    source.onmessage = (event) => {
      if (event.data === "[DONE]") {
        source.close();
        setIsConnected(false);
        return;
      }
      accumulated += event.data;
      setData(accumulated);
      updateLastMessage(accumulated);
    };

    source.onerror = () => {
      setError("SSE connection error");
      setIsConnected(false);
      source.close();
    };

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [url, updateLastMessage]);

  return { data, isConnected, error };
}
