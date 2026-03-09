import { useSessionStore } from "../stores/useSessionStore";
import { api } from "../lib/api";

export function useApi() {
  const sessionId = useSessionStore((s) => s.sessionId);

  return {
    uploadFile: (file: File) => api.uploadFile(sessionId, file),
    getLayers: () => api.getLayers(sessionId),
    deleteLayer: (id: number) => api.deleteLayer(sessionId, id),
    dispatch: (prompt: string) => api.dispatch(sessionId, prompt),
    getChatHistory: () => api.getChatHistory(sessionId),
    sendMessage: (role: string, content: string) =>
      api.sendMessage(sessionId, role, content),
    getLayerData: api.getLayerData,
    getLayerGeoJSON: api.getLayerGeoJSON,
  };
}
