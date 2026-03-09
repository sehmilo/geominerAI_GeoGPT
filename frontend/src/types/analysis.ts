export interface AnalysisOutput {
  id: number;
  title: string;
  output_type: "text" | "figure" | "dataframe" | "geojson";
  content: any;
  figure_url?: string;
  timestamp: string;
}

export interface DispatchResponse {
  task_id: string;
  intent: string;
  metadata: Record<string, any>;
  outputs: AnalysisOutput[];
}
