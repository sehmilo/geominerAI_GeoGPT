export interface Layer {
  id: number;
  name: string;
  layer_type: string;
  metadata: Record<string, any>;
  created_at: string;
  row_count?: number;
  has_geodata?: boolean;
}
