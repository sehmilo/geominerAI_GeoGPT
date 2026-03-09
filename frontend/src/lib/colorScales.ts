export function getClusterColor(cluster: string): string {
  const c = (cluster || "").toUpperCase();
  if (["HH", "HOT", "HIGH"].some((k) => c.includes(k))) return "#ef4444";
  if (["LL", "COLD", "LOW"].some((k) => c.includes(k))) return "#3b82f6";
  if (c.includes("MED")) return "#f97316";
  if (c.includes("OUTLIER")) return "#a855f7";
  return "#6b7280";
}

export function getClusterRgba(
  cluster: string,
  alpha = 200
): [number, number, number, number] {
  const c = (cluster || "").toUpperCase();
  if (["HH", "HOT", "HIGH"].some((k) => c.includes(k)))
    return [239, 68, 68, alpha];
  if (["LL", "COLD", "LOW"].some((k) => c.includes(k)))
    return [59, 130, 246, alpha];
  if (c.includes("MED")) return [249, 115, 22, alpha];
  if (c.includes("OUTLIER")) return [168, 85, 247, alpha];
  return [107, 114, 128, alpha];
}
