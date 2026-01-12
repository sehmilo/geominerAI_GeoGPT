import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, Any

EPS = 1e-9

def latlon_to_xy_m(df: pd.DataFrame, lat_col="lat", lon_col="lon") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert lat/lon degrees to local meters using an equirectangular approximation.
    Good enough for local-scale hotspot mapping.
    """
    lat0 = float(pd.to_numeric(df[lat_col], errors="coerce").dropna().mean())
    lon0 = float(pd.to_numeric(df[lon_col], errors="coerce").dropna().mean())

    lat = np.deg2rad(pd.to_numeric(df[lat_col], errors="coerce").values.astype(float))
    lon = np.deg2rad(pd.to_numeric(df[lon_col], errors="coerce").values.astype(float))
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)

    # meters per radian
    R = 6371000.0
    x = (lon - lon0r) * np.cos(lat0r) * R
    y = (lat - lat0r) * R
    return np.vstack([x, y]).T, {"lat0": lat0, "lon0": lon0}


def _knn_weights(X: np.ndarray, k: int = 8) -> np.ndarray:
    n = X.shape[0]
    k2 = min(max(2, k), max(2, n - 1))
    nn = NearestNeighbors(n_neighbors=k2 + 1, metric="euclidean")
    nn.fit(X)
    d, idx = nn.kneighbors(X)
    # drop self
    d = d[:, 1:]
    idx = idx[:, 1:]
    # inverse distance weights
    w = 1.0 / (d + 1.0)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, idx[i]] = w[i]
    # row standardize
    row_sums = W.sum(axis=1, keepdims=True) + EPS
    W = W / row_sums
    return W


def getis_ord_gistar(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Lightweight Gi* style z-score approximation.
    This is a practical implementation for exploration targeting, not a full GIS clone.
    """
    x = values.astype(float)
    x_mean = np.nanmean(x)
    x_std = np.nanstd(x) + EPS
    z = (x - x_mean) / x_std
    g = W.dot(z)
    # standardize again
    return (g - np.mean(g)) / (np.std(g) + EPS)


def local_morans_i(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Lightweight Local Moran's I approximation with row-standardized weights.
    """
    x = values.astype(float)
    x_mean = np.nanmean(x)
    x_dev = x - x_mean
    m2 = np.nanmean(x_dev ** 2) + EPS
    Ii = (x_dev / m2) * (W.dot(x_dev))
    # standardize
    return (Ii - np.mean(Ii)) / (np.std(Ii) + EPS)


def grid_binning(df: pd.DataFrame, value_col: str, cell_m: float = 250.0) -> pd.DataFrame:
    X, _ = latlon_to_xy_m(df)
    x = X[:, 0]
    y = X[:, 1]
    v = pd.to_numeric(df[value_col], errors="coerce").values

    gx = np.floor(x / cell_m).astype(int)
    gy = np.floor(y / cell_m).astype(int)

    out = df.copy()
    out["_gx"] = gx
    out["_gy"] = gy
    # aggregate mean per cell
    cell_mean = out.groupby(["_gx", "_gy"])[value_col].mean().reset_index().rename(columns={value_col: "cell_mean"})
    out = out.merge(cell_mean, on=["_gx", "_gy"], how="left")

    # classify cell mean into Low/Medium/High by tertiles
    q1 = out["cell_mean"].quantile(0.33)
    q2 = out["cell_mean"].quantile(0.66)

    def cls(a):
        if pd.isna(a):
            return "Neutral"
        if a <= q1:
            return "Low"
        if a <= q2:
            return "Medium"
        return "High"

    out["cluster"] = out["cell_mean"].apply(cls)
    out["method"] = f"Grid binning ({int(cell_m)} m)"
    return out


def hotspot_analysis(
    df: pd.DataFrame,
    value_col: str,
    method: str = "Gi*",
    k: int = 8,
    grid_cell_m: float = 250.0,
    z_thresh: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    method: 'Gi*', 'Moran', 'Grid'
    Returns (df_with_cluster, meta)
    """
    dd = df.copy()
    dd = dd.dropna(subset=["lat", "lon"])
    dd[value_col] = pd.to_numeric(dd[value_col], errors="coerce")
    dd = dd.dropna(subset=[value_col])

    meta = {"method": method, "value_col": value_col, "n": int(len(dd))}

    if len(dd) < 8 and method in ["Gi*", "Moran"]:
        # Too sparse, force grid
        out = grid_binning(dd, value_col=value_col, cell_m=grid_cell_m)
        meta["note"] = "Data sparse for neighbor-based stats. Switched to Grid binning."
        return out, meta

    if method == "Grid":
        out = grid_binning(dd, value_col=value_col, cell_m=grid_cell_m)
        return out, meta

    X, _ = latlon_to_xy_m(dd)
    W = _knn_weights(X, k=k)

    vals = dd[value_col].values.astype(float)

    if method == "Gi*":
        z = getis_ord_gistar(vals, W)
        dd["z"] = z
        dd["cluster"] = np.where(z >= z_thresh, "Hot", np.where(z <= -z_thresh, "Cold", "Neutral"))
        dd["method"] = f"Getis-Ord Gi* (k={k})"
    elif method == "Moran":
        zi = local_morans_i(vals, W)
        # HH/LL based on value and statistic sign
        v_z = (vals - np.mean(vals)) / (np.std(vals) + EPS)
        dd["z"] = zi
        hh = (zi >= z_thresh) & (v_z >= 0)
        ll = (zi >= z_thresh) & (v_z < 0)
        hl_lh = (zi <= -z_thresh)  # outlier-ish
        dd["cluster"] = np.where(hh, "HH", np.where(ll, "LL", np.where(hl_lh, "Outlier", "Neutral")))
        dd["method"] = f"Local Moran's I (k={k})"
    else:
        raise ValueError("method must be one of: Gi*, Moran, Grid")

    # Orientation estimate from high cluster points
    high_mask = dd["cluster"].isin(["Hot", "HH", "High"])
    orient = {"trend_azimuth_deg": None, "trench_azimuth_deg": None, "n_high": int(high_mask.sum())}
    if high_mask.sum() >= 3:
        Xh, _ = latlon_to_xy_m(dd.loc[high_mask])
        # PCA on xy
        C = np.cov(Xh.T)
        eigvals, eigvecs = np.linalg.eig(C)
        v = eigvecs[:, np.argmax(eigvals)]
        az = (np.degrees(np.arctan2(v[0], v[1])) + 360.0) % 180.0  # 0-180
        trench = (az + 90.0) % 180.0
        orient["trend_azimuth_deg"] = float(az)
        orient["trench_azimuth_deg"] = float(trench)

    meta["orientation"] = orient
    return dd, meta
