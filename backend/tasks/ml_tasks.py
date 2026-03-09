"""Celery tasks for ML training (Phase 3: background execution)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tasks.celery_app import celery_app


@celery_app.task(name="ml.train_baselines")
def task_train_baselines(dataframe_json: str, target_col: str = "prospectivity_class") -> dict:
    """Train ML baselines in background worker."""
    import pandas as pd
    from io import BytesIO
    from core.snree_ml import train_baselines

    df = pd.read_json(BytesIO(dataframe_json.encode()), orient="split")
    result = train_baselines(df, target_col=target_col)

    # Strip non-serializable model objects
    serializable = {
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "classes": result["classes"],
        "models": {},
    }
    for name, info in result["models"].items():
        serializable["models"][name] = {
            "accuracy": info["accuracy"],
            "features": info["features"],
            "report": info.get("report"),
        }
    return serializable


@celery_app.task(name="ml.hotspot_analysis")
def task_hotspot_analysis(
    dataframe_json: str, value_col: str, method: str = "Gi*",
    k: int = 8, grid_cell_m: float = 250.0, z_thresh: float = 1.0,
) -> dict:
    """Run hotspot analysis in background worker."""
    import pandas as pd
    from io import BytesIO
    from core.snree_hotspots import hotspot_analysis

    df = pd.read_json(BytesIO(dataframe_json.encode()), orient="split")
    hot_df, hot_meta = hotspot_analysis(
        df.dropna(subset=["lat", "lon"]),
        value_col=value_col, method=method, k=k,
        grid_cell_m=grid_cell_m, z_thresh=z_thresh,
    )
    return {
        "data": hot_df.to_dict(orient="records"),
        "meta": hot_meta,
    }
