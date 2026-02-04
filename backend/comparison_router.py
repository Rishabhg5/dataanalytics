from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
# Ensure you import db correctly based on your file structure
from server import db, get_current_user 

router = APIRouter(prefix="/api/comparison", tags=["comparison"])



class ComparisonRequest(BaseModel):
    dataset_ids: List[str]
    limit: int = 50

# ... keep your /preview endpoint as is ...



@router.post("/analyze")
async def compare_datasets(request: ComparisonRequest, user: dict = Depends(get_current_user)):
    """
    Advanced Statistical Comparison: Drift, Distributions, and Data Quality.
    """
    if len(request.dataset_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 datasets to compare")

    # 1. Fetch Data
    datasets_meta = []
    data_frames = {}
    
    for ds_id in request.dataset_ids:
        meta = await db.datasets.find_one({"id": ds_id})
        data_doc = await db.dataset_data.find_one({"dataset_id": ds_id})
        
        if not meta or not data_doc:
            continue
            
        datasets_meta.append(meta)
        df = pd.DataFrame(data_doc['data'])
        name = meta.get('title', meta['name'])
        data_frames[name] = df

    if len(data_frames) < 2:
        raise HTTPException(status_code=400, detail="Could not load datasets")

    # 2. Schema Comparison
    col_map = {name: set(df.columns) for name, df in data_frames.items()}
    common_columns = set.intersection(*col_map.values())
    unique_columns = {name: list(cols - common_columns) for name, cols in col_map.items()}

    # 3. Data Quality Comparison (New Section)
    quality_comparison = []
    for name, df in data_frames.items():
        quality_comparison.append({
            "dataset": name,
            "total_rows": len(df),
            "missing_cells": int(df.isnull().sum().sum()),
            "missing_percent": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum())
        })

    # 4. Numeric Deep Dive (KS Test & Histograms)
    numeric_comparison = []
    first_df_name = list(data_frames.keys())[0]
    first_df = data_frames[first_df_name]
    
    # Identify numeric columns present in ALL datasets
    common_numeric = []
    for col in common_columns:
        # Check if numeric in ALL frames
        is_numeric_all = all(pd.api.types.is_numeric_dtype(df[col]) for df in data_frames.values())
        if is_numeric_all:
            common_numeric.append(col)
    
    for col in common_numeric:
        col_data = {}
        values_list = []
        
        # Collect values for this column from all datasets
        for name, df in data_frames.items():
            clean_series = df[col].dropna()
            values_list.append(clean_series)
            col_data[name] = {
                "mean": float(clean_series.mean()),
                "std": float(clean_series.std()),
                "min": float(clean_series.min()),
                "max": float(clean_series.max()),
                "zeros": int((clean_series == 0).sum())
            }

        # --- A. Statistical Drift Test (KS Test) ---
        # We compare the first two datasets as a baseline
        drift_status = "Stable"
        p_value = 1.0
        if len(values_list) >= 2:
            try:
                # KS Test compares the shape of distributions
                # Null hypothesis: samples are from the same distribution
                # p < 0.05 => Reject null => Distributions are DIFFERENT
                stat, p_value = stats.ks_2samp(values_list[0], values_list[1])
                if p_value < 0.05:
                    drift_status = "High Drift Detected"
                elif p_value < 0.15:
                    drift_status = "Moderate Drift"
            except:
                pass

        # --- B. Histogram Generation (For Visual Comparison) ---
        # Create common bins so datasets can be compared side-by-side
        try:
            all_values = np.concatenate(values_list)
            hist_counts = {}
            if len(all_values) > 0:
                # Calculate 10 bins across the global range
                counts, bin_edges = np.histogram(all_values, bins=10)
                
                for i, (name, df) in enumerate(data_frames.items()):
                    # Re-bin each dataset using the global edges
                    ds_counts, _ = np.histogram(df[col].dropna(), bins=bin_edges)
                    # Normalize to percentage to compare datasets of different sizes
                    total = len(df)
                    hist_counts[name] = [int(c)/total*100 if total > 0 else 0 for c in ds_counts]
                
                bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
            else:
                bin_labels = []
        except:
            hist_counts = {}
            bin_labels = []

        numeric_comparison.append({
            "column": col,
            "stats": col_data,
            "drift_analysis": {
                "status": drift_status,
                "test": "Kolmogorov-Smirnov",
                "p_value": float(p_value)
            },
            "histogram": {
                "labels": bin_labels,
                "datasets": hist_counts
            }
        })

    return {
        "summary": {
            "datasets": [d.get('title', d['name']) for d in datasets_meta],
            "common_cols": len(common_columns)
        },
        "quality_comparison": quality_comparison,
        "schema_diff": unique_columns,
        "numeric_comparison": numeric_comparison
    }