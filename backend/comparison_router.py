from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from server import db, get_current_user, serialize_for_json  # Import from your main server file

router = APIRouter(prefix="/api/comparison", tags=["comparison"])

class ComparisonRequest(BaseModel):
    dataset_ids: List[str]
    
class ComparisonResult(BaseModel):
    summary: Dict[str, Any]
    schema_diff: Dict[str, Any]
    numeric_comparison: List[Dict[str, Any]]
    categorical_comparison: List[Dict[str, Any]]

@router.post("/analyze", response_model=ComparisonResult)
async def compare_datasets(request: ComparisonRequest, user: dict = Depends(get_current_user)):
    """
    Intelligently compare 2 or more datasets.
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
            raise HTTPException(status_code=404, detail=f"Dataset {ds_id} not found")
            
        datasets_meta.append(meta)
        df = pd.DataFrame(data_doc['data'])
        # Add a source column for tracking
        df['_source_dataset'] = meta.get('title', meta['name'])
        data_frames[meta.get('title', meta['name'])] = df

    # 2. Schema Comparison
    all_columns = set()
    col_map = {}
    for name, df in data_frames.items():
        cols = set(df.columns) - {'_source_dataset'}
        all_columns.update(cols)
        col_map[name] = cols
        
    common_columns = set.intersection(*col_map.values())
    unique_columns = {name: list(cols - common_columns) for name, cols in col_map.items()}

    # 3. Numeric Comparison (on common columns)
    numeric_comparison = []
    
    # Get common numeric columns
    first_df_name = list(data_frames.keys())[0]
    first_df = data_frames[first_df_name]
    common_numeric = [c for c in common_columns if pd.api.types.is_numeric_dtype(first_df[c])]
    
    for col in common_numeric:
        col_stats = {
            "column": col,
            "metrics": []
        }
        
        # Calculate stats for this column across all datasets
        values_list = []
        for name, df in data_frames.items():
            if col in df:
                series = df[col].dropna()
                values_list.append(series.values)
                
                stat = {
                    "dataset": name,
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "count": int(len(series))
                }
                col_stats["metrics"].append(stat)
        
        # Intelligent Insight: ANOVA / T-Test to check if difference is significant
        if len(values_list) >= 2:
            try:
                # One-way ANOVA
                f_stat, p_val = stats.f_oneway(*values_list)
                col_stats["statistical_test"] = {
                    "test": "ANOVA",
                    "p_value": float(p_val),
                    "significant": float(p_val) < 0.05,
                    "insight": "Significant difference detected" if p_val < 0.05 else "Distributions are similar"
                }
            except:
                col_stats["statistical_test"] = None

        numeric_comparison.append(col_stats)

    # 4. Categorical Comparison (Top 5 values)
    categorical_comparison = []
    common_cat = [c for c in common_columns if not pd.api.types.is_numeric_dtype(first_df[c])]
    
    for col in common_cat:
        col_stats = {"column": col, "distributions": []}
        
        for name, df in data_frames.items():
            if col in df:
                # Get top 5 value counts
                vc = df[col].value_counts(normalize=True).head(5)
                dist_data = [{"value": str(k), "percentage": float(v)*100} for k, v in vc.items()]
                
                col_stats["distributions"].append({
                    "dataset": name,
                    "top_values": dist_data
                })
        categorical_comparison.append(col_stats)

    return {
        "summary": {
            "datasets_compared": [d.get('title', d['name']) for d in datasets_meta],
            "total_rows_combined": sum(len(df) for df in data_frames.values()),
            "common_columns_count": len(common_columns)
        },
        "schema_diff": {
            "common_columns": list(common_columns),
            "unique_columns_per_dataset": unique_columns
        },
        "numeric_comparison": numeric_comparison,
        "categorical_comparison": categorical_comparison
    }