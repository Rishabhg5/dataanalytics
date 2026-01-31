from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
class Dataset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DatasetCreate(BaseModel):
    name: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]

class CleaningOperation(BaseModel):
    dataset_id: str
    operation: str
    column: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class AnalyticsRequest(BaseModel):
    dataset_id: str
    analysis_type: str
    columns: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None

class Dashboard(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    dataset_id: str
    charts: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper functions
def serialize_for_json(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

# Routes
@api_router.get("/")
async def root():
    return {"message": "E1 Analytics API"}

@api_router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV or Excel file"""
    try:
        contents = await file.read()
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store data in MongoDB
        dataset_id = str(uuid.uuid4())
        data_records = df.to_dict('records')
        
        # Serialize data
        for record in data_records:
            for key in record:
                record[key] = serialize_for_json(record[key])
        
        await db.dataset_data.insert_one({
            "dataset_id": dataset_id,
            "data": data_records
        })
        
        # Create dataset metadata
        column_types = {col: str(df[col].dtype) for col in df.columns}
        dataset = Dataset(
            id=dataset_id,
            name=file.filename,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            column_types=column_types
        )
        
        doc = dataset.model_dump()
        doc['uploaded_at'] = doc['uploaded_at'].isoformat()
        await db.datasets.insert_one(doc)
        
        return dataset
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/datasets", response_model=List[Dataset])
async def get_datasets():
    """Get all datasets"""
    datasets = await db.datasets.find({}, {"_id": 0}).to_list(1000)
    for ds in datasets:
        if isinstance(ds['uploaded_at'], str):
            ds['uploaded_at'] = datetime.fromisoformat(ds['uploaded_at'])
    return datasets

@api_router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, limit: int = Query(100, ge=1, le=10000)):
    """Get dataset with data preview"""
    dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not data_doc:
        raise HTTPException(status_code=404, detail="Dataset data not found")
    
    return {
        "dataset": dataset,
        "data": data_doc['data'][:limit],
        "total_rows": len(data_doc['data'])
    }

@api_router.post("/datasets/{dataset_id}/clean")
async def clean_dataset(dataset_id: str, operation: CleaningOperation):
    """Apply cleaning operations"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        if operation.operation == "remove_duplicates":
            df = df.drop_duplicates()
        
        elif operation.operation == "fill_missing":
            if operation.column and operation.parameters:
                method = operation.parameters.get('method', 'mean')
                if method == 'mean':
                    df[operation.column] = df[operation.column].fillna(df[operation.column].mean())
                elif method == 'median':
                    df[operation.column] = df[operation.column].fillna(df[operation.column].median())
                elif method == 'mode':
                    df[operation.column] = df[operation.column].fillna(df[operation.column].mode()[0])
                elif method == 'forward_fill':
                    df[operation.column] = df[operation.column].fillna(method='ffill')
        
        elif operation.operation == "drop_missing":
            if operation.column:
                df = df.dropna(subset=[operation.column])
            else:
                df = df.dropna()
        
        elif operation.operation == "convert_type":
            if operation.column and operation.parameters:
                target_type = operation.parameters.get('type')
                if target_type == 'numeric':
                    df[operation.column] = pd.to_numeric(df[operation.column], errors='coerce')
                elif target_type == 'datetime':
                    df[operation.column] = pd.to_datetime(df[operation.column], errors='coerce')
        
        # Update dataset
        data_records = df.to_dict('records')
        for record in data_records:
            for key in record:
                record[key] = serialize_for_json(record[key])
        
        await db.dataset_data.update_one(
            {"dataset_id": dataset_id},
            {"$set": {"data": data_records}}
        )
        
        await db.datasets.update_one(
            {"id": dataset_id},
            {"$set": {"rows": len(df)}}
        )
        
        return {"success": True, "rows_remaining": len(df)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/descriptive")
async def descriptive_analytics(request: AnalyticsRequest):
    """Get descriptive statistics"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if request.columns:
            numeric_cols = [col for col in request.columns if col in numeric_cols]
        
        stats_dict = {}
        for col in numeric_cols:
            stats_dict[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'count': int(df[col].count()),
                'missing': int(df[col].isna().sum())
            }
        
        return {"statistics": stats_dict}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/time-series")
async def time_series_analysis(request: AnalyticsRequest):
    """Perform time-series analysis"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        date_col = request.parameters.get('date_column') if request.parameters else None
        value_col = request.parameters.get('value_column') if request.parameters else None
        
        if not date_col or not value_col:
            raise HTTPException(status_code=400, detail="date_column and value_column required")
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Calculate rolling statistics
        window = request.parameters.get('window', 7)
        df['rolling_mean'] = df[value_col].rolling(window=window).mean()
        df['rolling_std'] = df[value_col].rolling(window=window).std()
        
        result = df[[date_col, value_col, 'rolling_mean', 'rolling_std']].to_dict('records')
        for record in result:
            for key in record:
                record[key] = serialize_for_json(record[key])
        
        return {"data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/trends")
async def detect_trends(request: AnalyticsRequest):
    """Detect trends in data"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        column = request.parameters.get('column') if request.parameters else None
        if not column:
            raise HTTPException(status_code=400, detail="column parameter required")
        
        # Simple linear trend
        values = df[column].dropna().values
        x = np.arange(len(values))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        trend_line = slope * x + intercept
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "trend": "increasing" if slope > 0 else "decreasing",
            "trend_line": trend_line.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/anomalies")
async def detect_anomalies(request: AnalyticsRequest):
    """Detect anomalies using Isolation Forest"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        columns = request.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            raise HTTPException(status_code=400, detail="No numeric columns found")
        
        X = df[columns].dropna()
        
        # Isolation Forest
        contamination = request.parameters.get('contamination', 0.1) if request.parameters else 0.1
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        
        anomalies = (predictions == -1).sum()
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        return {
            "total_anomalies": int(anomalies),
            "anomaly_percentage": float(anomalies / len(X) * 100),
            "anomaly_indices": anomaly_indices[:100]  # Limit to first 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/forecast")
async def forecast_data(request: AnalyticsRequest):
    """Forecast future values"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        column = request.parameters.get('column') if request.parameters else None
        periods = request.parameters.get('periods', 10) if request.parameters else 10
        
        if not column:
            raise HTTPException(status_code=400, detail="column parameter required")
        
        values = df[column].dropna().values
        
        if len(values) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points for forecasting")
        
        try:
            # Exponential Smoothing
            model = ExponentialSmoothing(values, trend='add', seasonal=None)
            fitted = model.fit()
            forecast = fitted.forecast(steps=periods)
            
            return {
                "historical": values.tolist(),
                "forecast": forecast.tolist(),
                "forecast_periods": periods
            }
        except:
            # Fallback to simple moving average
            last_values = values[-min(5, len(values)):]
            forecast = [float(np.mean(last_values))] * periods
            return {
                "historical": values.tolist(),
                "forecast": forecast,
                "forecast_periods": periods
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/dashboards")
async def create_dashboard(dashboard: Dashboard):
    """Create a new dashboard"""
    doc = dashboard.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.dashboards.insert_one(doc)
    return dashboard

@api_router.get("/dashboards", response_model=List[Dashboard])
async def get_dashboards():
    """Get all dashboards"""
    dashboards = await db.dashboards.find({}, {"_id": 0}).to_list(1000)
    for dash in dashboards:
        if isinstance(dash['created_at'], str):
            dash['created_at'] = datetime.fromisoformat(dash['created_at'])
    return dashboards

@api_router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    await db.datasets.delete_one({"id": dataset_id})
    await db.dataset_data.delete_one({"dataset_id": dataset_id})
    return {"success": True}

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()