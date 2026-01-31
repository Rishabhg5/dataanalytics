from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    """Upload CSV, Excel, JSON, or Text file"""
    try:
        contents = await file.read()
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            json_data = json.loads(contents.decode('utf-8'))
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                df = pd.DataFrame([json_data])
            else:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
        elif file.filename.endswith('.txt'):
            # Try to read as CSV with tab or comma delimiter
            try:
                df = pd.read_csv(io.BytesIO(contents), delimiter='\t')
            except:
                df = pd.read_csv(io.BytesIO(contents), delimiter=',')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Supported: CSV, Excel, JSON, TXT")
        
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

@api_router.post("/datasets/upload-from-api")
async def upload_from_api(api_url: str, headers: Optional[Dict[str, str]] = None):
    """Upload data from API endpoint"""
    try:
        import requests
        response = requests.get(api_url, headers=headers or {}, timeout=30)
        response.raise_for_status()
        
        json_data = response.json()
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            df = pd.DataFrame([json_data])
        else:
            raise HTTPException(status_code=400, detail="API response must be JSON array or object")
        
        # Store data
        dataset_id = str(uuid.uuid4())
        data_records = df.to_dict('records')
        
        for record in data_records:
            for key in record:
                record[key] = serialize_for_json(record[key])
        
        await db.dataset_data.insert_one({
            "dataset_id": dataset_id,
            "data": data_records
        })
        
        column_types = {col: str(df[col].dtype) for col in df.columns}
        dataset = Dataset(
            id=dataset_id,
            name=f"API_Data_{api_url.split('/')[-1]}",
            filename=f"api_data_{dataset_id}.json",
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

@api_router.post("/datasets/upload-from-mysql")
async def upload_from_mysql(
    host: str,
    database: str,
    user: str,
    password: str,
    query: str,
    port: int = 3306
):
    """Upload data from MySQL database"""
    try:
        import pymysql
        
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        df = pd.read_sql(query, connection)
        connection.close()
        
        # Store data
        dataset_id = str(uuid.uuid4())
        data_records = df.to_dict('records')
        
        for record in data_records:
            for key in record:
                record[key] = serialize_for_json(record[key])
        
        await db.dataset_data.insert_one({
            "dataset_id": dataset_id,
            "data": data_records
        })
        
        column_types = {col: str(df[col].dtype) for col in df.columns}
        dataset = Dataset(
            id=dataset_id,
            name=f"MySQL_{database}_{dataset_id[:8]}",
            filename=f"mysql_data_{dataset_id}.csv",
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

@api_router.get("/datasets/{dataset_id}/auto-charts")
async def generate_auto_charts(dataset_id: str):
    """Generate predefined charts automatically"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
        df = pd.DataFrame(data_doc['data'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        charts = []
        
        # 1. Distribution Chart (Bar/Column)
        if len(numeric_cols) >= 1 and len(all_cols) >= 1:
            chart_data = df.head(20)[[all_cols[0]] + numeric_cols[:2]].to_dict('records')
            for record in chart_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            charts.append({
                "type": "bar",
                "title": "Distribution Overview",
                "description": f"Comparison of {', '.join(numeric_cols[:2])} across {all_cols[0]}",
                "data": chart_data,
                "x_axis": all_cols[0],
                "y_axis": numeric_cols[:2],
                "insight": f"This chart shows the distribution and comparison of key metrics. Higher values in {numeric_cols[0]} indicate stronger performance in that category."
            })
        
        # 2. Trend Analysis (Line Chart)
        if len(numeric_cols) >= 1:
            chart_data = df.head(50)[[all_cols[0]] + numeric_cols[:3]].to_dict('records')
            for record in chart_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            # Calculate trend
            values = df[numeric_cols[0]].dropna().values[:50]
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                trend = "upward" if slope > 0 else "downward"
            else:
                trend = "stable"
            
            charts.append({
                "type": "line",
                "title": "Trend Analysis Over Time",
                "description": f"Tracking patterns in {', '.join(numeric_cols[:3])}",
                "data": chart_data,
                "x_axis": all_cols[0],
                "y_axis": numeric_cols[:3],
                "insight": f"The data shows a {trend} trend. This pattern suggests {'growth and positive momentum' if trend == 'upward' else 'decline or stabilization'} in the key metrics over the observation period."
            })
        
        # 3. Composition Chart (Pie)
        if len(numeric_cols) >= 1:
            pie_data = df.head(10)[[all_cols[0], numeric_cols[0]]].to_dict('records')
            for record in pie_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            total = df[numeric_cols[0]].sum()
            top_value = df[numeric_cols[0]].max()
            top_pct = (top_value / total * 100) if total > 0 else 0
            
            charts.append({
                "type": "pie",
                "title": "Composition Breakdown",
                "description": f"Percentage distribution of {numeric_cols[0]}",
                "data": pie_data,
                "value_column": numeric_cols[0],
                "label_column": all_cols[0],
                "insight": f"The largest segment accounts for {top_pct:.1f}% of the total {numeric_cols[0]}. This distribution helps identify dominant categories and areas requiring attention."
            })
        
        # 4. Comparison Chart (Area)
        if len(numeric_cols) >= 2:
            chart_data = df.head(30)[[all_cols[0]] + numeric_cols[:2]].to_dict('records')
            for record in chart_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            corr = df[numeric_cols[:2]].corr().iloc[0, 1] if len(numeric_cols) >= 2 else 0
            relationship = "strong positive" if corr > 0.7 else "moderate positive" if corr > 0.3 else "weak or negative"
            
            charts.append({
                "type": "area",
                "title": "Comparative Analysis",
                "description": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                "data": chart_data,
                "x_axis": all_cols[0],
                "y_axis": numeric_cols[:2],
                "insight": f"The analysis reveals a {relationship} relationship between these variables (correlation: {corr:.2f}). This suggests {'they move together' if corr > 0.5 else 'independent patterns' if corr > -0.5 else 'inverse relationship'}."
            })
        
        # 5. Correlation Scatter Plot
        if len(numeric_cols) >= 2:
            scatter_data = df.head(100)[numeric_cols[:2]].to_dict('records')
            for record in scatter_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            corr = df[numeric_cols[:2]].corr().iloc[0, 1]
            
            charts.append({
                "type": "scatter",
                "title": "Correlation Analysis",
                "description": f"Scatter plot showing relationship between variables",
                "data": scatter_data,
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "insight": f"The scatter plot reveals a correlation coefficient of {corr:.3f}. {'Strong correlation indicates predictable patterns' if abs(corr) > 0.7 else 'Moderate correlation suggests some relationship' if abs(corr) > 0.3 else 'Low correlation indicates independent variables'}, useful for forecasting and decision-making."
            })
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset['name'],
            "charts": charts,
            "summary": f"Generated {len(charts)} predefined charts with insights for comprehensive data visualization."
        }
        
    except Exception as e:
        logger.error(f"Auto-chart generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/{dataset_id}/pdf")
async def generate_pdf_report(dataset_id: str):
    """Generate comprehensive PDF report with analytics"""
    try:
        # Fetch dataset
        dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id}, {"_id": 0})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset data not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        # Create PDF
        pdf_filename = f"/tmp/report_{dataset_id}.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4F46E5'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=10,
            spaceBefore=15
        )
        
        # Title
        story.append(Paragraph("E1 Analytics - Comprehensive Data Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Dataset Information
        story.append(Paragraph("Dataset Overview", heading_style))
        dataset_info = [
            ['Property', 'Value'],
            ['Dataset Name', dataset['name']],
            ['Total Rows', str(dataset['rows'])],
            ['Total Columns', str(dataset['columns'])],
            ['Upload Date', str(dataset['uploaded_at'])[:19]],
        ]
        t = Table(dataset_info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Descriptive Statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            story.append(Paragraph("Descriptive Statistics", heading_style))
            
            stats_data = [['Column', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']]
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                stats_data.append([
                    col,
                    f"{df[col].mean():.2f}",
                    f"{df[col].median():.2f}",
                    f"{df[col].std():.2f}",
                    f"{df[col].min():.2f}",
                    f"{df[col].max():.2f}"
                ])
            
            stats_table = Table(stats_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Trend Analysis
        if len(numeric_cols) > 0:
            story.append(Paragraph("Trend Analysis", heading_style))
            
            col = numeric_cols[0]
            values = df[col].dropna().values[:100]  # Limit to 100 points
            if len(values) > 1:
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_data = [
                    ['Metric', 'Value'],
                    ['Analyzed Column', col],
                    ['Trend Direction', 'Increasing' if slope > 0 else 'Decreasing'],
                    ['Slope', f"{slope:.4f}"],
                    ['R-squared', f"{r_value**2:.4f}"],
                    ['P-value', f"{p_value:.4f}"]
                ]
                
                trend_table = Table(trend_data, colWidths=[2*inch, 4*inch])
                trend_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                story.append(trend_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Create trend chart
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(x, values, 'o-', label='Actual', color='#4F46E5', linewidth=2)
                    ax.plot(x, slope * x + intercept, '--', label='Trend', color='#10B981', linewidth=2)
                    ax.set_xlabel('Index')
                    ax.set_ylabel(col)
                    ax.set_title(f'Trend Analysis - {col}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    chart_path = f"/tmp/trend_{dataset_id}.png"
                    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    img = Image(chart_path, width=5*inch, height=2.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
                except Exception as e:
                    logger.error(f"Chart error: {e}")
        
        # Anomaly Detection
        if len(numeric_cols) > 0:
            story.append(Paragraph("Anomaly Detection", heading_style))
            
            try:
                X = df[numeric_cols[:3]].dropna()  # Use first 3 numeric columns
                if len(X) > 10:
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    predictions = clf.fit_predict(X)
                    anomalies = (predictions == -1).sum()
                    
                    anomaly_data = [
                        ['Metric', 'Value'],
                        ['Total Records Analyzed', str(len(X))],
                        ['Anomalies Detected', str(anomalies)],
                        ['Anomaly Percentage', f"{(anomalies/len(X)*100):.2f}%"],
                        ['Columns Analyzed', ', '.join(numeric_cols[:3])]
                    ]
                    
                    anomaly_table = Table(anomaly_data, colWidths=[2.5*inch, 3.5*inch])
                    anomaly_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F59E0B')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(1, 0.95, 0.8)),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    story.append(anomaly_table)
                    story.append(Spacer(1, 0.3*inch))
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
        
        # Predictive Analysis - Forecasting
        if len(numeric_cols) > 0:
            story.append(Paragraph("Predictive Analysis - Forecasting", heading_style))
            
            col = numeric_cols[0]
            values = df[col].dropna().values
            
            if len(values) >= 10:
                try:
                    model = ExponentialSmoothing(values[-50:], trend='add', seasonal=None)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=10)
                    
                    forecast_data = [
                        ['Metric', 'Value'],
                        ['Forecasted Column', col],
                        ['Historical Data Points', str(len(values))],
                        ['Forecast Periods', '10'],
                        ['Average Forecast Value', f"{np.mean(forecast):.2f}"],
                        ['Forecast Trend', 'Increasing' if forecast[-1] > forecast[0] else 'Decreasing']
                    ]
                    
                    forecast_table = Table(forecast_data, colWidths=[2.5*inch, 3.5*inch])
                    forecast_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    story.append(forecast_table)
                    story.append(Spacer(1, 0.3*inch))
                    
                    # Create forecast chart
                    try:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        historical = values[-30:]
                        ax.plot(range(len(historical)), historical, 'o-', label='Historical', color='#4F46E5', linewidth=2)
                        ax.plot(range(len(historical), len(historical) + len(forecast)), forecast, 's--', label='Forecast', color='#F59E0B', linewidth=2)
                        ax.set_xlabel('Time Period')
                        ax.set_ylabel(col)
                        ax.set_title(f'Forecast - {col}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        forecast_chart_path = f"/tmp/forecast_{dataset_id}.png"
                        plt.savefig(forecast_chart_path, dpi=100, bbox_inches='tight')
                        plt.close()
                        
                        img = Image(forecast_chart_path, width=5*inch, height=2.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.3*inch))
                    except Exception as e:
                        logger.error(f"Forecast chart error: {e}")
                except Exception as e:
                    logger.error(f"Forecasting error: {e}")
        
        # Prescriptive Analysis
        story.append(Paragraph("Prescriptive Analysis & Recommendations", heading_style))
        
        recommendations = []
        
        # Check data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 5:
            recommendations.append(f"• Data Quality: {missing_pct:.1f}% missing values detected. Consider data cleaning to improve analysis accuracy.")
        else:
            recommendations.append("• Data Quality: Good data quality with minimal missing values.")
        
        # Check for trends
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            values = df[col].dropna().values[:100]
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                if abs(slope) > 0.01:
                    direction = "upward" if slope > 0 else "downward"
                    recommendations.append(f"• Trend Alert: Strong {direction} trend detected in {col}. Monitor closely for business impact.")
        
        # Data volume recommendation
        if len(df) < 100:
            recommendations.append("• Data Volume: Limited data points. Consider collecting more data for robust predictive analysis.")
        else:
            recommendations.append(f"• Data Volume: Good sample size with {len(df)} records for reliable analysis.")
        
        # General recommendations
        recommendations.append("• Regular Monitoring: Schedule periodic data refreshes to maintain up-to-date insights.")
        recommendations.append("• Advanced Analytics: Consider implementing machine learning models for deeper insights.")
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", footer_style))
        story.append(Paragraph("E1 Analytics - Data Intelligence Platform", footer_style))
        
        # Build PDF
        doc.build(story)
        
        return FileResponse(pdf_filename, filename=f"analytics_report_{dataset['name']}.pdf", media_type="application/pdf")
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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