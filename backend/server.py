from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query, Depends, Header
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
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
from passlib.context import CryptContext
import jwt
import hashlib
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'e1-analytics-secret-key')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES', 30))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer(auto_error=False)

# Create the main app
app = FastAPI(title="E1 Analytics API", version="2.0.0")
api_router = APIRouter(prefix="/api")

# ===================== MODELS =====================

# Role hierarchy: admin > manager > analyst > viewer
ROLE_HIERARCHY = {
    "admin": 4,
    "manager": 3,
    "analyst": 2,
    "viewer": 1
}

ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users", "view_audit", "mask_data", "export"],
    "manager": ["read", "write", "delete", "view_audit", "export"],
    "analyst": ["read", "write", "export"],
    "viewer": ["read"]
}

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "viewer"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    created_at: datetime
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class Dataset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    title: Optional[str] = None
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: Optional[str] = None
    access_level: str = "public"  # public, private, restricted
    sensitive_columns: List[str] = []
    
class DatasetOverview(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dataset_id: str
    statistics: Optional[Dict[str, Any]] = None
    auto_charts: Optional[Dict[str, Any]] = None
    trends: Optional[Dict[str, Any]] = None
    anomalies: Optional[Dict[str, Any]] = None
    forecast: Optional[Dict[str, Any]] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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

class AuditLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GeneratedReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str
    dataset_name: str
    title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: Optional[str] = None
    generated_by_email: Optional[str] = None
    pdf_filename: str
    charts_included: List[Dict[str, Any]] = []
    statistics_snapshot: Optional[Dict[str, Any]] = None
    insights_snapshot: Optional[Dict[str, Any]] = None

class AIInsightRequest(BaseModel):
    dataset_id: str
    insight_type: str  # chart_description, data_analysis, prescriptive
    context: Optional[Dict[str, Any]] = None

class PrescriptiveRequest(BaseModel):
    dataset_id: str
    business_context: Optional[str] = None
    optimization_goals: Optional[List[str]] = None

# ===================== HELPER FUNCTIONS =====================

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

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token - returns None if no auth provided"""
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user or not user.get("is_active", True):
            return None
        return user
    except:
        return None

async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Require authentication - raises error if not authenticated"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return user

def check_permission(user: dict, required_permission: str) -> bool:
    """Check if user has required permission based on role"""
    if not user:
        return False
    role = user.get("role", "viewer")
    permissions = ROLE_PERMISSIONS.get(role, [])
    return required_permission in permissions

def has_higher_role(user_role: str, target_role: str) -> bool:
    """Check if user role is higher than target role"""
    return ROLE_HIERARCHY.get(user_role, 0) > ROLE_HIERARCHY.get(target_role, 0)

async def log_audit(user_id: str, user_email: str, action: str, resource_type: str, 
                   resource_id: str = None, details: dict = None, ip_address: str = None):
    """Log audit entry"""
    audit = AuditLog(
        user_id=user_id,
        user_email=user_email,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address
    )
    doc = audit.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.audit_logs.insert_one(doc)

def mask_sensitive_data(data: List[dict], sensitive_columns: List[str], user_role: str) -> List[dict]:
    """Mask sensitive data based on user role"""
    if user_role == "admin":
        return data  # Admins see everything
    
    masked_data = []
    for row in data:
        masked_row = row.copy()
        for col in sensitive_columns:
            if col in masked_row:
                value = str(masked_row[col]) if masked_row[col] else ""
                if len(value) > 4:
                    masked_row[col] = value[:2] + "*" * (len(value) - 4) + value[-2:]
                else:
                    masked_row[col] = "****"
        masked_data.append(masked_row)
    return masked_data

# ===================== AI INTEGRATION =====================

async def generate_ai_insight(prompt: str, context: str = "") -> str:
    """Generate AI-powered insight using OpenAI via emergentintegrations"""
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            logger.warning("EMERGENT_LLM_KEY not found, using fallback insights")
            return generate_fallback_insight(prompt)
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"analytics-{uuid.uuid4()}",
            system_message="You are an expert data analyst providing actionable business insights. Be concise, specific, and focus on actionable recommendations. Always provide insights in 2-3 sentences max."
        ).with_model("openai", "gpt-5.2")
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        user_message = UserMessage(text=full_prompt)
        response = await chat.send_message(user_message)
        return response
    except Exception as e:
        logger.error(f"AI insight generation error: {e}")
        return generate_fallback_insight(prompt)

def generate_fallback_insight(prompt: str) -> str:
    """Generate fallback insight when AI is unavailable"""
    if "trend" in prompt.lower():
        return "The data shows notable patterns that warrant further investigation. Consider monitoring key metrics regularly to identify emerging trends."
    elif "anomaly" in prompt.lower():
        return "Anomalies detected in the data may indicate outliers or data quality issues. Review flagged records for potential business impact."
    elif "forecast" in prompt.lower():
        return "Forecasting models suggest continued patterns based on historical data. Consider external factors that may influence future values."
    elif "recommendation" in prompt.lower() or "prescriptive" in prompt.lower():
        return "Based on the analysis, focus on optimizing high-impact metrics while monitoring risk factors. Implement regular review cycles."
    return "Data analysis reveals patterns worth exploring further. Consider segmenting the data for deeper insights."

# ===================== AUTH ROUTES =====================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if email already exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate role
    if user_data.role not in ROLE_HIERARCHY:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {list(ROLE_HIERARCHY.keys())}")
    
    # Create user
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password_hash": hash_password(user_data.password),
        "role": user_data.role,
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    # Log audit
    await log_audit(user_id, user_data.email, "register", "user", user_id)
    
    # Generate token
    access_token = create_access_token({"sub": user_id, "email": user_data.email, "role": user_data.role})
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=user_id,
            email=user_data.email,
            name=user_data.name,
            role=user_data.role,
            created_at=datetime.now(timezone.utc),
            is_active=True
        )
    )

@api_router.post("/auth/login", response_model=TokenResponse)
async def login_user(credentials: UserLogin):
    """Login user and return JWT token"""
    user = await db.users.find_one({"email": credentials.email})
    
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="Account is disabled")
    
    # Log audit
    await log_audit(user["id"], user["email"], "login", "auth")
    
    # Generate token
    access_token = create_access_token({"sub": user["id"], "email": user["email"], "role": user["role"]})
    
    created_at = user.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            role=user["role"],
            created_at=created_at,
            is_active=user.get("is_active", True)
        )
    )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(user: dict = Depends(require_auth)):
    """Get current user information"""
    created_at = user.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        created_at=created_at,
        is_active=user.get("is_active", True)
    )

@api_router.get("/auth/users")
async def list_users(user: dict = Depends(require_auth)):
    """List all users (admin/manager only)"""
    if not check_permission(user, "manage_users") and user["role"] != "manager":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    users = await db.users.find({}, {"_id": 0, "password_hash": 0}).to_list(1000)
    
    # Managers can only see users below their level
    if user["role"] == "manager":
        users = [u for u in users if ROLE_HIERARCHY.get(u["role"], 0) < ROLE_HIERARCHY.get("manager", 0)]
    
    return users

@api_router.put("/auth/users/{user_id}/role")
async def update_user_role(user_id: str, new_role: str, user: dict = Depends(require_auth)):
    """Update user role (admin only)"""
    if not check_permission(user, "manage_users"):
        raise HTTPException(status_code=403, detail="Only admins can change user roles")
    
    if new_role not in ROLE_HIERARCHY:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {list(ROLE_HIERARCHY.keys())}")
    
    target_user = await db.users.find_one({"id": user_id})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Cannot change own role
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"role": new_role, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    await log_audit(user["id"], user["email"], "update_role", "user", user_id, 
                   {"old_role": target_user["role"], "new_role": new_role})
    
    return {"success": True, "message": f"Role updated to {new_role}"}

@api_router.put("/auth/users/{user_id}/status")
async def toggle_user_status(user_id: str, user: dict = Depends(require_auth)):
    """Enable/disable user (admin only)"""
    if not check_permission(user, "manage_users"):
        raise HTTPException(status_code=403, detail="Only admins can manage user status")
    
    target_user = await db.users.find_one({"id": user_id})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")
    
    new_status = not target_user.get("is_active", True)
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"is_active": new_status, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    await log_audit(user["id"], user["email"], "toggle_status", "user", user_id, {"is_active": new_status})
    
    return {"success": True, "is_active": new_status}

# ===================== AUDIT ROUTES =====================

@api_router.get("/audit/logs")
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    user: dict = Depends(require_auth)
):
    """Get audit logs (admin/manager only)"""
    if not check_permission(user, "view_audit"):
        raise HTTPException(status_code=403, detail="Insufficient permissions to view audit logs")
    
    query = {}
    if action:
        query["action"] = action
    if resource_type:
        query["resource_type"] = resource_type
    
    logs = await db.audit_logs.find(query, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return logs

# ===================== DATA MASKING ROUTES =====================

@api_router.put("/datasets/{dataset_id}/sensitive-columns")
async def update_sensitive_columns(
    dataset_id: str,
    sensitive_columns: List[str],
    user: dict = Depends(require_auth)
):
    """Mark columns as sensitive for data masking (admin/manager only)"""
    if not check_permission(user, "mask_data") and user["role"] != "manager":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate columns exist
    invalid_cols = [col for col in sensitive_columns if col not in dataset.get("column_names", [])]
    if invalid_cols:
        raise HTTPException(status_code=400, detail=f"Invalid columns: {invalid_cols}")
    
    await db.datasets.update_one(
        {"id": dataset_id},
        {"$set": {"sensitive_columns": sensitive_columns}}
    )
    
    await log_audit(user["id"], user["email"], "update_sensitive_columns", "dataset", dataset_id,
                   {"columns": sensitive_columns})
    
    return {"success": True, "sensitive_columns": sensitive_columns}

# ===================== AI-POWERED ANALYTICS ROUTES =====================

@api_router.post("/ai/describe-chart")
async def describe_chart_ai(request: AIInsightRequest, user: dict = Depends(get_current_user)):
    """Generate AI-powered chart description"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = await db.datasets.find_one({"id": request.dataset_id}, {"_id": 0})
        df = pd.DataFrame(data_doc['data'])
        
        # Build context for AI
        context = request.context or {}
        chart_type = context.get("chart_type", "visualization")
        columns = context.get("columns", df.columns.tolist()[:3])
        
        # Calculate statistics for context
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_summary = {}
        for col in numeric_cols[:3]:
            stats_summary[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "trend": "increasing" if df[col].iloc[-10:].mean() > df[col].iloc[:10].mean() else "decreasing" if len(df) > 20 else "stable"
            }
        
        prompt = f"""Analyze this {chart_type} visualization for dataset '{dataset.get('title', dataset['name'])}':
        
Columns analyzed: {', '.join(columns)}
Total records: {len(df)}
Statistics: {json.dumps(stats_summary, indent=2)}

Provide a concise, actionable insight about what this visualization reveals. Focus on business implications."""

        insight = await generate_ai_insight(prompt)
        
        if user:
            await log_audit(user["id"], user["email"], "ai_describe_chart", "dataset", request.dataset_id)
        
        return {"insight": insight, "chart_type": chart_type, "dataset_id": request.dataset_id}
        
    except Exception as e:
        logger.error(f"AI chart description error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/ai/analyze-data")
async def analyze_data_ai(request: AIInsightRequest, user: dict = Depends(get_current_user)):
    """Generate comprehensive AI-powered data analysis"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = await db.datasets.find_one({"id": request.dataset_id}, {"_id": 0})
        df = pd.DataFrame(data_doc['data'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Comprehensive analysis
        analysis = {
            "data_quality": {
                "total_records": len(df),
                "missing_percentage": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
                "duplicate_rows": int(df.duplicated().sum())
            },
            "statistical_summary": {},
            "correlations": {},
            "trends": {}
        }
        
        # Statistical summary
        for col in numeric_cols[:5]:
            analysis["statistical_summary"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "skewness": float(df[col].skew()) if len(df) > 2 else 0
            }
        
        # Correlations
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols[:5]].corr()
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:5]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.5:
                        analysis["correlations"][f"{col1}_vs_{col2}"] = float(corr)
        
        # Trend detection
        for col in numeric_cols[:3]:
            values = df[col].dropna().values
            if len(values) > 10:
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                analysis["trends"][col] = {
                    "direction": "increasing" if slope > 0 else "decreasing",
                    "strength": float(abs(r_value))
                }
        
        # Generate AI insight
        prompt = f"""Analyze this dataset comprehensively:

Dataset: {dataset.get('title', dataset['name'])}
Records: {len(df)}
Data Quality: {analysis['data_quality']['missing_percentage']:.1f}% missing data
Key Statistics: {json.dumps(analysis['statistical_summary'], indent=2)}
Strong Correlations: {json.dumps(analysis['correlations'], indent=2)}
Trends: {json.dumps(analysis['trends'], indent=2)}

Provide key findings and business implications in 3-4 bullet points."""

        ai_insight = await generate_ai_insight(prompt)
        analysis["ai_insight"] = ai_insight
        
        if user:
            await log_audit(user["id"], user["email"], "ai_analyze_data", "dataset", request.dataset_id)
        
        return analysis
        
    except Exception as e:
        logger.error(f"AI data analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/ai/prescriptive")
async def get_prescriptive_analytics(request: PrescriptiveRequest, user: dict = Depends(get_current_user)):
    """Generate prescriptive analytics - 'What should we do?' recommendations"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = await db.datasets.find_one({"id": request.dataset_id}, {"_id": 0})
        df = pd.DataFrame(data_doc['data'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        prescriptive = {
            "recommendations": [],
            "optimization_opportunities": [],
            "risk_factors": [],
            "action_items": []
        }
        
        # Data Quality Recommendations
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 10:
            prescriptive["recommendations"].append({
                "category": "Data Quality",
                "priority": "HIGH",
                "recommendation": f"Address data quality issues - {missing_pct:.1f}% missing data",
                "action": "Implement data validation rules and establish data collection protocols",
                "expected_impact": "Improved analysis accuracy and decision reliability"
            })
        
        # Trend-based Recommendations
        for col in numeric_cols[:3]:
            values = df[col].dropna().values
            if len(values) > 20:
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                
                if abs(r_value) > 0.5:
                    if slope > 0:
                        prescriptive["optimization_opportunities"].append({
                            "metric": col,
                            "current_trend": "Positive growth",
                            "recommendation": f"Capitalize on positive trend in {col}",
                            "action": "Increase investment and resources in this area",
                            "projected_benefit": f"Continue {abs(slope):.2f} units growth per period"
                        })
                    else:
                        prescriptive["risk_factors"].append({
                            "metric": col,
                            "current_trend": "Declining",
                            "risk_level": "MEDIUM" if abs(slope) < 1 else "HIGH",
                            "recommendation": f"Investigate declining trend in {col}",
                            "action": "Conduct root cause analysis and implement corrective measures"
                        })
        
        # Anomaly-based Recommendations
        if len(numeric_cols) > 0:
            try:
                X = df[numeric_cols[:3]].dropna()
                if len(X) > 10:
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    predictions = clf.fit_predict(X)
                    anomaly_pct = (predictions == -1).sum() / len(X) * 100
                    
                    if anomaly_pct > 15:
                        prescriptive["risk_factors"].append({
                            "metric": "Data Anomalies",
                            "current_status": f"{anomaly_pct:.1f}% anomalous records",
                            "risk_level": "HIGH" if anomaly_pct > 20 else "MEDIUM",
                            "recommendation": "Review anomalous data points for fraud or errors",
                            "action": "Establish anomaly monitoring and alert systems"
                        })
            except Exception as e:
                logger.warning(f"Anomaly detection error: {e}")
        
        # Correlation-based Optimization
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:5]:
                    corr = corr_matrix.loc[col1, col2]
                    if corr > 0.7:
                        prescriptive["optimization_opportunities"].append({
                            "insight": f"Strong correlation ({corr:.2f}) between {col1} and {col2}",
                            "recommendation": f"Use {col1} as leading indicator for {col2}",
                            "action": "Build predictive models leveraging this relationship"
                        })
        
        # Generate AI-powered strategic recommendations
        business_context = request.business_context or "general business analytics"
        optimization_goals = request.optimization_goals or ["improve efficiency", "reduce costs", "increase growth"]
        
        prompt = f"""Based on this data analysis, provide strategic prescriptive recommendations:

Dataset: {dataset.get('title', dataset['name'])}
Business Context: {business_context}
Optimization Goals: {', '.join(optimization_goals)}

Current Findings:
- Data Quality: {missing_pct:.1f}% missing data
- Identified Risks: {len(prescriptive['risk_factors'])} factors
- Optimization Opportunities: {len(prescriptive['optimization_opportunities'])} opportunities

Provide 3-5 specific, actionable strategic recommendations with expected outcomes."""

        ai_recommendations = await generate_ai_insight(prompt)
        prescriptive["ai_strategic_recommendations"] = ai_recommendations
        
        # Generate specific action items
        prescriptive["action_items"] = [
            {
                "priority": 1,
                "action": "Review and clean data quality issues" if missing_pct > 5 else "Maintain current data quality standards",
                "timeline": "Immediate",
                "owner": "Data Team"
            },
            {
                "priority": 2,
                "action": "Investigate declining trends" if any(r["current_trend"] == "Declining" for r in prescriptive.get("risk_factors", [])) else "Monitor current growth trends",
                "timeline": "This week",
                "owner": "Analytics Team"
            },
            {
                "priority": 3,
                "action": "Build predictive models for key correlations",
                "timeline": "This month",
                "owner": "Data Science Team"
            }
        ]
        
        if user:
            await log_audit(user["id"], user["email"], "prescriptive_analytics", "dataset", request.dataset_id)
        
        return prescriptive
        
    except Exception as e:
        logger.error(f"Prescriptive analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== ML MODEL ROUTES =====================

@api_router.post("/ml/predict")
async def ml_prediction(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
    """Run ML prediction model on dataset"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        
        target_column = request.parameters.get("target_column") if request.parameters else None
        if not target_column:
            raise HTTPException(status_code=400, detail="target_column parameter required")
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found in dataset")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if len(feature_cols) < 1:
            raise HTTPException(status_code=400, detail="Need at least one numeric feature column")
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_column].fillna(df[target_column].mean())
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Model performance
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        
        result = {
            "model_type": "Random Forest Regressor",
            "target_column": target_column,
            "features_used": feature_cols,
            "performance": {
                "r2_score": float(r2),
                "mean_absolute_error": float(mae)
            },
            "feature_importance": sorted_importance,
            "predictions_sample": predictions[:10].tolist(),
            "actual_sample": y[:10].tolist()
        }
        
        # Generate AI insight about the model
        prompt = f"""Analyze this ML model performance and provide recommendations:

Model: Random Forest Regressor
Target: {target_column}
R² Score: {r2:.3f}
Mean Absolute Error: {mae:.2f}
Top Features: {list(sorted_importance.keys())[:3]}

What does this model performance indicate and how can it be improved?"""
        
        result["ai_insight"] = await generate_ai_insight(prompt)
        
        if user:
            await log_audit(user["id"], user["email"], "ml_prediction", "dataset", request.dataset_id)
        
        return result
        
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/ml/cluster")
async def ml_clustering(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
    """Run clustering analysis on dataset"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": request.dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.DataFrame(data_doc['data'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for clustering")
        
        columns = request.columns or numeric_cols[:5]
        n_clusters = request.parameters.get("n_clusters", 3) if request.parameters else 3
        
        # Prepare data
        X = df[columns].dropna()
        if len(X) < n_clusters:
            raise HTTPException(status_code=400, detail=f"Need at least {n_clusters} data points for {n_clusters} clusters")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        X['cluster'] = clusters
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = X[X['cluster'] == i]
            cluster_stats[f"cluster_{i}"] = {
                "size": len(cluster_data),
                "percentage": float(len(cluster_data) / len(X) * 100),
                "center": {col: float(cluster_data[col].mean()) for col in columns}
            }
        
        # Generate AI insight
        prompt = f"""Analyze these customer/data segments from clustering:

Number of clusters: {n_clusters}
Cluster sizes: {[cluster_stats[f'cluster_{i}']['size'] for i in range(n_clusters)]}
Features analyzed: {columns}

Provide actionable insights about what these segments represent and how to target them."""
        
        ai_insight = await generate_ai_insight(prompt)
        
        result = {
            "model_type": "K-Means Clustering",
            "n_clusters": n_clusters,
            "columns_analyzed": columns,
            "cluster_stats": cluster_stats,
            "inertia": float(kmeans.inertia_),
            "ai_insight": ai_insight
        }
        
        if user:
            await log_audit(user["id"], user["email"], "ml_clustering", "dataset", request.dataset_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== EXISTING ROUTES (UPDATED) =====================

@api_router.get("/")
async def root():
    return {"message": "E1 Analytics API", "version": "2.0.0", "features": ["AI Analytics", "ML Models", "Prescriptive Analytics", "RBAC"]}

@api_router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), title: str = None, user: dict = Depends(get_current_user)):
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
            title=title or file.filename.split('.')[0],
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            column_types=column_types,
            owner_id=user["id"] if user else None
        )
        
        doc = dataset.model_dump()
        doc['uploaded_at'] = doc['uploaded_at'].isoformat()
        await db.datasets.insert_one(doc)
        
        if user:
            await log_audit(user["id"], user["email"], "upload_dataset", "dataset", dataset_id)
        
        return dataset
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/datasets/upload-from-api")
async def upload_from_api(api_url: str, headers: Optional[Dict[str, str]] = None, user: dict = Depends(get_current_user)):
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
            column_types=column_types,
            owner_id=user["id"] if user else None
        )
        
        doc = dataset.model_dump()
        doc['uploaded_at'] = doc['uploaded_at'].isoformat()
        await db.datasets.insert_one(doc)
        
        if user:
            await log_audit(user["id"], user["email"], "upload_from_api", "dataset", dataset_id)
        
        return dataset
        
    except Exception as e:
        logger.error(f"API upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/datasets/upload-from-mysql")
async def upload_from_mysql(
    host: str,
    database: str,
    user_db: str,
    password: str,
    query: str,
    port: int = 3306,
    user: dict = Depends(get_current_user)
):
    """Upload data from MySQL database"""
    try:
        import pymysql
        
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user_db,
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
            column_types=column_types,
            owner_id=user["id"] if user else None
        )
        
        doc = dataset.model_dump()
        doc['uploaded_at'] = doc['uploaded_at'].isoformat()
        await db.datasets.insert_one(doc)
        
        if user:
            await log_audit(user["id"], user["email"], "upload_from_mysql", "dataset", dataset_id)
        
        return dataset
        
    except Exception as e:
        logger.error(f"MySQL upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/datasets", response_model=List[Dataset])
async def get_datasets(search: Optional[str] = None, user: dict = Depends(get_current_user)):
    """Get all datasets with optional search"""
    query = {}
    if search:
        query = {"$or": [
            {"title": {"$regex": search, "$options": "i"}},
            {"name": {"$regex": search, "$options": "i"}}
        ]}
    
    datasets = await db.datasets.find(query, {"_id": 0}).to_list(1000)
    for ds in datasets:
        if isinstance(ds['uploaded_at'], str):
            ds['uploaded_at'] = datetime.fromisoformat(ds['uploaded_at'])
        if 'title' not in ds or not ds['title']:
            ds['title'] = ds['name'].split('.')[0]
        # Set defaults for new fields
        if 'sensitive_columns' not in ds:
            ds['sensitive_columns'] = []
        if 'access_level' not in ds:
            ds['access_level'] = 'public'
    return datasets

@api_router.put("/datasets/{dataset_id}/title")
async def update_dataset_title(dataset_id: str, title: str, user: dict = Depends(get_current_user)):
    """Update dataset title"""
    result = await db.datasets.update_one(
        {"id": dataset_id},
        {"$set": {"title": title}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if user:
        await log_audit(user["id"], user["email"], "update_title", "dataset", dataset_id)
    
    return {"success": True}

@api_router.post("/datasets/{dataset_id}/overview")
async def save_dataset_overview(dataset_id: str, overview: DatasetOverview, user: dict = Depends(get_current_user)):
    """Save dataset overview with all analysis"""
    doc = overview.model_dump()
    doc['last_updated'] = doc['last_updated'].isoformat()
    
    await db.dataset_overviews.update_one(
        {"dataset_id": dataset_id},
        {"$set": doc},
        upsert=True
    )
    return {"success": True}

@api_router.get("/datasets/{dataset_id}/overview")
async def get_dataset_overview(dataset_id: str, user: dict = Depends(get_current_user)):
    """Get stored dataset overview"""
    overview = await db.dataset_overviews.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not overview:
        return None
    return overview

@api_router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, limit: int = Query(100, ge=1, le=10000), user: dict = Depends(get_current_user)):
    """Get dataset with data preview"""
    dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not data_doc:
        raise HTTPException(status_code=404, detail="Dataset data not found")
    
    data = data_doc['data'][:limit]
    
    # Apply data masking if user is not admin
    sensitive_cols = dataset.get('sensitive_columns', [])
    user_role = user["role"] if user else "viewer"
    if sensitive_cols and user_role != "admin":
        data = mask_sensitive_data(data, sensitive_cols, user_role)
    
    return {
        "dataset": dataset,
        "data": data,
        "total_rows": len(data_doc['data'])
    }

@api_router.post("/datasets/{dataset_id}/clean")
async def clean_dataset(dataset_id: str, operation: CleaningOperation, user: dict = Depends(get_current_user)):
    """Apply cleaning operations"""
    try:
        # Check write permission
        if user and not check_permission(user, "write"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
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
        
        if user:
            await log_audit(user["id"], user["email"], "clean_dataset", "dataset", dataset_id, 
                          {"operation": operation.operation})
        
        return {"success": True, "rows_remaining": len(df)}
        
    except Exception as e:
        logger.error(f"Cleaning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/descriptive")
async def descriptive_analytics(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
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
        logger.error(f"Descriptive analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/time-series")
async def time_series_analysis(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
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
        logger.error(f"Time series error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/trends")
async def detect_trends(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
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
        logger.error(f"Trend detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/anomalies")
async def detect_anomalies(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
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
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analytics/forecast")
async def forecast_data(request: AnalyticsRequest, user: dict = Depends(get_current_user)):
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
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/dashboards")
async def create_dashboard(dashboard: Dashboard, user: dict = Depends(get_current_user)):
    """Create a new dashboard"""
    doc = dashboard.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.dashboards.insert_one(doc)
    
    if user:
        await log_audit(user["id"], user["email"], "create_dashboard", "dashboard", dashboard.id)
    
    return dashboard

@api_router.get("/dashboards", response_model=List[Dashboard])
async def get_dashboards(user: dict = Depends(get_current_user)):
    """Get all dashboards"""
    dashboards = await db.dashboards.find({}, {"_id": 0}).to_list(1000)
    for dash in dashboards:
        if isinstance(dash['created_at'], str):
            dash['created_at'] = datetime.fromisoformat(dash['created_at'])
    return dashboards

@api_router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, user: dict = Depends(get_current_user)):
    """Delete a dataset"""
    if user and not check_permission(user, "delete"):
        raise HTTPException(status_code=403, detail="Insufficient permissions to delete")
    
    await db.datasets.delete_one({"id": dataset_id})
    await db.dataset_data.delete_one({"dataset_id": dataset_id})
    
    if user:
        await log_audit(user["id"], user["email"], "delete_dataset", "dataset", dataset_id)
    
    return {"success": True}

@api_router.get("/datasets/{dataset_id}/auto-charts")
async def generate_auto_charts(dataset_id: str, user: dict = Depends(get_current_user)):
    """Generate predefined charts automatically (without AI insights for speed)"""
    try:
        data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
        df = pd.DataFrame(data_doc['data'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        charts = []
        
        # Helper function to generate basic insights (no AI)
        def generate_basic_insight(chart_type: str, data_context: dict) -> str:
            if chart_type == "bar distribution":
                return f"Shows distribution of {len(data_context.get('metrics', []))} metrics across {data_context.get('data_points', 0)} data points."
            elif chart_type == "trend line":
                trend = data_context.get('trend', 'stable')
                return f"Data shows {trend} trend. {'Consider capitalizing on growth.' if trend == 'upward' else 'Monitor for potential optimization.' if trend == 'downward' else 'Stable baseline for testing.'}"
            elif chart_type == "pie composition":
                top_pct = data_context.get('top_percentage', 0)
                return f"Top segment represents {top_pct:.1f}% of total. {'High concentration - consider diversification.' if top_pct > 50 else 'Well-distributed segments.'}"
            elif chart_type == "area comparison":
                corr = data_context.get('correlation', 0)
                return f"Correlation: {corr:.2f}. {'Strong relationship detected.' if abs(corr) > 0.7 else 'Moderate correlation.' if abs(corr) > 0.3 else 'Weak correlation.'}"
            elif chart_type == "scatter correlation":
                corr = data_context.get('correlation', 0)
                return f"Scatter analysis shows {abs(corr):.2f} correlation. {'Variables move together.' if corr > 0.5 else 'Variables move inversely.' if corr < -0.5 else 'Limited linear relationship.'}"
            return "Analysis complete."
        
        # 1. Distribution Chart (Bar/Column)
        if len(numeric_cols) >= 1 and len(all_cols) >= 1:
            chart_data = df.head(20)[[all_cols[0]] + numeric_cols[:2]].to_dict('records')
            for record in chart_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            insight = generate_basic_insight("bar distribution", {
                "data_points": len(chart_data),
                "metrics": numeric_cols[:2]
            })
            
            charts.append({
                "type": "bar",
                "title": "Distribution Overview",
                "description": f"Comparison of {', '.join(numeric_cols[:2])} across {all_cols[0]}",
                "data": chart_data,
                "x_axis": all_cols[0],
                "y_axis": numeric_cols[:2],
                "insight": insight
            })
        
        # 2. Trend Analysis (Line Chart)
        if len(numeric_cols) >= 1:
            chart_data = df.head(50)[[all_cols[0]] + numeric_cols[:3]].to_dict('records')
            for record in chart_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            # Calculate trend
            values = df[numeric_cols[0]].dropna().values[:50]
            trend = "stable"
            slope = 0
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                trend = "upward" if slope > 0 else "downward"
            
            insight = generate_basic_insight("trend line", {
                "data_points": len(chart_data),
                "metrics": numeric_cols[:3],
                "trend": trend,
                "slope": slope
            })
            
            charts.append({
                "type": "line",
                "title": "Trend Analysis Over Time",
                "description": f"Tracking patterns in {', '.join(numeric_cols[:3])}",
                "data": chart_data,
                "x_axis": all_cols[0],
                "y_axis": numeric_cols[:3],
                "insight": insight
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
            
            insight = generate_basic_insight("pie composition", {
                "data_points": len(pie_data),
                "metrics": [numeric_cols[0]],
                "top_percentage": top_pct
            })
            
            charts.append({
                "type": "pie",
                "title": "Composition Breakdown",
                "description": f"Percentage distribution of {numeric_cols[0]}",
                "data": pie_data,
                "value_column": numeric_cols[0],
                "label_column": all_cols[0],
                "insight": insight
            })
        
        # 4. Comparison Chart (Area)
        if len(numeric_cols) >= 2:
            chart_data = df.head(30)[[all_cols[0]] + numeric_cols[:2]].to_dict('records')
            for record in chart_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            corr = df[numeric_cols[:2]].corr().iloc[0, 1] if len(numeric_cols) >= 2 else 0
            
            insight = generate_basic_insight("area comparison", {
                "data_points": len(chart_data),
                "metrics": numeric_cols[:2],
                "correlation": corr
            })
            
            charts.append({
                "type": "area",
                "title": "Comparative Analysis",
                "description": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                "data": chart_data,
                "x_axis": all_cols[0],
                "y_axis": numeric_cols[:2],
                "insight": insight
            })
        
        # 5. Correlation Scatter Plot
        if len(numeric_cols) >= 2:
            scatter_data = df.head(100)[numeric_cols[:2]].to_dict('records')
            for record in scatter_data:
                for key in record:
                    record[key] = serialize_for_json(record[key])
            
            corr = df[numeric_cols[:2]].corr().iloc[0, 1]
            
            insight = generate_basic_insight("scatter correlation", {
                "data_points": len(scatter_data),
                "metrics": numeric_cols[:2],
                "correlation": corr
            })
            
            charts.append({
                "type": "scatter",
                "title": "Correlation Analysis",
                "description": f"Scatter plot showing relationship between variables",
                "data": scatter_data,
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "insight": insight
            })
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset['name'],
            "charts": charts,
            "summary": f"Generated {len(charts)} charts with insights."
        }
        
    except Exception as e:
        logger.error(f"Auto-chart generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/{dataset_id}/pdf")
async def generate_pdf_report(dataset_id: str, user: dict = Depends(get_current_user)):
    """Generate comprehensive PDF report - creates NEW report each time with timestamp"""
    try:
        # Fetch dataset
        dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id}, {"_id": 0})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset data not found")
        
        df = pd.DataFrame(data_doc['data'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Generate unique report ID and filename with timestamp
        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"/tmp/report_{dataset_id}_{timestamp_str}.pdf"
        
        # Capture current charts snapshot
        charts_snapshot = []
        try:
            # Generate charts data for the report
            all_cols = df.columns.tolist()
            
            if len(numeric_cols) >= 1 and len(all_cols) >= 1:
                chart_data = df.head(20)[[all_cols[0]] + numeric_cols[:2]].to_dict('records')
                for record in chart_data:
                    for key in record:
                        record[key] = serialize_for_json(record[key])
                charts_snapshot.append({
                    "type": "bar",
                    "title": "Distribution Overview",
                    "data_points": len(chart_data),
                    "columns": [all_cols[0]] + numeric_cols[:2]
                })
            
            if len(numeric_cols) >= 1:
                values = df[numeric_cols[0]].dropna().values[:50]
                trend = "stable"
                if len(values) > 1:
                    x = np.arange(len(values))
                    slope, _, _, _, _ = stats.linregress(x, values)
                    trend = "upward" if slope > 0 else "downward"
                charts_snapshot.append({
                    "type": "line",
                    "title": "Trend Analysis",
                    "trend": trend,
                    "data_points": len(values)
                })
        except Exception as e:
            logger.warning(f"Charts snapshot error: {e}")
        
        # Capture statistics snapshot
        statistics_snapshot = {}
        for col in numeric_cols[:5]:
            statistics_snapshot[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#4F46E5'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=10,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#475569'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#334155'),
            spaceAfter=6,
            leading=16
        )
        
        # Calculate metrics
        total_rows = len(df)
        missing_pct = (df.isnull().sum().sum() / (total_rows * len(df.columns))) * 100
        
        # Trend analysis for summary
        trend_direction = "stable"
        trend_strength = "N/A"
        if len(numeric_cols) > 0:
            primary_col = numeric_cols[0]
            values = df[primary_col].dropna().values[:100]
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                trend_direction = "upward" if slope > 0 else "downward"
                trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
        
        # ============ COVER PAGE ============
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("DATA ANALYTICS REPORT", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        cover_info = [
            ['Report Generated:', datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')],
            ['Dataset Name:', dataset.get('title', dataset['name'])],
            ['Total Records:', f"{dataset['rows']:,}"],
            ['Data Dimensions:', f"{dataset['columns']} columns"],
        ]
        
        cover_table = Table(cover_info, colWidths=[2*inch, 4*inch])
        cover_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(cover_table)
        story.append(Spacer(1, 0.5*inch))
        
        watermark = Paragraph("<i>E1 Analytics - Data Intelligence Platform</i>", 
            ParagraphStyle('Watermark', parent=styles['Normal'], fontSize=10, textColor=colors.grey, alignment=TA_CENTER))
        story.append(watermark)
        story.append(PageBreak())
        
        # ============ 1. EXECUTIVE SUMMARY (ORIGINAL) ============
        story.append(Paragraph("1. EXECUTIVE SUMMARY", heading_style))
        story.append(Paragraph("Actionable Overview", subheading_style))
        
        exec_summary = f"""
        This comprehensive analysis examines {total_rows:,} records across {len(df.columns)} dimensions. 
        The dataset represents {dataset.get('title', dataset['name'])} and has been processed to extract actionable insights.
        
        <b>Key Findings:</b><br/>
        • Data Quality: {100-missing_pct:.1f}% complete with {missing_pct:.1f}% missing values requiring attention<br/>
        • Primary Trend: {trend_strength.capitalize()} {trend_direction} trend detected in key metrics<br/>
        • Business Impact: The data reveals {'growth opportunities' if trend_direction == 'upward' else 'areas requiring optimization'}<br/>
        • Confidence Level: {'High' if missing_pct < 5 else 'Moderate' if missing_pct < 15 else 'Requires data quality improvement'}<br/>
        
        <b>Strategic Implications:</b><br/>
        {'The positive trends indicate momentum that should be sustained through continued investment and monitoring.' if trend_direction == 'upward' else 'The current patterns suggest opportunities for strategic interventions to reverse declining metrics.' if trend_direction == 'downward' else 'Stable patterns provide a foundation for testing new initiatives.'}
        """
        
        story.append(Paragraph(exec_summary, body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # ============ 2. KEY PERFORMANCE INDICATORS (ORIGINAL) ============
        story.append(Paragraph("2. KEY PERFORMANCE INDICATORS (KPIs)", heading_style))
        
        kpi_data = [['KPI', 'Current Value', 'Benchmark', 'Status', 'Insight']]
        
        if len(numeric_cols) > 0:
            for col in numeric_cols[:4]:
                mean_val = df[col].mean()
                median_val = df[col].median()
                benchmark = median_val
                status = "✓ Above" if mean_val > benchmark else "✗ Below" if mean_val < benchmark else "= On Target"
                insight = f"{'Strong performance' if mean_val > benchmark else 'Needs improvement'}"
                
                kpi_data.append([
                    col[:20],
                    f"{mean_val:.2f}",
                    f"{benchmark:.2f}",
                    status,
                    insight
                ])
        
        # Add data quality KPI
        completeness = 100 - missing_pct
        kpi_data.append([
            'Data Completeness',
            f"{completeness:.1f}%",
            '95%',
            "✓ Above" if completeness >= 95 else "✗ Below",
            "Good" if completeness >= 95 else "Needs cleaning"
        ])
        
        kpi_table = Table(kpi_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1*inch, 1.6*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.3*inch))
        
        # ============ 3. DETAILED DESCRIPTIVE STATISTICS (ORIGINAL) ============
        story.append(Paragraph("3. DETAILED DESCRIPTIVE STATISTICS", heading_style))
        
        if numeric_cols:
            stats_data = [['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3']]
            for col in numeric_cols[:5]:
                stats_data.append([
                    col[:15],
                    f"{df[col].mean():.2f}",
                    f"{df[col].median():.2f}",
                    f"{df[col].std():.2f}",
                    f"{df[col].min():.2f}",
                    f"{df[col].max():.2f}",
                    f"{df[col].quantile(0.25):.2f}",
                    f"{df[col].quantile(0.75):.2f}"
                ])
            
            stats_table = Table(stats_data, colWidths=[1.3*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(stats_table)
        
        story.append(PageBreak())
        
        # ============ 4. DATA VISUALIZATIONS & INSIGHTS (ORIGINAL) ============
        story.append(Paragraph("4. DATA VISUALIZATIONS & INSIGHTS", heading_style))
        
        # Visualization 1: Trend Analysis
        if len(numeric_cols) > 0:
            story.append(Paragraph("4.1 Trend Analysis", subheading_style))
            
            col = numeric_cols[0]
            values = df[col].dropna().values[:100]
            if len(values) > 1:
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_insight = f"""
                <b>Key Findings:</b><br/>
                • Trend Direction: {('Upward' if slope > 0 else 'Downward')} ({slope:.4f} per unit)<br/>
                • Strength: R² = {r_value**2:.3f} ({'Strong' if abs(r_value) > 0.7 else 'Moderate' if abs(r_value) > 0.3 else 'Weak'} correlation)<br/>
                • Statistical Significance: p-value = {p_value:.4f}<br/>
                • Business Interpretation: {'Consistent growth pattern suggests positive momentum' if slope > 0 else 'Declining trend requires strategic intervention'}<br/>
                """
                story.append(Paragraph(trend_insight, body_style))
                
                try:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(x, values, 'o-', label='Actual Data', color='#4F46E5', linewidth=2, markersize=4)
                    ax.plot(x, slope * x + intercept, '--', label='Trend Line', color='#10B981', linewidth=2)
                    ax.fill_between(x, values, alpha=0.3, color='#4F46E5')
                    ax.set_xlabel('Time Period', fontsize=10)
                    ax.set_ylabel(col, fontsize=10)
                    ax.set_title(f'Trend Analysis: {col}', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    chart_path = f"/tmp/trend_{dataset_id}.png"
                    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    img = Image(chart_path, width=6*inch, height=3.5*inch)
                    story.append(img)
                except Exception as e:
                    logger.error(f"Trend chart error: {e}")
        
        story.append(Spacer(1, 0.3*inch))
        
        # Distribution Analysis
        if len(numeric_cols) >= 2:
            story.append(Paragraph("4.2 Distribution Analysis", subheading_style))
            
            try:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                
                axes[0].hist(df[col1].dropna(), bins=20, color='#4F46E5', alpha=0.7, edgecolor='white')
                axes[0].set_title(f'{col1[:15]} Distribution', fontsize=10)
                axes[0].set_xlabel(col1[:15], fontsize=8)
                axes[0].set_ylabel('Frequency', fontsize=8)
                
                axes[1].hist(df[col2].dropna(), bins=20, color='#10B981', alpha=0.7, edgecolor='white')
                axes[1].set_title(f'{col2[:15]} Distribution', fontsize=10)
                axes[1].set_xlabel(col2[:15], fontsize=8)
                axes[1].set_ylabel('Frequency', fontsize=8)
                
                plt.tight_layout()
                dist_path = f"/tmp/dist_{dataset_id}.png"
                plt.savefig(dist_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                img = Image(dist_path, width=6*inch, height=2.5*inch)
                story.append(img)
            except Exception as e:
                logger.error(f"Distribution chart error: {e}")
        
        story.append(PageBreak())
        
        # ============ 5. ANOMALY DETECTION (ORIGINAL) ============
        story.append(Paragraph("5. ANOMALY DETECTION", heading_style))
        
        if len(numeric_cols) > 0:
            try:
                X = df[numeric_cols[:5]].dropna()
                if len(X) > 10:
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    predictions = clf.fit_predict(X)
                    anomaly_count = (predictions == -1).sum()
                    anomaly_pct = anomaly_count / len(X) * 100
                    
                    anomaly_text = f"""
                    <b>Isolation Forest Results:</b><br/>
                    • Total Records Analyzed: {len(X):,}<br/>
                    • Anomalies Detected: {anomaly_count} ({anomaly_pct:.1f}%)<br/>
                    • Risk Level: {"HIGH" if anomaly_pct > 15 else "MEDIUM" if anomaly_pct > 5 else "LOW"}<br/>
                    • Recommendation: {"Investigate flagged records immediately" if anomaly_pct > 15 else "Review anomalies during regular audit" if anomaly_pct > 5 else "Continue standard monitoring"}
                    """
                    story.append(Paragraph(anomaly_text, body_style))
                else:
                    story.append(Paragraph("Insufficient data for anomaly detection (minimum 10 records required).", body_style))
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                story.append(Paragraph("Anomaly detection unavailable.", body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # ============ 6. STRATEGIC RECOMMENDATIONS (ORIGINAL) ============
        story.append(Paragraph("6. STRATEGIC RECOMMENDATIONS", heading_style))
        
        recommendations = f"""
        Based on the analysis, here are the key strategic recommendations:<br/><br/>
        
        <b>1. Data Quality Improvement:</b><br/>
        {'Address the ' + str(missing_pct) + '% missing data through systematic data collection and validation processes.' if missing_pct > 5 else 'Maintain current data quality standards through regular audits.'}
        <br/><br/>
        
        <b>2. Trend Optimization:</b><br/>
        {'Capitalize on the positive growth trajectory by increasing investment in key performing areas.' if trend_direction == 'upward' else 'Investigate root causes of declining metrics and implement corrective measures.' if trend_direction == 'downward' else 'Use stable baseline to test new optimization strategies.'}
        <br/><br/>
        
        <b>3. Monitoring & Governance:</b><br/>
        Implement continuous monitoring dashboards to track KPIs in real-time and enable proactive decision-making.
        """
        
        story.append(Paragraph(recommendations, body_style))
        
        story.append(PageBreak())
        
        # ============ 7. AI-POWERED INSIGHTS (NEW - APPENDED) ============
        story.append(Paragraph("7. AI-POWERED INSIGHTS", heading_style))
        story.append(Paragraph("<i>Advanced analytics powered by artificial intelligence</i>", 
            ParagraphStyle('AINote', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#6366F1'))))
        story.append(Spacer(1, 0.2*inch))
        
        # Generate AI executive summary
        ai_summary_prompt = f"""Generate an executive summary for this data analysis:

Dataset: {dataset.get('title', dataset['name'])}
Total Records: {total_rows}
Columns: {dataset['columns']}
Missing Data: {missing_pct:.1f}%
Numeric Columns: {len(numeric_cols)}
Trend: {trend_direction} ({trend_strength})

Provide 3-4 key bullet points with business insights and implications."""

        ai_exec_summary = await generate_ai_insight(ai_summary_prompt)
        
        story.append(Paragraph("7.1 AI-Generated Executive Insight", subheading_style))
        story.append(Paragraph(ai_exec_summary.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # ============ 8. ML-POWERED ANALYTICS (NEW - APPENDED) ============
        story.append(Paragraph("8. ML-POWERED ANALYTICS", heading_style))
        
        # Clustering Analysis
        story.append(Paragraph("8.1 Data Segmentation (K-Means Clustering)", subheading_style))
        
        if len(numeric_cols) >= 2:
            try:
                X_cluster = df[numeric_cols[:4]].dropna()
                if len(X_cluster) > 3:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_cluster)
                    
                    kmeans = KMeans(n_clusters=min(3, len(X_cluster)), random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                    
                    cluster_text = f"""
                    <b>Segmentation Results:</b><br/>
                    • Number of Segments: {len(cluster_sizes)}<br/>
                    • Segment Sizes: {', '.join([f'Segment {i}: {s} ({s/len(clusters)*100:.1f}%)' for i, s in cluster_sizes.items()])}<br/>
                    • Application: Use segments for targeted strategies and personalized approaches
                    """
                    story.append(Paragraph(cluster_text, body_style))
            except Exception as e:
                logger.error(f"Clustering error: {e}")
                story.append(Paragraph("Clustering analysis unavailable.", body_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Predictive Forecasting
        story.append(Paragraph("8.2 Predictive Forecasting", subheading_style))
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            values = df[col].dropna().values
            
            if len(values) >= 10:
                try:
                    model = ExponentialSmoothing(values[-50:], trend='add', seasonal=None)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=10)
                    
                    forecast_text = f"""
                    <b>Forecast for {col}:</b><br/>
                    • Historical Data Points: {len(values)}<br/>
                    • Forecast Horizon: 10 periods<br/>
                    • Average Forecasted Value: {np.mean(forecast):.2f}<br/>
                    • Trend Direction: {"Increasing" if forecast[-1] > forecast[0] else "Decreasing"}<br/>
                    • Confidence: {"High" if len(values) > 50 else "Moderate"}
                    """
                    story.append(Paragraph(forecast_text, body_style))
                    
                    # Create forecast chart
                    try:
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        historical = values[-30:]
                        ax.plot(range(len(historical)), historical, 'o-', label='Historical', color='#4F46E5', linewidth=2, markersize=4)
                        ax.plot(range(len(historical), len(historical) + len(forecast)), forecast, 's--', label='Forecast', color='#F59E0B', linewidth=2, markersize=5)
                        ax.axvline(x=len(historical)-0.5, color='red', linestyle=':', linewidth=1, alpha=0.7)
                        ax.fill_between(range(len(historical), len(historical) + len(forecast)), 
                                       forecast * 0.9, forecast * 1.1, alpha=0.2, color='#F59E0B')
                        ax.set_xlabel('Time Period', fontsize=10)
                        ax.set_ylabel(col, fontsize=10)
                        ax.set_title(f'Predictive Forecast: {col}', fontsize=12, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        forecast_chart_path = f"/tmp/forecast_{dataset_id}.png"
                        plt.savefig(forecast_chart_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        img = Image(forecast_chart_path, width=6*inch, height=3*inch)
                        story.append(img)
                    except Exception as e:
                        logger.error(f"Forecast chart error: {e}")
                        
                except Exception as e:
                    logger.error(f"Forecasting error: {e}")
                    story.append(Paragraph("Forecasting unavailable.", body_style))
            else:
                story.append(Paragraph("Insufficient data for forecasting (minimum 10 data points required).", body_style))
        
        story.append(PageBreak())
        
        # ============ 9. PRESCRIPTIVE ANALYTICS (NEW - APPENDED) ============
        story.append(Paragraph("9. PRESCRIPTIVE ANALYTICS", heading_style))
        story.append(Paragraph("AI-Generated Strategic Recommendations", subheading_style))
        
        prescriptive_prompt = f"""Based on this data analysis, provide 5 specific prescriptive recommendations:

Dataset: {dataset.get('title', dataset['name'])}
Records: {total_rows}
Data Quality: {100-missing_pct:.1f}% complete
Key Metrics: {numeric_cols[:5]}
Trend: {trend_direction}

Provide actionable recommendations in format:
1. [Category]: [Specific Action] - [Expected Outcome]"""

        prescriptive_insights = await generate_ai_insight(prescriptive_prompt)
        
        story.append(Paragraph("<b>AI-Generated Strategic Recommendations:</b>", body_style))
        story.append(Paragraph(prescriptive_insights.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Action Items Table
        story.append(Paragraph("9.1 Priority Action Items", subheading_style))
        
        action_data = [
            ['Priority', 'Action Item', 'Owner', 'Timeline', 'Expected Impact'],
            ['P0', 'Address data quality issues' if missing_pct > 5 else 'Maintain data quality', 'Data Team', 'Immediate', 'HIGH'],
            ['P1', 'Investigate anomalies', 'Analytics', 'This Week', 'MEDIUM'],
            ['P2', 'Optimize key metrics', 'Business', 'This Month', 'HIGH'],
            ['P3', 'Build predictive models', 'Data Science', 'Next Quarter', 'HIGH'],
        ]
        
        action_table = Table(action_data, colWidths=[0.6*inch, 2.2*inch, 1*inch, 1*inch, 1*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(action_table)
        
        # ============ FOOTER ============
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(f"Report Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC", footer_style))
        story.append(Paragraph(f"Report ID: {report_id}", footer_style))
        story.append(Paragraph("E1 Analytics - AI-Powered Data Intelligence Platform", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Save report to database for history tracking
        report_record = GeneratedReport(
            id=report_id,
            dataset_id=dataset_id,
            dataset_name=dataset.get('title', dataset['name']),
            title=f"Analytics Report - {timestamp.strftime('%B %d, %Y %H:%M')}",
            generated_at=timestamp,
            generated_by=user["id"] if user else None,
            generated_by_email=user["email"] if user else None,
            pdf_filename=pdf_filename,
            charts_included=charts_snapshot,
            statistics_snapshot=statistics_snapshot
        )
        
        report_doc = report_record.model_dump()
        report_doc['generated_at'] = report_doc['generated_at'].isoformat()
        await db.generated_reports.insert_one(report_doc)
        
        if user:
            await log_audit(user["id"], user["email"], "generate_pdf_report", "report", report_id, 
                          {"dataset_id": dataset_id, "filename": pdf_filename})
        
        # Return with timestamped filename
        return FileResponse(
            pdf_filename, 
            filename=f"analytics_report_{dataset['name']}_{timestamp_str}.pdf", 
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/history/{dataset_id}")
async def get_report_history(dataset_id: str, user: dict = Depends(get_current_user)):
    """Get history of all generated reports for a dataset"""
    reports = await db.generated_reports.find(
        {"dataset_id": dataset_id}, 
        {"_id": 0}
    ).sort("generated_at", -1).to_list(100)
    
    return reports

@api_router.get("/reports/all")
async def get_all_reports(user: dict = Depends(get_current_user)):
    """Get all generated reports across all datasets"""
    reports = await db.generated_reports.find(
        {}, 
        {"_id": 0}
    ).sort("generated_at", -1).to_list(100)
    
    return reports

@api_router.get("/reports/download/{report_id}")
async def download_report(report_id: str, user: dict = Depends(get_current_user)):
    """Download a specific report by ID"""
    report = await db.generated_reports.find_one({"id": report_id}, {"_id": 0})
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    pdf_path = report.get("pdf_filename")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report file not found on server")
    
    return FileResponse(
        pdf_path,
        filename=f"report_{report_id}.pdf",
        media_type="application/pdf"
    )

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
