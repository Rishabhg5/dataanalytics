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
# OR if you want to use PyJWT directly:
import jwt
import bcrypt
import hashlib


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
  
    password_bytes = password.encode('utf-8')
    
    # If password is too long for bcrypt, pre-hash it with SHA-256
    if len(password_bytes) > 72:
        password_bytes = hashlib.sha256(password_bytes).digest()
    
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash (Python 3.14 compatible)
    Uses SHA-256 pre-hash for passwords > 72 bytes
    """
    try:
        password_bytes = plain_password.encode('utf-8')
        
        # If password is too long for bcrypt, pre-hash it with SHA-256
        if len(password_bytes) > 72:
            password_bytes = hashlib.sha256(password_bytes).digest()
        
        return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))
    except Exception:
        return False

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


# ============================================
# API ROUTES (SIMPLIFIED)
# ============================================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if email already exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate role
    if user_data.role not in ROLE_HIERARCHY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {list(ROLE_HIERARCHY.keys())}"
        )
    
    # Hash password (now handles any length)
    try:
        password_hash = hash_password(user_data.password)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process password")
    
    # Create user
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password_hash": password_hash,
        "role": user_data.role,
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    # Log audit
    await log_audit(user_id, user_data.email, "register", "user", user_id)
    
    # Generate token
    access_token = create_access_token({
        "sub": user_id,
        "email": user_data.email,
        "role": user_data.role
    })
    
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
    
    # Verify password (now handles any length)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="Account is disabled")
    
    # Log audit
    await log_audit(user["id"], user["email"], "login", "auth")
    
    # Generate token
    access_token = create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"]
    })
    
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
    if not check_permission(user, "write"):
        raise HTTPException(status_code=403, detail="Insufficient permissions to upload data")
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
    if not check_permission(user, "write"):
        raise HTTPException(status_code=403, detail="Insufficient permissions to upload data")
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
    if not check_permission(user, "write"):
        raise HTTPException(status_code=403, detail="Insufficient permissions to upload data")
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
async def generate_pdf_report(dataset_id: str):
    """Generate comprehensive PDF report with AI-style automated analysis, executive summary, KPIs, visualizations, intelligent recommendations, relationship analysis, and performance predictions"""
    try:
        # Fetch dataset
        print(f"Generating PDF report for dataset {dataset_id}")
        dataset = await db.datasets.find_one({"id": dataset_id})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        data_doc = await db.dataset_data.find_one({"dataset_id": dataset_id}, {"_id": 0})
        if not data_doc:
            raise HTTPException(status_code=404, detail="Dataset data not found")
        
        df = pd.DataFrame(data_doc['data'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure /tmp directory exists
        os.makedirs("/tmp", exist_ok=True)
        
        # ============ AUTOMATED INTELLIGENT ANALYSIS ============
        def generate_intelligent_insights(df, numeric_cols, dataset_name):
            """Generate AI-style insights using statistical analysis"""
            insights = {
                'critical_insights': [],
                'risk_assessment': [],
                'opportunities': [],
                'strategic_recommendations': [],
                'key_takeaways': []
            }
            
            # Calculate basic metrics
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_pct = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100
            completeness = 100 - missing_pct
            
            # DATA QUALITY INSIGHTS
            if completeness >= 95:
                insights['critical_insights'].append(
                    f"Excellent data quality with {completeness:.1f}% completeness, providing high confidence for decision-making"
                )
                insights['opportunities'].append(
                    f"Clean dataset enables advanced analytics and machine learning implementations"
                )
            elif completeness >= 85:
                insights['critical_insights'].append(
                    f"Good data quality at {completeness:.1f}% completeness, suitable for most analytical purposes"
                )
                insights['risk_assessment'].append(
                    f"{missing_pct:.1f}% missing data could introduce bias in specific analyses"
                )
                insights['strategic_recommendations'].append(
                    "Implement data validation at collection points to reduce missing values below 5%"
                )
            else:
                insights['risk_assessment'].append(
                    f"Data quality concern: {missing_pct:.1f}% missing values may significantly impact analysis reliability"
                )
                insights['strategic_recommendations'].append(
                    "PRIORITY: Launch immediate data quality improvement initiative to achieve >90% completeness"
                )
            
            # SAMPLE SIZE ANALYSIS
            if total_rows > 1000:
                insights['critical_insights'].append(
                    f"Large sample size ({total_rows:,} records) provides robust statistical power for predictions"
                )
            elif total_rows > 100:
                insights['critical_insights'].append(
                    f"Adequate sample size ({total_rows:,} records) supports reliable trend analysis"
                )
            else:
                insights['risk_assessment'].append(
                    f"Limited sample size ({total_rows:,} records) may constrain predictive accuracy"
                )
                insights['strategic_recommendations'].append(
                    "Expand data collection to achieve minimum 500 records for robust forecasting"
                )
            
            if numeric_cols:
                # TREND ANALYSIS
                primary_col = numeric_cols[0]
                values = df[primary_col].dropna().values
                if len(values) > 1:
                    x = np.arange(len(values[:100]))
                    slope, _, r_value, p_value, _ = stats.linregress(x, values[:100])
                    r_squared = r_value ** 2
                    
                    trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
                    trend_direction = "upward" if slope > 0 else "downward"
                    
                    if trend_direction == "upward" and r_squared > 0.5:
                        insights['opportunities'].append(
                            f"Strong upward trend detected in {primary_col} (R²={r_squared:.3f}), indicating consistent growth momentum"
                        )
                        insights['strategic_recommendations'].append(
                            f"Capitalize on positive trajectory in {primary_col} through increased investment and resource allocation"
                        )
                    elif trend_direction == "downward" and r_squared > 0.5:
                        insights['risk_assessment'].append(
                            f"Declining trend in {primary_col} (R²={r_squared:.3f}) requires immediate corrective action"
                        )
                        insights['strategic_recommendations'].append(
                            f"Conduct root cause analysis for {primary_col} decline and implement turnaround strategy within 30 days"
                        )
                    
                    if p_value < 0.05:
                        insights['critical_insights'].append(
                            f"Statistically significant {trend_direction} trend (p={p_value:.4f}) in primary metric, not due to random chance"
                        )
                
                # VARIABILITY ANALYSIS
                cv_values = []
                for col in numeric_cols[:5]:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if mean_val != 0:
                        cv = (std_val / mean_val) * 100
                        cv_values.append(cv)
                
                if cv_values:
                    avg_cv = np.mean(cv_values)
                    if avg_cv < 20:
                        insights['opportunities'].append(
                            f"Low variability (CV={avg_cv:.1f}%) indicates stable, predictable metrics ideal for forecasting"
                        )
                    elif avg_cv > 50:
                        insights['risk_assessment'].append(
                            f"High variability (CV={avg_cv:.1f}%) suggests volatile conditions requiring careful monitoring"
                        )
                        insights['strategic_recommendations'].append(
                            "Implement early warning system to detect significant deviations from expected ranges"
                        )
                
                # CORRELATION ANALYSIS
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols[:5]].corr()
                    strong_correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                strong_correlations.append({
                                    'col1': corr_matrix.columns[i],
                                    'col2': corr_matrix.columns[j],
                                    'correlation': corr_val
                                })
                    
                    if strong_correlations:
                        top_corr = strong_correlations[0]
                        insights['critical_insights'].append(
                            f"Strong correlation ({top_corr['correlation']:.3f}) between {top_corr['col1']} and {top_corr['col2']} enables predictive modeling"
                        )
                        insights['opportunities'].append(
                            f"Leverage {top_corr['col1']}-{top_corr['col2']} relationship to build leading indicator framework"
                        )
                
                # OUTLIER ANALYSIS
                try:
                    X = df[numeric_cols[:3]].dropna()
                    if len(X) > 10:
                        clf = IsolationForest(contamination=0.1, random_state=42)
                        predictions = clf.fit_predict(X)
                        anomalies = (predictions == -1).sum()
                        anomaly_pct = (anomalies / len(X)) * 100
                        
                        if anomaly_pct < 5:
                            insights['opportunities'].append(
                                f"Low anomaly rate ({anomaly_pct:.1f}%) confirms data consistency and operational stability"
                            )
                        elif anomaly_pct > 15:
                            insights['risk_assessment'].append(
                                f"High anomaly rate ({anomaly_pct:.1f}%) indicates potential data quality issues or operational inconsistencies"
                            )
                            insights['strategic_recommendations'].append(
                                f"Investigate {int(anomalies)} anomalous records to identify systemic issues or process breakdowns"
                            )
                except:
                    pass
                
                # FORECAST ANALYSIS
                try:
                    if len(values) >= 10:
                        model = ExponentialSmoothing(values[-50:], trend='add', seasonal=None)
                        fitted = model.fit()
                        forecast = fitted.forecast(steps=10)
                        
                        forecast_trend = "increasing" if forecast[-1] > forecast[0] else "decreasing"
                        growth_pct = ((forecast[-1] - values[-1]) / values[-1] * 100) if values[-1] != 0 else 0
                        
                        if forecast_trend == "increasing" and abs(growth_pct) > 10:
                            insights['opportunities'].append(
                                f"Forecast predicts {abs(growth_pct):.1f}% growth over next period - prepare for scaling requirements"
                            )
                        elif forecast_trend == "decreasing" and abs(growth_pct) > 10:
                            insights['risk_assessment'].append(
                                f"Forecast indicates {abs(growth_pct):.1f}% decline - implement contingency plans immediately"
                            )
                except:
                    pass
            
            # GENERATE KEY TAKEAWAYS
            if insights['critical_insights']:
                insights['key_takeaways'].append(insights['critical_insights'][0])
            
            if insights['opportunities']:
                insights['key_takeaways'].append(insights['opportunities'][0])
            else:
                insights['key_takeaways'].append(
                    f"Dataset provides {completeness:.1f}% complete view with {total_rows:,} records for analysis"
                )
            
            if insights['risk_assessment']:
                insights['key_takeaways'].append(insights['risk_assessment'][0])
            else:
                insights['key_takeaways'].append(
                    "No significant risks identified - operational metrics within expected ranges"
                )
            
            # Ensure we have at least 3 takeaways
            while len(insights['key_takeaways']) < 3:
                insights['key_takeaways'].append(
                    f"Continue monitoring {numeric_cols[0] if numeric_cols else 'key metrics'} for emerging patterns"
                )
            
            # Ensure we have recommendations
            if not insights['strategic_recommendations']:
                insights['strategic_recommendations'] = [
                    "Establish automated monitoring dashboards for real-time insights",
                    "Schedule quarterly deep-dive analysis to track progress",
                    "Implement data governance framework to maintain quality standards"
                ]
            
            return insights
        
        # Generate intelligent insights
        ai_insights = generate_intelligent_insights(df, numeric_cols, dataset['name'])
        
        # Create PDF
        pdf_filename = f"/tmp/report_{dataset_id}.pdf"
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
        
        insight_style = ParagraphStyle(
            'InsightStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1F2937'),
            spaceAfter=8,
            leading=16,
            leftIndent=10,
            rightIndent=10
        )
        
        # ============ COVER PAGE ============
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("DATA ANALYTICS REPORT", title_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<i>Intelligent Insights & Analysis</i>", 
            ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, textColor=colors.HexColor('#6366F1'), alignment=TA_CENTER, fontName='Helvetica-Oblique')))
        story.append(Spacer(1, 0.3*inch))
        
        cover_info = [
            ['Report Generated:', datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')],
            ['Dataset Name:', dataset['name']],
            ['Total Records:', f"{dataset['rows']:,}"],
            ['Data Dimensions:', f"{dataset['columns']} columns"],
            ['Analysis Type:', 'Comprehensive with Automated Insights'],
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
        
        watermark = Paragraph("<i>Memat Data Analytics - Data Intelligence Platform | Automated Analysis</i>", 
            ParagraphStyle('Watermark', parent=styles['Normal'], fontSize=10, textColor=colors.grey, alignment=TA_CENTER))
        story.append(watermark)
        story.append(PageBreak())
        
        # ============ EXECUTIVE SUMMARY ============
        story.append(Paragraph("1. EXECUTIVE SUMMARY", heading_style))
        story.append(Paragraph("Actionable Overview", subheading_style))
        
        # Calculate key metrics for summary
        total_rows = len(df)
        missing_pct = (df.isnull().sum().sum() / (total_rows * len(df.columns))) * 100
        
        # Trend analysis for summary
        if len(numeric_cols) > 0:
            primary_col = numeric_cols[0]
            values = df[primary_col].dropna().values[:100]
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _, r_value, _, _ = stats.linregress(x, values)
                trend_direction = "upward" if slope > 0 else "downward"
                trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
            else:
                trend_direction = "stable"
                trend_strength = "N/A"
        else:
            trend_direction = "N/A"
            trend_strength = "N/A"
        
        exec_summary = f"""
        This comprehensive analysis examines {total_rows:,} records across {len(df.columns)} dimensions. 
        The dataset represents {dataset['name']} and has been processed to extract actionable insights.
        
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
        
        # ============ KEY PERFORMANCE INDICATORS ============
        story.append(Paragraph("2. KEY PERFORMANCE INDICATORS (KPIs)", heading_style))
        
        kpi_data = [['KPI', 'Current Value', 'Benchmark', 'Status', 'Insight']]
        
        if len(numeric_cols) > 0:
            for col in numeric_cols[:4]:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                max_val = df[col].max()
                
                # Benchmark (using median as benchmark)
                benchmark = median_val
                variance = ((mean_val - benchmark) / benchmark * 100) if benchmark != 0 else 0
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
        
        # ============ DETAILED STATISTICS ============
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
        
        # ============ DATA VISUALIZATIONS ============
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
                    
                    if os.path.exists(chart_path):
                        img = Image(chart_path, width=6*inch, height=3.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                    else:
                        logger.error(f"Trend chart not created: {chart_path}")
                except Exception as e:
                    logger.error(f"Trend chart error: {e}")
        
        # Visualization 2: Distribution Analysis
        if len(numeric_cols) >= 2:
            story.append(Paragraph("4.2 Distribution & Comparison", subheading_style))
            
            corr = df[numeric_cols[:2]].corr().iloc[0, 1] if len(numeric_cols) >= 2 else 0
            dist_insight = f"""
            <b>Comparative Analysis:</b><br/>
            • Correlation Coefficient: {corr:.3f}<br/>
            • Relationship Type: {('Strong Positive' if corr > 0.7 else 'Moderate Positive' if corr > 0.3 else 'Weak/Negative')}<br/>
            • Implication: {'Variables move together predictably' if abs(corr) > 0.5 else 'Variables show independent patterns'}<br/>
            • Strategic Use: {'Can use one metric to predict the other' if abs(corr) > 0.6 else 'Monitor metrics independently'}<br/>
            """
            story.append(Paragraph(dist_insight, body_style))
            
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
                
                # Histogram
                ax1.hist(df[numeric_cols[0]].dropna(), bins=20, color='#4F46E5', alpha=0.7, edgecolor='black')
                ax1.set_xlabel(numeric_cols[0], fontsize=9)
                ax1.set_ylabel('Frequency', fontsize=9)
                ax1.set_title('Distribution', fontsize=10, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Scatter plot
                ax2.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5, color='#10B981', s=30)
                ax2.set_xlabel(numeric_cols[0], fontsize=9)
                ax2.set_ylabel(numeric_cols[1], fontsize=9)
                ax2.set_title('Correlation', fontsize=10, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                chart_path = f"/tmp/dist_{dataset_id}.png"
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                if os.path.exists(chart_path):
                    img = Image(chart_path, width=6*inch, height=2.5*inch)
                    story.append(img)
                else:
                    logger.error(f"Distribution chart not created: {chart_path}")
            except Exception as e:
                logger.error(f"Distribution chart error: {e}")
        
        story.append(PageBreak())
        
        # ============ ANOMALY DETECTION ============
        story.append(Paragraph("5. ANOMALY DETECTION & OUTLIERS", heading_style))
        
        anomaly_added = False
        if len(numeric_cols) > 0:
            try:
                X = df[numeric_cols[:3]].dropna()
                if len(X) > 10:
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    predictions = clf.fit_predict(X)
                    anomalies = (predictions == -1).sum()
                    anomaly_pct = (anomalies/len(X)*100)
                    
                    anomaly_insight = f"""
                    <b>Anomaly Detection Results:</b><br/>
                    • Total Data Points Analyzed: {len(X):,}<br/>
                    • Anomalies Detected: {anomalies} ({anomaly_pct:.2f}%)<br/>
                    • Severity Assessment: {('Low - Normal variation' if anomaly_pct < 5 else 'Medium - Some outliers present' if anomaly_pct < 15 else 'High - Significant outliers detected')}<br/>
                    • Data Quality Impact: {('Minimal - Data is reliable' if anomaly_pct < 5 else 'Moderate - Review flagged records' if anomaly_pct < 15 else 'Significant - Investigate data collection')}<br/>
                    • Recommended Action: {('Continue monitoring' if anomaly_pct < 5 else 'Review anomalous records for patterns' if anomaly_pct < 15 else 'Conduct detailed investigation of outliers')}<br/>
                    """
                    story.append(Paragraph(anomaly_insight, body_style))
                    
                    anomaly_data = [
                        ['Metric', 'Value'],
                        ['Total Records Analyzed', f"{len(X):,}"],
                        ['Anomalies Detected', str(anomalies)],
                        ['Anomaly Rate', f"{anomaly_pct:.2f}%"],
                        ['Columns Analyzed', ', '.join(numeric_cols[:3])],
                        ['Detection Method', 'Isolation Forest Algorithm']
                    ]
                    
                    anomaly_table = Table(anomaly_data, colWidths=[2.5*inch, 3.5*inch])
                    anomaly_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F59E0B')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(1, 0.95, 0.8)),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('PADDING', (0, 0), (-1, -1), 8)
                    ]))
                    story.append(anomaly_table)
                    anomaly_added = True
                else:
                    story.append(Paragraph(
                        f"<b>Note:</b> Insufficient data points ({len(X)}) for reliable anomaly detection. Minimum 10 records required.",
                        body_style
                    ))
                    anomaly_added = True
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                story.append(Paragraph(
                    f"<b>Note:</b> Anomaly detection could not be performed due to data structure limitations.",
                    body_style
                ))
                anomaly_added = True
        
        if not anomaly_added:
            story.append(Paragraph(
                "<b>Note:</b> No numeric columns available for anomaly detection analysis.",
                body_style
            ))
        
        story.append(Spacer(1, 0.3*inch))
        
        # ============ PREDICTIVE FORECASTING ============
        story.append(Paragraph("6. PREDICTIVE ANALYTICS & FORECASTING", heading_style))
        
        forecast_trend = "stable"
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            values = df[col].dropna().values
            
            if len(values) >= 10:
                try:
                    model = ExponentialSmoothing(values[-50:], trend='add', seasonal=None)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=10)
                    
                    avg_forecast = np.mean(forecast)
                    forecast_trend = "increasing" if forecast[-1] > forecast[0] else "decreasing"
                    
                    forecast_insight = f"""
                    <b>Forecast Analysis:</b><br/>
                    • Forecasted Column: {col}<br/>
                    • Historical Data Points: {len(values):,}<br/>
                    • Forecast Horizon: 10 periods<br/>
                    • Predicted Average: {avg_forecast:.2f}<br/>
                    • Forecast Trend: {forecast_trend.capitalize()}<br/>
                    • Model Type: Exponential Smoothing<br/>
                    • Confidence: {'High' if len(values) > 50 else 'Moderate' if len(values) > 20 else 'Low'} (based on {len(values)} data points)<br/>
                    <br/>
                    <b>Business Implications:</b><br/>
                    {f'Expected growth of {((forecast[-1] - values[-1]) / values[-1] * 100):.1f}% over forecast period. ' if forecast_trend == 'increasing' else f'Expected decline of {((values[-1] - forecast[-1]) / values[-1] * 100):.1f}% over forecast period. '}
                    {'Plan for increased capacity and resources.' if forecast_trend == 'increasing' else 'Prepare contingency plans and corrective measures.'}
                    """
                    story.append(Paragraph(forecast_insight, body_style))
                    
                    try:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        historical = values[-30:]
                        ax.plot(range(len(historical)), historical, 'o-', label='Historical', color='#4F46E5', linewidth=2)
                        ax.plot(range(len(historical), len(historical) + len(forecast)), forecast, 's--', label='Forecast', color='#F59E0B', linewidth=2, markersize=6)
                        ax.axvline(x=len(historical)-0.5, color='red', linestyle=':', linewidth=1, label='Forecast Start')
                        ax.fill_between(range(len(historical), len(historical) + len(forecast)), forecast * 0.9, forecast * 1.1, alpha=0.2, color='#F59E0B', label='Confidence Band')
                        ax.set_xlabel('Time Period', fontsize=10)
                        ax.set_ylabel(col, fontsize=10)
                        ax.set_title(f'Predictive Forecast: {col}', fontsize=12, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        forecast_chart_path = f"/tmp/forecast_{dataset_id}.png"
                        plt.savefig(forecast_chart_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        if os.path.exists(forecast_chart_path):
                            img = Image(forecast_chart_path, width=6*inch, height=3.5*inch)
                            story.append(img)
                        else:
                            logger.error(f"Forecast chart not created: {forecast_chart_path}")
                    except Exception as e:
                        logger.error(f"Forecast chart error: {e}")
                except Exception as e:
                    logger.error(f"Forecasting error: {e}")
        
        story.append(PageBreak())
        
        # ============ AUTOMATED INTELLIGENT ANALYSIS ============
        story.append(Paragraph("7. INTELLIGENT AUTOMATED ANALYSIS", heading_style))
        
        # Add analysis badge
        analysis_badge = Paragraph(
            "🔍 <b>Automated Analysis System</b> | Advanced statistical pattern recognition",
            ParagraphStyle('AnalysisBadge', parent=styles['Normal'], fontSize=10, 
                          textColor=colors.HexColor('#6366F1'), alignment=TA_CENTER,
                          spaceAfter=12, fontName='Helvetica-Bold')
        )
        story.append(analysis_badge)
        
        # Critical Insights
        if ai_insights['critical_insights']:
            story.append(Paragraph("<b>🎯 Critical Insights</b>", subheading_style))
            for i, insight in enumerate(ai_insights['critical_insights'], 1):
                story.append(Paragraph(f"{i}. {insight}", insight_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Risk Assessment
        if ai_insights['risk_assessment']:
            story.append(Paragraph("<b>⚠️ Risk Assessment</b>", subheading_style))
            for i, risk in enumerate(ai_insights['risk_assessment'], 1):
                story.append(Paragraph(f"{i}. {risk}", insight_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Opportunities
        if ai_insights['opportunities']:
            story.append(Paragraph("<b>💡 Opportunities Identified</b>", subheading_style))
            for i, opp in enumerate(ai_insights['opportunities'], 1):
                story.append(Paragraph(f"{i}. {opp}", insight_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Strategic Recommendations
        if ai_insights['strategic_recommendations']:
            story.append(Paragraph("<b>📊 Strategic Recommendations</b>", subheading_style))
            for i, rec in enumerate(ai_insights['strategic_recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec}", insight_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Add disclaimer
        disclaimer = Paragraph(
            "<i>Note: Automated analysis is generated based on statistical algorithms and pattern recognition. Professional judgment should be applied when implementing recommendations.</i>",
            ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, 
                          textColor=colors.grey, alignment=TA_CENTER)
        )
        story.append(disclaimer)
        
        story.append(PageBreak())
        
        # ============ COMPREHENSIVE DATA INSIGHTS ============
        story.append(Paragraph("8. COMPREHENSIVE DATA INSIGHTS & ANALYSIS", heading_style))
        
        story.append(Paragraph("8.1 Dataset Overview & Composition", subheading_style))
        
        # Detailed dataset composition analysis
        total_cells = len(df) * len(df.columns)
        filled_cells = total_cells - df.isnull().sum().sum()
        empty_cells = df.isnull().sum().sum()
        
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        
        composition_text = f"""
        <b>Excel File Structure Analysis:</b><br/>
        <br/>
        Your Excel dataset contains a total of <b>{len(df):,} rows</b> and <b>{len(df.columns)} columns</b>, 
        creating a matrix of <b>{total_cells:,} individual data cells</b>. This represents a 
        {('large' if len(df) > 1000 else 'medium-sized' if len(df) > 100 else 'small')} dataset suitable for 
        {('advanced analytics and machine learning' if len(df) > 1000 else 'statistical analysis and trend detection' if len(df) > 100 else 'basic analysis and pattern identification')}.
        <br/><br/>
        <b>Data Completeness Analysis:</b><br/>
        Of the {total_cells:,} total cells in your Excel file:<br/>
        • <b>{filled_cells:,} cells ({(filled_cells/total_cells*100):.1f}%)</b> contain data<br/>
        • <b>{empty_cells:,} cells ({(empty_cells/total_cells*100):.1f}%)</b> are empty or contain missing values<br/>
        <br/>
        This {('excellent' if (filled_cells/total_cells*100) > 95 else 'good' if (filled_cells/total_cells*100) > 85 else 'moderate')} 
        level of data completeness means your analysis is {('highly reliable' if (filled_cells/total_cells*100) > 95 else 'generally reliable' if (filled_cells/total_cells*100) > 85 else 'adequate but could benefit from data cleaning')}.
        <br/><br/>
        <b>Column Type Distribution:</b><br/>
        Your dataset is composed of different types of data columns:<br/>
        • <b>Numeric Columns: {len(numeric_cols)}</b> ({(len(numeric_cols)/len(df.columns)*100):.1f}% of all columns)<br/>
        These columns contain quantitative data like measurements, counts, amounts, or ratings that can be mathematically analyzed.<br/>
        <br/>
        • <b>Text Columns: {len(text_cols)}</b> ({(len(text_cols)/len(df.columns)*100):.1f}% of all columns)<br/>
        These columns contain categorical data, names, descriptions, or labels that classify or identify your records.<br/>
        <br/>
        """
        
        if date_cols:
            composition_text += f"""
            • <b>Date/Time Columns: {len(date_cols)}</b> ({(len(date_cols)/len(df.columns)*100):.1f}% of all columns)<br/>
            These columns track temporal information, enabling time-series analysis and trend tracking over periods.<br/>
            <br/>
            """
        
        if bool_cols:
            composition_text += f"""
            • <b>Boolean Columns: {len(bool_cols)}</b> ({(len(bool_cols)/len(df.columns)*100):.1f}% of all columns)<br/>
            These columns contain yes/no or true/false values, useful for binary classifications and flags.<br/>
            <br/>
            """
        
        composition_text += f"""
        <b>Data Density & Information Richness:</b><br/>
        The numeric data in your Excel file provides {len(numeric_cols) * len(df):,} individual numeric data points 
        for quantitative analysis. With {len(text_cols)} text columns, you have rich categorical information that adds 
        context and enables segmentation analysis. This combination of quantitative and qualitative data creates a 
        well-rounded dataset for comprehensive business intelligence.
        """
        
        story.append(Paragraph(composition_text, body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # ============ DETAILED COLUMN-BY-COLUMN ANALYSIS ============
        story.append(Paragraph("8.2 Column-by-Column Deep Dive Analysis", subheading_style))
        
        column_analysis_intro = f"""
        <b>Understanding Each Column in Your Data:</b><br/>
        <br/>
        Let's examine each column in detail to understand what information it contains, how it behaves, 
        and what insights it can provide for your business decisions. This analysis covers data distribution, 
        patterns, and potential issues in each column.
        """
        story.append(Paragraph(column_analysis_intro, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Analyze each numeric column in detail
        if numeric_cols:
            for idx, col in enumerate(numeric_cols[:8], 1):  # Analyze up to 8 numeric columns
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    median_val = col_data.median()
                    std_val = col_data.std()
                    min_val = col_data.min()
                    max_val = col_data.max()
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                    range_val = max_val - min_val
                    missing_count = df[col].isnull().sum()
                    missing_pct = (missing_count / len(df)) * 100
                    unique_count = col_data.nunique()
                    
                    # Determine data characteristics
                    if cv < 20:
                        variability = "LOW VARIABILITY - Data points are tightly clustered around the average"
                    elif cv < 50:
                        variability = "MODERATE VARIABILITY - Data shows reasonable spread with some fluctuation"
                    else:
                        variability = "HIGH VARIABILITY - Data is widely dispersed with significant fluctuations"
                    
                    # Check for outliers
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    outlier_pct = (outliers / len(col_data)) * 100
                    
                    # Distribution skew
                    if mean_val > median_val * 1.1:
                        skew_info = "RIGHT-SKEWED - Higher values occur more frequently than lower values"
                    elif mean_val < median_val * 0.9:
                        skew_info = "LEFT-SKEWED - Lower values occur more frequently than higher values"
                    else:
                        skew_info = "NORMALLY DISTRIBUTED - Values are symmetrically distributed around the average"
                    
                    column_detail = f"""
                    <b>Column #{idx}: {col}</b><br/>
                    <br/>
                    <b>Basic Statistics:</b><br/>
                    This column contains numeric values ranging from <b>{min_val:.2f}</b> (minimum) to <b>{max_val:.2f}</b> (maximum), 
                    spanning a total range of <b>{range_val:.2f}</b> units. The average value across all records is <b>{mean_val:.2f}</b>, 
                    while the median (middle value) is <b>{median_val:.2f}</b>. The standard deviation of <b>{std_val:.2f}</b> indicates 
                    how much values typically vary from the average.
                    <br/><br/>
                    <b>Data Quality & Completeness:</b><br/>
                    Out of {len(df):,} total records, this column has <b>{len(col_data):,} filled values</b> and 
                    <b>{missing_count} missing values</b> ({missing_pct:.1f}% missing). 
                    {('This excellent data completeness ensures reliable analysis.' if missing_pct < 5 else 'Some data gaps exist but analysis remains viable.' if missing_pct < 15 else 'Significant missing data may impact analysis reliability - consider data collection improvements.')}
                    The column contains <b>{unique_count:,} unique distinct values</b>, 
                    {('indicating high granularity and detailed tracking' if unique_count > len(col_data) * 0.5 else 'showing some repeated patterns in the data' if unique_count > len(col_data) * 0.1 else 'with many repeated values suggesting categorical nature')}.
                    <br/><br/>
                    <b>Distribution Pattern:</b><br/>
                    {skew_info}. The coefficient of variation is <b>{cv:.1f}%</b>, indicating {variability}.
                    <br/><br/>
                    <b>Quartile Analysis:</b><br/>
                    • 25% of values fall below <b>{q1:.2f}</b> (First Quartile)<br/>
                    • 50% of values fall below <b>{median_val:.2f}</b> (Median/Second Quartile)<br/>
                    • 75% of values fall below <b>{q3:.2f}</b> (Third Quartile)<br/>
                    • The interquartile range (middle 50% of data) spans <b>{iqr:.2f}</b> units<br/>
                    <br/>
                    <b>Outlier Detection:</b><br/>
                    Using statistical methods, we identified <b>{outliers} potential outliers</b> ({outlier_pct:.1f}% of data). 
                    {('This low outlier rate suggests consistent, reliable data.' if outlier_pct < 5 else 'Moderate outlier presence - review these values to ensure data quality.' if outlier_pct < 15 else 'High outlier rate - investigate these anomalies as they may indicate data errors or exceptional cases requiring special attention.')}
                    <br/><br/>
                    <b>Business Interpretation:</b><br/>
                    {f'The average {col} of {mean_val:.2f} with relatively ' + ('low' if cv < 20 else 'moderate' if cv < 50 else 'high') + f' variation suggests ' + ('predictable and stable performance' if cv < 20 else 'some fluctuation but manageable trends' if cv < 50 else 'volatile conditions requiring careful monitoring') + '. '}
                    {f'Values typically range from {q1:.2f} to {q3:.2f}, which can be used as benchmark targets for performance evaluation.'}
                    """
                    
                    story.append(Paragraph(column_detail, body_style))
                    story.append(Spacer(1, 0.25*inch))
        
        # Analyze text columns
        if text_cols:
            story.append(Paragraph("8.3 Categorical Data Analysis", subheading_style))
            
            for idx, col in enumerate(text_cols[:5], 1):  # Analyze up to 5 text columns
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    unique_values = col_data.nunique()
                    missing_count = df[col].isnull().sum()
                    missing_pct = (missing_count / len(df)) * 100
                    mode_value = col_data.mode()[0] if len(col_data.mode()) > 0 else "N/A"
                    mode_count = (col_data == mode_value).sum()
                    mode_pct = (mode_count / len(col_data)) * 100
                    
                    # Get top categories
                    value_counts = col_data.value_counts()
                    top_5_categories = value_counts.head(5)
                    
                    text_detail = f"""
                    <b>Text Column: {col}</b><br/>
                    <br/>
                    <b>Category Distribution:</b><br/>
                    This categorical column contains <b>{unique_values:,} unique categories or values</b> across 
                    {len(col_data):,} records. The most common value is "<b>{mode_value}</b>", which appears 
                    <b>{mode_count:,} times</b> ({mode_pct:.1f}% of all records). This 
                    {('high concentration' if mode_pct > 50 else 'moderate concentration' if mode_pct > 20 else 'low concentration')} 
                    suggests {('a dominant category that drives most of your data' if mode_pct > 50 else 'some clear patterns in category distribution' if mode_pct > 20 else 'diverse categories with balanced representation')}.
                    <br/><br/>
                    <b>Top 5 Most Frequent Categories:</b><br/>
                    """
                    
                    for i, (cat, count) in enumerate(top_5_categories.items(), 1):
                        cat_pct = (count / len(col_data)) * 100
                        text_detail += f"• #{i}: <b>{cat}</b> - {count:,} occurrences ({cat_pct:.1f}%)<br/>"
                    
                    text_detail += f"""
                    <br/>
                    <b>Data Quality:</b><br/>
                    Missing values: <b>{missing_count}</b> ({missing_pct:.1f}%). 
                    {('Excellent data completeness.' if missing_pct < 5 else 'Good data quality with minor gaps.' if missing_pct < 15 else 'Consider data validation to reduce missing categories.')}
                    <br/><br/>
                    <b>Diversity Analysis:</b><br/>
                    With {unique_values:,} distinct categories in {len(col_data):,} records, this column shows 
                    {('very high diversity - almost every record is unique' if unique_values / len(col_data) > 0.8 else 'high diversity with many distinct values' if unique_values / len(col_data) > 0.3 else 'moderate diversity with some repeated patterns' if unique_values / len(col_data) > 0.1 else 'low diversity with frequently repeated categories')}.
                    This level of categorization is ideal for {('detailed segmentation and granular analysis' if unique_values / len(col_data) > 0.5 else 'grouping and pattern identification' if unique_values / len(col_data) > 0.1 else 'broad categorization and high-level grouping')}.
                    """
                    
                    story.append(Paragraph(text_detail, body_style))
                    story.append(Spacer(1, 0.25*inch))
        
        story.append(PageBreak())
        
        # ============ DATA RELATIONSHIPS & INTERACTIONS ============
        story.append(Paragraph("8.4 How Your Data Columns Interact With Each Other", subheading_style))
        
        interaction_intro = f"""
        <b>Understanding Column Relationships:</b><br/>
        <br/>
        In your Excel data, columns don't exist in isolation - they interact, influence each other, and create patterns 
        that reveal important business insights. This section analyzes how different columns in your dataset relate to 
        one another, which relationships are strong enough to use for predictions, and what these connections mean for 
        your business decisions.
        <br/><br/>
        <b>Why Column Relationships Matter:</b><br/>
        When two columns move together (correlation), it means changes in one can help predict changes in the other. 
        This is invaluable for:<br/>
        • <b>Forecasting:</b> Use leading indicators to predict future outcomes<br/>
        • <b>Root Cause Analysis:</b> Understand what drives changes in key metrics<br/>
        • <b>Efficiency:</b> Focus monitoring on the most impactful factors<br/>
        • <b>Optimization:</b> Know which levers to pull for maximum effect<br/>
        """
        story.append(Paragraph(interaction_intro, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # ============ PREDICTIVE INSIGHTS ============
        story.append(Paragraph("8.5 Predictive Analysis: What Your Data Tells About Future Performance", subheading_style))
        
        if len(numeric_cols) >= 2:
            prediction_intro = f"""
            <b>Machine Learning-Based Predictions:</b><br/>
            <br/>
            Using advanced statistical modeling, we've analyzed how the columns in your Excel file can be used to 
            predict future outcomes. This analysis uses the patterns hidden in your historical data to forecast 
            what's likely to happen next, giving you a powerful tool for proactive decision-making.
            <br/><br/>
            <b>What We're Predicting:</b><br/>
            We've built a predictive model that uses <b>{min(len(numeric_cols)-1, 5)} of your data columns</b> 
            to forecast the values of <b>{numeric_cols[0]}</b>. This means by tracking these key factors, you can 
            anticipate future performance before it happens, allowing you to take preventive action or capitalize 
            on opportunities early.
            <br/><br/>
            <b>How Accurate Are These Predictions?</b><br/>
            The predictive model has been tested on actual historical data from your Excel file to validate its accuracy. 
            """
            
            # Calculate prediction statistics if possible
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_absolute_error
                
                target_col = numeric_cols[0]
                feature_cols = numeric_cols[1:min(6, len(numeric_cols))]
                data_clean = df[[target_col] + feature_cols].dropna()
                
                if len(data_clean) >= 20:
                    X = data_clean[feature_cols]
                    y = data_clean[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    accuracy_pct = r2 * 100
                    
                    prediction_intro += f"""
                    The model achieves <b>{accuracy_pct:.1f}% accuracy</b>, meaning it can explain {accuracy_pct:.1f}% of the 
                    variation in {target_col}. On average, predictions are within <b>±{mae:.2f}</b> of the actual values.
                    <br/><br/>
                    <b>Reliability Assessment:</b><br/>
                    """
                    
                    if r2 > 0.8:
                        prediction_intro += f"""
                        <b>EXCELLENT RELIABILITY:</b> This high accuracy level means you can confidently use these predictions 
                        for strategic planning, budget forecasting, and resource allocation. The model captures the underlying 
                        patterns in your data extremely well.
                        """
                    elif r2 > 0.6:
                        prediction_intro += f"""
                        <b>GOOD RELIABILITY:</b> The predictions are reliable enough for planning and trend identification. 
                        While not perfect, the model captures the major patterns in your data and can guide decision-making 
                        effectively when combined with business judgment.
                        """
                    elif r2 > 0.4:
                        prediction_intro += f"""
                        <b>MODERATE RELIABILITY:</b> The predictions show the general direction and trends but should be used 
                        for guidance rather than precise forecasts. Combine these insights with other information sources and 
                        expert knowledge for best results.
                        """
                    else:
                        prediction_intro += f"""
                        <b>LIMITED RELIABILITY:</b> The predictions reveal some patterns but have significant uncertainty. 
                        Use these insights to understand relationships between variables rather than for precise forecasting. 
                        Consider collecting more data or additional relevant factors.
                        """
                    
                    prediction_intro += f"""
                    <br/><br/>
                    <b>Practical Applications:</b><br/>
                    Based on this predictive model, here's how you can use your Excel data:<br/>
                    <br/>
                    1. <b>Early Warning System:</b> Monitor the {len(feature_cols)} predictor columns daily/weekly. 
                    When they change, you'll know {target_col} is likely to change soon, giving you advance warning.<br/>
                    <br/>
                    2. <b>Scenario Planning:</b> Input different values for your predictor columns to see how {target_col} 
                    would likely respond. This helps test "what-if" scenarios before implementing changes.<br/>
                    <br/>
                    3. <b>Target Setting:</b> Use the relationships to set realistic targets. If you want {target_col} 
                    to reach a specific value, the model shows what levels the other columns need to achieve.<br/>
                    <br/>
                    4. <b>Root Cause Analysis:</b> When {target_col} underperforms, check the predictor columns to identify 
                    which factors are driving the issue and where to focus improvement efforts.<br/>
                    """
                    
            except:
                prediction_intro += """
                Predictive modeling requires clean data without missing values. Some predictive capabilities may be 
                limited in this dataset, but relationship analysis still provides valuable insights.
                """
        else:
            prediction_intro = f"""
            <b>Predictive Analysis Availability:</b><br/>
            <br/>
            Your dataset contains {len(numeric_cols)} numeric column(s). Predictive modeling typically requires at least 
            2 numeric columns - one to predict and others to use as predictors. Consider adding more quantitative metrics 
            to your Excel file to enable advanced predictive analytics and forecasting capabilities.
            """
        
        story.append(Paragraph(prediction_intro, body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # ============ DATA PATTERNS & TRENDS ============
        story.append(Paragraph("8.6 Patterns, Trends & What They Mean", subheading_style))
        
        if len(numeric_cols) > 0:
            patterns_text = f"""
            <b>Trend Analysis Across Your Data:</b><br/>
            <br/>
            By analyzing the progression of values in your Excel file, we can identify whether your metrics are 
            improving, declining, or staying stable. These trends reveal the underlying health of your operations 
            and help predict future performance.
            <br/><br/>
            """
            
            # Analyze trends for first few columns
            for col in numeric_cols[:3]:
                values = df[col].dropna().values
                if len(values) > 1:
                    x = np.arange(len(values[:100]))
                    slope, _, r_value, p_value, _ = stats.linregress(x, values[:100])
                    r_squared = r_value ** 2
                    
                    trend_direction = "UPWARD" if slope > 0 else "DOWNWARD"
                    trend_strength = "STRONG" if abs(r_value) > 0.7 else "MODERATE" if abs(r_value) > 0.3 else "WEAK"
                    
                    patterns_text += f"""
                    <b>{col} Trend Analysis:</b><br/>
                    This metric shows a <b>{trend_strength} {trend_direction} TREND</b> with R² = {r_squared:.3f}. 
                    """
                    
                    if slope > 0 and r_squared > 0.5:
                        patterns_text += f"""
                        This is excellent news - {col} is consistently increasing over time with {r_squared*100:.1f}% 
                        of the variation following this upward pattern. This strong positive trend suggests sustained 
                        growth or improvement. Continue current strategies and look for ways to accelerate this positive momentum.
                        """
                    elif slope < 0 and r_squared > 0.5:
                        patterns_text += f"""
                        This declining trend in {col} requires immediate attention. With {r_squared*100:.1f}% of the 
                        variation following this downward pattern, this is a systematic issue, not random fluctuation. 
                        Conduct root cause analysis and implement corrective measures within the next 30 days.
                        """
                    elif r_squared < 0.3:
                        patterns_text += f"""
                        {col} shows relatively stable performance without clear directional trends. This stability can be 
                        positive (consistent operations) or negative (lack of growth). Review whether this stability aligns 
                        with business objectives.
                        """
                    else:
                        patterns_text += f"""
                        {col} shows a {trend_direction.lower()} direction but with moderate strength. There's a pattern 
                        emerging that deserves monitoring. Track this metric closely over the next periods to confirm 
                        whether the trend strengthens or weakens.
                        """
                    
                    patterns_text += "<br/><br/>"
        
        story.append(Paragraph(patterns_text, body_style))
        
        story.append(PageBreak())
        
        # ============ COLUMN RELATIONSHIP ANALYSIS ============
        story.append(Paragraph("9. COLUMN RELATIONSHIP & INTERACTION ANALYSIS", heading_style))
        
        if len(numeric_cols) >= 2:
            story.append(Paragraph("9.1 Correlation Matrix & Relationships", subheading_style))
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols[:6]].corr()
            
            # Find strong relationships
            strong_relationships = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:
                        strong_relationships.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_val,
                            'strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong' if abs(corr_val) > 0.6 else 'Moderate',
                            'direction': 'Positive' if corr_val > 0 else 'Negative'
                        })
            
            strong_relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            strongest_corr = f"{strong_relationships[0]['correlation']:.3f}" if strong_relationships else 'N/A'
            relationship_text = f"""
            <b>Correlation Analysis Summary:</b><br/>
            The analysis examined {len(numeric_cols)} numeric columns to identify relationships and dependencies.<br/>
            <br/>
            <b>Key Findings:</b><br/>
            • Total Column Pairs Analyzed: {len(numeric_cols) * (len(numeric_cols) - 1) // 2}<br/>
            • Significant Relationships Found: {len(strong_relationships)}<br/>
            • Strongest Correlation: {strongest_corr}<br/>
            <br/>
            """
            story.append(Paragraph(relationship_text, body_style))
            
            if strong_relationships:
                # Create relationship table
                rel_data = [['Column 1', 'Column 2', 'Correlation', 'Strength', 'Type', 'Business Implication']]
                
                for rel in strong_relationships[:10]:
                    implication = ""
                    if rel['direction'] == 'Positive':
                        if rel['strength'] == 'Very Strong':
                            implication = "Highly predictive - use for forecasting"
                        elif rel['strength'] == 'Strong':
                            implication = "Can predict one from the other"
                        else:
                            implication = "Some predictive capability"
                    else:
                        if rel['strength'] == 'Very Strong':
                            implication = "Inverse relationship - monitor both"
                        elif rel['strength'] == 'Strong':
                            implication = "Trade-off relationship exists"
                        else:
                            implication = "Weak inverse relationship"
                    
                    rel_data.append([
                        rel['col1'][:15],
                        rel['col2'][:15],
                        f"{rel['correlation']:.3f}",
                        rel['strength'],
                        rel['direction'],
                        implication
                    ])
                
                rel_table = Table(rel_data, colWidths=[1.2*inch, 1.2*inch, 0.8*inch, 0.9*inch, 0.8*inch, 1.6*inch])
                rel_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8B5CF6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.95, 0.92, 0.98)),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.98, 0.96, 0.99)])
                ]))
                story.append(rel_table)
                story.append(Spacer(1, 0.2*inch))
                
                # Create correlation heatmap
                try:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
                    
                    ax.set_xticks(np.arange(len(corr_matrix.columns)))
                    ax.set_yticks(np.arange(len(corr_matrix.columns)))
                    ax.set_xticklabels([col[:15] for col in corr_matrix.columns], rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels([col[:15] for col in corr_matrix.columns], fontsize=8)
                    
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15, fontsize=9)
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.columns)):
                            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=7)
                    
                    ax.set_title('Correlation Heatmap - Column Relationships', fontsize=12, fontweight='bold', pad=20)
                    plt.tight_layout()
                    
                    heatmap_path = f"/tmp/heatmap_{dataset_id}.png"
                    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    if os.path.exists(heatmap_path):
                        img = Image(heatmap_path, width=6*inch, height=4.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    logger.error(f"Heatmap creation error: {e}")
                
                # Detailed relationship insights
                story.append(Paragraph("9.2 Detailed Relationship Insights", subheading_style))
                
                for i, rel in enumerate(strong_relationships[:5], 1):
                    detailed_insight = f"""
                    <b>Relationship #{i}: {rel['col1']} ↔ {rel['col2']}</b><br/>
                    • Correlation: {rel['correlation']:.3f} ({rel['strength']} {rel['direction']})<br/>
                    • Type: {('Move together' if rel['direction'] == 'Positive' else 'Move in opposite directions')}<br/>
                    • Predictive Power: {('High' if abs(rel['correlation']) > 0.7 else 'Moderate' if abs(rel['correlation']) > 0.5 else 'Limited')}<br/>
                    • Application: {('Use as leading/lagging indicators' if abs(rel['correlation']) > 0.6 else 'Monitor together')}<br/>
                    """
                    story.append(Paragraph(detailed_insight, body_style))
                    story.append(Spacer(1, 0.15*inch))
            else:
                story.append(Paragraph(
                    "No significant correlations detected. Columns appear to operate independently.",
                    body_style
                ))
        
        story.append(PageBreak())
        
        # ============ PERFORMANCE PREDICTION (NEW) ============
        story.append(Paragraph("10. PERFORMANCE PREDICTION & FEATURE IMPORTANCE", heading_style))
        
        if len(numeric_cols) >= 2:
            story.append(Paragraph("10.1 Predictive Performance Analysis", subheading_style))
            
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                target_col = numeric_cols[0]
                feature_cols = numeric_cols[1:min(6, len(numeric_cols))]
                
                data_clean = df[[target_col] + feature_cols].dropna()
                
                if len(data_clean) >= 20:
                    X = data_clean[feature_cols]
                    y = data_clean[target_col]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf_model.fit(X_train, y_train)
                    
                    y_pred = rf_model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    feature_importance = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    performance_text = f"""
                    <b>Predictive Model Performance:</b><br/>
                    • Target Variable: {target_col}<br/>
                    • Features Used: {len(feature_cols)}<br/>
                    • R² Score: {r2:.3f} ({('Excellent' if r2 > 0.8 else 'Good' if r2 > 0.6 else 'Moderate')} accuracy)<br/>
                    • Mean Absolute Error: {mae:.2f}<br/>
                    <br/>
                    <b>Interpretation:</b> {int(r2 * 100)}% of variance in {target_col} explained by other columns.<br/>
                    """
                    story.append(Paragraph(performance_text, body_style))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Feature Importance Table
                    story.append(Paragraph("10.2 Feature Importance Rankings", subheading_style))
                    
                    fi_data = [['Rank', 'Feature', 'Importance', 'Impact', 'Insight']]
                    
                    for idx, row in feature_importance.iterrows():
                        rank = len(fi_data)
                        importance = row['importance']
                        impact = 'Critical' if importance > 0.3 else 'High' if importance > 0.2 else 'Medium' if importance > 0.1 else 'Low'
                        insight = 'Primary driver' if importance > 0.25 else 'Significant factor' if importance > 0.15 else 'Moderate influence' if importance > 0.08 else 'Minor factor'
                        
                        fi_data.append([
                            str(rank),
                            row['feature'][:20],
                            f"{importance:.3f}",
                            impact,
                            insight
                        ])
                    
                    fi_table = Table(fi_data, colWidths=[0.5*inch, 1.8*inch, 1*inch, 0.9*inch, 2.3*inch])
                    fi_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.9, 0.98, 0.94)),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('FONTSIZE', (0, 1), (-1, -1), 8)
                    ]))
                    story.append(fi_table)
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Feature importance chart
                    try:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        colors_bar = ['#10B981' if imp > 0.2 else '#3B82F6' if imp > 0.1 else '#94A3B8' 
                                     for imp in feature_importance['importance']]
                        
                        bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors_bar)
                        ax.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
                        ax.set_ylabel('Feature', fontsize=10, fontweight='bold')
                        ax.set_title(f'Feature Importance for {target_col}', fontsize=12, fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)
                        
                        for i, (bar, imp) in enumerate(zip(bars, feature_importance['importance'])):
                            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{imp:.3f}', va='center', fontsize=8, fontweight='bold')
                        
                        plt.tight_layout()
                        fi_chart_path = f"/tmp/feature_importance_{dataset_id}.png"
                        plt.savefig(fi_chart_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        if os.path.exists(fi_chart_path):
                            img = Image(fi_chart_path, width=6*inch, height=3.5*inch)
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        logger.error(f"Feature importance chart error: {e}")
                    
                    # Actual vs Predicted plot
                    try:
                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        ax.scatter(y_test, y_pred, alpha=0.6, color='#4F46E5', s=60, edgecolors='black', linewidth=0.5)
                        
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
                        
                        ax.set_xlabel(f'Actual {target_col}', fontsize=11, fontweight='bold')
                        ax.set_ylabel(f'Predicted {target_col}', fontsize=11, fontweight='bold')
                        ax.set_title(f'Prediction Accuracy\nR² = {r2:.3f}', fontsize=11, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        pred_chart_path = f"/tmp/prediction_{dataset_id}.png"
                        plt.savefig(pred_chart_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        if os.path.exists(pred_chart_path):
                            img = Image(pred_chart_path, width=5.5*inch, height=4*inch)
                            story.append(img)
                    except Exception as e:
                        logger.error(f"Prediction plot error: {e}")
                        
                else:
                    story.append(Paragraph(
                        f"Insufficient data ({len(data_clean)} records) for predictive modeling. Minimum 20 records required.",
                        body_style
                    ))
            except Exception as e:
                logger.error(f"Performance prediction error: {e}")
                story.append(Paragraph(
                    "Performance prediction could not be completed due to data limitations.",
                    body_style
                ))
        else:
            story.append(Paragraph(
                "Performance prediction requires at least 2 numeric columns.",
                body_style
            ))
        
        story.append(PageBreak())
        
        # ============ CONTEXT & BENCHMARKS ============
        story.append(Paragraph("11. CONTEXT & PERFORMANCE BENCHMARKS", heading_style))
        
        benchmark_text = f"""
        <b>Industry Context:</b><br/>
        This analysis provides insights relative to standard analytical benchmarks and best practices.<br/>
        <br/>
        <b>Data Quality Benchmarks:</b><br/>
        • Completeness Target: 95% or higher<br/>
        • Your Score: {100-missing_pct:.1f}% {'✓ Exceeds' if (100-missing_pct) >= 95 else '✗ Below'} benchmark<br/>
        • Industry Average: 92-96%<br/>
        <br/>
        <b>Statistical Reliability:</b><br/>
        • Minimum Sample Size for Confidence: 30 records<br/>
        • Your Dataset: {len(df):,} records {'✓' if len(df) >= 30 else '✗'}<br/>
        • Confidence Level: {('High - Large sample enables robust analysis' if len(df) > 1000 else 'Good - Adequate sample size' if len(df) > 100 else 'Moderate - Consider expanding dataset')}<br/>
        <br/>
        <b>Variability Assessment:</b><br/>
        """
        
        if len(numeric_cols) > 0:
            avg_cv = np.mean([df[col].std() / df[col].mean() * 100 for col in numeric_cols if df[col].mean() != 0])
            benchmark_text += f"""
            • Average Coefficient of Variation: {avg_cv:.1f}%<br/>
            • Interpretation: {('Low variability - Stable metrics' if avg_cv < 20 else 'Moderate variability - Some fluctuation' if avg_cv < 50 else 'High variability - Volatile metrics')}<br/>
            • Benchmark Range: 15-35% for stable business metrics<br/>
            """
        
        story.append(Paragraph(benchmark_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # ============ CONCLUSION & KEY TAKEAWAYS ============
        story.append(Paragraph("12. CONCLUSION & KEY TAKEAWAYS", heading_style))
        
        story.append(Paragraph("Executive Summary", subheading_style))
        
        conclusion_text = f"""
        <b>Analysis Overview:</b><br/>
        This comprehensive analysis of {dataset['name']} has revealed {('positive trends and growth opportunities' if trend_direction == 'upward' else 'areas requiring strategic attention' if trend_direction == 'downward' else 'stable patterns with optimization potential')}. 
        The dataset comprising {len(df):,} records across {len(df.columns)} dimensions provides {'robust' if len(df) > 1000 else 'adequate'} statistical foundation for decision-making.
        <br/><br/>
        <b>Statistical Assessment:</b><br/>
        • Data Quality Score: {100-missing_pct:.1f}% ({('Excellent' if missing_pct < 5 else 'Good' if missing_pct < 10 else 'Adequate')})<br/>
        • Trend Analysis: {trend_strength.capitalize()} {trend_direction} pattern detected<br/>
        • Forecast Outlook: {('Favorable growth trajectory' if forecast_trend == 'increasing' else 'Declining trend - intervention needed' if forecast_trend == 'decreasing' else 'Stable outlook')}<br/>
        • Overall Health: {('Strong' if missing_pct < 5 and len(df) > 500 else 'Good' if missing_pct < 10 else 'Needs Improvement')}<br/>
        """
        story.append(Paragraph(conclusion_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Key Takeaways
        story.append(Paragraph("🎯 Key Carry-Away Points", subheading_style))
        
        carry_away_style = ParagraphStyle(
            'CarryAwayBox',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1F2937'),
            spaceAfter=8,
            leading=18,
            leftIndent=15,
            rightIndent=10,
            fontName='Helvetica'
        )
        
        for i, takeaway in enumerate(ai_insights['key_takeaways'][:5], 1):
            story.append(Paragraph(f"<b>{i}.</b> {takeaway}", carry_away_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Action Items
        story.append(Paragraph("📋 Immediate Next Steps", subheading_style))
        
        action_items = f"""
        <b>Priority Actions (Next 7-14 Days):</b><br/>
        1. Review automated insights with key stakeholders and validate against business context<br/>
        2. {'Address data quality issues to achieve >95% completeness' if missing_pct > 5 else 'Maintain current data quality standards with regular audits'}<br/>
        3. {'Implement corrective measures for declining trends' if trend_direction == 'downward' else 'Capitalize on positive momentum' if trend_direction == 'upward' else 'Test new optimization strategies'}<br/>
        4. {ai_insights['strategic_recommendations'][0] if ai_insights['strategic_recommendations'] else 'Continue monitoring key metrics'}<br/>
        5. Schedule follow-up analysis in 30-60 days to track progress<br/>
        <br/>
        <b>Long-Term Strategic Focus:</b><br/>
        • Establish automated monitoring dashboards for real-time insights<br/>
        • Build predictive models for proactive decision-making<br/>
        • Create feedback loops to continuously improve data quality<br/>
        • Integrate insights into strategic planning processes<br/>
        """
        story.append(Paragraph(action_items, body_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", footer_style))
        story.append(Paragraph("Memat Data Analytics Platform | Automated Analysis System", footer_style))
        story.append(Paragraph("This report includes automated intelligent analysis with relationship & performance predictions", footer_style))
        
        # Build PDF
        doc.build(story)
        
        if not os.path.exists(pdf_filename):
            raise Exception(f"PDF file was not created: {pdf_filename}")
        
        logger.info(f"Enhanced PDF report generated successfully for dataset {dataset_id}")
        
        return FileResponse(pdf_filename, filename=f"intelligent_analytics_report_{dataset['name']}.pdf", media_type="application/pdf")
        
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
from comparison_router import router as comparison_router
app.include_router(api_router)
app.include_router(comparison_router)

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
