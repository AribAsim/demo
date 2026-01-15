from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')
# Ensure HF symlink warning is disabled even if not in .env
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import re
import base64
import io
import requests
from bson import ObjectId
from transformers import pipeline
from PIL import Image
import numpy as np

# Safe environment variable loader
def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.environ.get(name, default or "")
    if isinstance(value, str):
        value = value.strip().strip('\'"')
    return value

# MongoDB connection with safe env loading
mongo_url = _get_env('MONGO_URL', 'mongodb://localhost:27017')
db_name = _get_env('DB_NAME', 'safebrowse_db')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Security setup
SECRET_KEY = _get_env('JWT_SECRET_KEY', 'safebrowse-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Create the main app
app = FastAPI(title="SafeBrowse AI API")
api_router = APIRouter(prefix="/api")

# ==================== IMPROVED AI MODEL INITIALIZATION ====================

logger = logging.getLogger("SafeBrowseAI")

# Multi-Model Approach for Better Accuracy
class NSFWDetector:
    def __init__(self):
        self.image_models = []
        self.text_models = []
        
        # Load Image Models (multiple for voting system)
        try:
            logger.info("Loading Image AI Models...")
            # Model 1: Falconsai (Fast, good accuracy)
            self.image_models.append({
                'name': 'falconsai',
                'model': pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                    device=-1
                ),
                'weight': 1.0
            })
            
            # Model 2: AdamCodd (Higher accuracy)
            try:
                self.image_models.append({
                    'name': 'adamcodd',
                    'model': pipeline(
                        "image-classification",
                        model="AdamCodd/vit-base-nsfw-detector",
                        device=-1
                    ),
                    'weight': 1.5
                })
            except Exception as e:
                logger.warning(f"Could not load secondary image model (likely RAM limits): {e}")
            
            logger.info(f"Loaded {len(self.image_models)} image models")
        except Exception as e:
            logger.error(f"Failed to load Image AI: {e}")
        
        # Load Text Models
        try:
            logger.info("Loading Text AI Models...")
            # Primary Model: Toxic Comment (The one we know works well on limited RAM)
            self.text_models.append({
                'name': 'toxic-comment',
                'model': pipeline(
                    "text-classification",
                    model="martin-ha/toxic-comment-model",
                    device=-1
                ),
                'weight': 1.0
            })
            
            # Secondary Model (Optional if RAM allows)
            try:
                self.text_models.append({
                    'name': 'unbiased-roberta',
                    'model': pipeline(
                        "text-classification",
                        model="unitary/unbiased-toxic-roberta",
                        device=-1
                    ),
                    'weight': 1.2
                })
            except Exception as e:
                logger.warning(f"Could not load secondary text model: {e}")
                
            logger.info(f"Loaded {len(self.text_models)} text models")
        except Exception as e:
            logger.error(f"Failed to load Text AI: {e}")

# Initialize the global detector instance
nsfw_detector = NSFWDetector()

# ==================== MODELS ====================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class ChildProfile(BaseModel):
    name: str
    age: int
    maturity_level: Optional[str] = None  # 'strict', 'moderate', 'lenient'
    blocked_sites: List[str] = Field(default_factory=list)
    whitelisted_sites: List[str] = Field(default_factory=list)

class ChildProfileResponse(BaseModel):
    id: str
    parent_id: str
    name: str
    age: int
    maturity_level: str
    blocked_sites: List[str]
    whitelisted_sites: List[str]
    created_at: datetime

class ContentAnalysisRequest(BaseModel):
    profile_id: str
    content_type: str  # 'text', 'image', 'url'
    content: str  # text content, base64 image, or URL
    context: Optional[str] = None

class ContentAnalysisResponse(BaseModel):
    is_safe: bool
    confidence: float
    reasons: List[str]
    blocked: bool

class ContentLog(BaseModel):
    profile_id: str
    content_type: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    is_safe: bool
    confidence: float
    reasons: List[str]
    content_snippet: str
    url: Optional[str] = None

class ContentLogResponse(BaseModel):
    id: str
    profile_id: str
    profile_name: str
    content_type: str
    detected_at: datetime
    is_safe: bool
    confidence: float
    reasons: List[str]
    content_snippet: str
    url: Optional[str] = None

class PINUpdate(BaseModel):
    pin: str

# ==================== SECURITY HELPERS ====================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# ==================== IMPROVED CONTENT FILTERING ====================

# Enhanced keyword lists with regex patterns
EXPLICIT_KEYWORDS = [
    r'\bporn', r'\bxxx\b', r'\bsex\b', r'\bnude\b', r'\bnaked\b', 
    r'\bnsfw\b', r'\bhentai\b', r'\berotic\b', 
    r'\bfuck', r'\bshit\b', r'\bbitch\b', r'\bdick\b',
    r'\bpussy\b', r'\bcock\b', r'\bcum\b', 
    r'\bmasturbat', r'\borgy\b', r'\brape\b',
    r'adult content', r'adult site', r'adult movie', r'adult video'
]

VIOLENCE_KEYWORDS = [
    r'\bkill\b', r'\bmurder\b', r'\bsuicide\b', r'\bdeath\b',
    r'\bgore\b', r'\btorture\b', r'\bbomb\b', r'\bweapon\b',
    r'\bgun\b', r'\bknife\b'
]

DRUG_KEYWORDS = [
    r'\bcocaine\b', r'\bheroin\b', r'\bmeth\b', r'\bdrug deal',
    r'\bget high\b'
]

# Meta keywords that are often safe in search suggestions but unsafe on regular sites
META_KEYWORDS = ['adult content', 'adult site', 'adult movie', 'adult video', 'explicit content']

# Compile patterns for efficiency
EXPLICIT_RE = re.compile('|'.join(EXPLICIT_KEYWORDS), re.IGNORECASE)
VIOLENCE_RE = re.compile('|'.join(VIOLENCE_KEYWORDS), re.IGNORECASE)
DRUG_RE = re.compile('|'.join(DRUG_KEYWORDS), re.IGNORECASE)

# Age-based thresholds
AGE_THRESHOLDS = {
    'text': {
        'strict': {'ai': 0.15, 'keyword': 1},      # Age 0-8
        'moderate': {'ai': 0.40, 'keyword': 1},    # Age 9-12
        'lenient': {'ai': 0.70, 'keyword': 2},     # Age 13+
    },
    'image': {
        'strict': {'ai': 0.20, 'votes': 1},        
        'moderate': {'ai': 0.50, 'votes': 1},      
        'lenient': {'ai': 0.75, 'votes': 2},       
    }
}

def get_age_category(age: int) -> str:
    if age <= 8: return 'strict'
    if age <= 12: return 'moderate'
    return 'lenient'

def is_search_engine(url: str) -> bool:
    if not url: return False
    url_lower = url.lower()
    search_engines = ['google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com', 'baidu.com', 'chrome://newtab']
    return any(se in url_lower for se in search_engines)

def analyze_text_content(text: str, age: int, url_context: Optional[str] = None) -> Tuple[bool, float, List[str]]:
    """
    Multi-model text analysis with ensemble voting and context awareness
    """
    if not text or not text.strip():
        return True, 0.0, []

    age_cat = get_age_category(age)
    thresholds = AGE_THRESHOLDS['text'][age_cat]
    reasons = []
    
    # 1. Layer: Keyword Match
    explicit_matches = EXPLICIT_RE.findall(text)
    violence_matches = VIOLENCE_RE.findall(text)
    drug_matches = DRUG_RE.findall(text)
    
    # Context-aware filtering decision:
    on_search_engine = is_search_engine(url_context)
    filtered_explicit = []
    if on_search_engine:
        for match in set(explicit_matches):
            if match.lower() not in META_KEYWORDS:
                filtered_explicit.append(match)
    else:
        filtered_explicit = explicit_matches

    if filtered_explicit:
        reasons.append(f"Explicit terms detected: {set(filtered_explicit)}")
    if violence_matches:
        reasons.append(f"Violence-related terms: {set(violence_matches)}")
    if drug_matches:
        reasons.append(f"Drug-related terms: {set(drug_matches)}")

    # 2. Layer: AI Ensemble
    ai_votes_toxic = 0.0
    ai_votes_safe = 0.0
    ai_scores = []
    
    for text_model in nsfw_detector.text_models:
        try:
            res = text_model['model'](text[:500])[0]
            label = res['label'].lower()
            score = res['score']
            
            # Weighted Voting
            if label in ['toxic', 'label_1', 'offensive'] or score > thresholds['ai']:
                ai_votes_toxic += text_model['weight']
                reasons.append(f"AI ({text_model['name']}): Toxic ({score:.2f})")
            else:
                ai_votes_safe += text_model['weight']
            
            ai_scores.append(score)
        except:
            continue
    
    avg_ai_confidence = sum(ai_scores) / len(ai_scores) if ai_scores else 0.0
    
    # Decision Logic
    is_safe = True
    
    # Hard block on hard keywords (if not on search engine bypass)
    if len(filtered_explicit) >= thresholds['keyword']:
        is_safe = False
    
    # Block on AI sentiment mismatch
    if ai_votes_toxic > ai_votes_safe:
        is_safe = False
        
    # Block on violence/drugs for younger kids
    if age <= 12 and (len(violence_matches) > 0 or len(drug_matches) > 0):
        is_safe = False

    return is_safe, avg_ai_confidence, reasons

def analyze_image_content(content: str, age: int) -> Tuple[bool, float, List[str]]:
    """
    Image analysis with multi-model voting.
    Supports both base64 strings and image URLs.
    """
    if not nsfw_detector.image_models:
        return True, 0.0, ["Image AI not ready"]

    try:
        img_data = None
        
        # Check if content is a URL
        if content.startswith(('http://', 'https://')):
            try:
                response = requests.get(content, timeout=10)
                if response.status_code == 200:
                    img_data = response.content
            except Exception as e:
                logger.warning(f"Failed to fetch image from URL {content}: {e}")
        
        # If not a URL or URL fetch failed, try base64
        if img_data is None:
            if "base64," in content:
                content = content.split("base64,")[1]
            try:
                img_data = base64.b64decode(content)
            except:
                return True, 0.0, ["Invalid image format"]

        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # Resize for performance
        img.thumbnail((400, 400))
        
        age_cat = get_age_category(age)
        thresholds = AGE_THRESHOLDS['image'][age_cat]
        
        nsfw_votes = 0.0
        safe_votes = 0.0
        scores = []
        reasons = []
        
        for img_model in nsfw_detector.image_models:
            results = img_model['model'](img)
            nsfw_score = 0.0
            for res in results:
                if any(k in res['label'].lower() for k in ['nsfw', 'porn', 'hentai', 'sexy', 'nudity', 'explicit', 'adult']):
                    nsfw_score = max(nsfw_score, res['score'])
            
            if nsfw_score > thresholds['ai']:
                nsfw_votes += img_model['weight']
                reasons.append(f"AI ({img_model['name']}): NSFW ({nsfw_score:.2f})")
            else:
                safe_votes += img_model['weight']
            
            scores.append(nsfw_score)
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        is_safe = nsfw_votes < thresholds['votes']
        
        return is_safe, avg_score, reasons
    except Exception as e:
        logger.error(f"Image Analysis Error: {e}")
        return True, 0.0, [str(e)]

def analyze_url(url: str, age: int) -> Tuple[bool, float, List[str]]:
    """
    URL analysis with domain-level blocking and age thresholds
    """
    url_lower = url.lower()
    reasons = []
    score = 0
    
    # Known adult domains (Expanded)
    adult_domains = [
        'pornhub', 'xvideos', 'xnxx', 'redtube', 'youporn', 
        'xhamster', 'hentai', 'onlyfans', 'chaturbate'
    ]
    for domain in adult_domains:
        if domain in url_lower:
            reasons.append(f"Adult website: {domain}")
            score += 100
    
    # Keyword detection in URL (Using regex)
    url_keywords = EXPLICIT_RE.findall(url_lower)
    if url_keywords:
        reasons.append(f"Explicit keywords in URL: {set(url_keywords)}")
        score += 50
    
    # Violence detection in URL
    val_keywords = VIOLENCE_RE.findall(url_lower)
    if val_keywords and age <= 12:
        reasons.append(f"Restricted content in URL: {set(val_keywords)}")
        score += 40

    confidence = min(score / 100.0, 1.0)
    
    # Age-based threshold for URL risk
    age_cat = get_age_category(age)
    limit = 30 if age_cat == 'strict' else (50 if age_cat == 'moderate' else 70)
    
    is_safe = score < limit
    
    return is_safe, confidence, reasons

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/signup", response_model=Token)
async def signup(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_dict = {
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "name": user_data.name,
        "pin": None,
        "created_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_dict)
    user_id = str(result.inserted_id)
    
    # Create token
    access_token = create_access_token(data={"sub": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name
        }
    }

@api_router.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    user_id = str(user["_id"])
    access_token = create_access_token(data={"sub": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": user["email"],
            "name": user["name"],
            "pin": user.get("pin")
        }
    }

@api_router.get("/auth/me")
async def get_me(current_user = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "name": current_user["name"],
        "pin": current_user.get("pin")
    }

@api_router.put("/auth/pin")
async def update_pin(pin_data: PINUpdate, current_user = Depends(get_current_user)):
    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"pin": pin_data.pin}}
    )
    return {"message": "PIN updated successfully"}

# ==================== PROFILE ROUTES ====================

@api_router.post("/profiles", response_model=ChildProfileResponse)
async def create_profile(profile: ChildProfile, current_user = Depends(get_current_user)):
    # Determine maturity level based on age if not provided
    maturity = profile.maturity_level
    if not maturity:
        if profile.age <= 8:
            maturity = 'strict'
        elif profile.age <= 12:
            maturity = 'moderate'
        else:
            maturity = 'lenient'
    
    profile_dict = {
        "parent_id": str(current_user["_id"]),
        "name": profile.name,
        "age": profile.age,
        "maturity_level": maturity,
        "blocked_sites": profile.blocked_sites or [],
        "whitelisted_sites": profile.whitelisted_sites or [],
        "created_at": datetime.utcnow()
    }
    
    result = await db.profiles.insert_one(profile_dict)
    profile_dict["id"] = str(result.inserted_id)
    
    return profile_dict

@api_router.get("/profiles", response_model=List[ChildProfileResponse])
async def get_profiles(current_user = Depends(get_current_user)):
    profiles = await db.profiles.find({"parent_id": str(current_user["_id"])}).to_list(100)
    
    result = []
    for profile in profiles:
        profile["id"] = str(profile["_id"])
        result.append(profile)
    
    return result

@api_router.get("/profiles/{profile_id}", response_model=ChildProfileResponse)
async def get_profile(profile_id: str, current_user = Depends(get_current_user)):
    profile = await db.profiles.find_one({
        "_id": ObjectId(profile_id),
        "parent_id": str(current_user["_id"])
    })
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    profile["id"] = str(profile["_id"])
    return profile

@api_router.put("/profiles/{profile_id}", response_model=ChildProfileResponse)
async def update_profile(profile_id: str, profile: ChildProfile, current_user = Depends(get_current_user)):
    result = await db.profiles.update_one(
        {"_id": ObjectId(profile_id), "parent_id": str(current_user["_id"])},
        {"$set": profile.dict()}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    updated_profile = await db.profiles.find_one({"_id": ObjectId(profile_id)})
    updated_profile["id"] = str(updated_profile["_id"])
    
    return updated_profile

@api_router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str, current_user = Depends(get_current_user)):
    result = await db.profiles.delete_one({
        "_id": ObjectId(profile_id),
        "parent_id": str(current_user["_id"])
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Also delete associated logs
    await db.logs.delete_many({"profile_id": profile_id})
    
    return {"message": "Profile deleted successfully"}

# ==================== CONTENT ANALYSIS ROUTES ====================

@api_router.post("/content/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(request: ContentAnalysisRequest):
    # Get profile to determine age-based thresholds
    profile = await db.profiles.find_one({"_id": ObjectId(request.profile_id)})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    age = profile["age"]
    is_safe = True
    confidence = 0.0
    reasons = []
    
    # Analyze based on content type
    if request.content_type == "text":
        is_safe, confidence, reasons = analyze_text_content(
            request.content, age, request.context
        )
    elif request.content_type == "url":
        is_safe, confidence, reasons = analyze_url(request.content, age)
    elif request.content_type == "image":
        is_safe, confidence, reasons = analyze_image_content(request.content, age)
    
    # Log the detection if harmful
    if not is_safe:
        log_dict = {
            "profile_id": request.profile_id,
            "content_type": request.content_type,
            "detected_at": datetime.utcnow(),
            "is_safe": is_safe,
            "confidence": confidence,
            "reasons": reasons,
            "content_snippet": request.content[:200] if request.content_type == "text" else "[Content blocked]",
            "url": request.context
        }
        await db.logs.insert_one(log_dict)
    
    return {
        "is_safe": is_safe,
        "confidence": confidence,
        "reasons": reasons,
        "blocked": not is_safe
    }

# ==================== LOGS ROUTES ====================

@api_router.get("/logs", response_model=List[ContentLogResponse])
async def get_logs(
    profile_id: Optional[str] = None,
    limit: int = 50,
    current_user = Depends(get_current_user)
):
    # Build query
    query = {}
    if profile_id:
        # Verify profile belongs to user
        profile = await db.profiles.find_one({
            "_id": ObjectId(profile_id),
            "parent_id": str(current_user["_id"])
        })
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        query["profile_id"] = profile_id
    else:
        # Get all profiles for user
        profiles = await db.profiles.find({"parent_id": str(current_user["_id"])}).to_list(100)
        profile_ids = [str(p["_id"]) for p in profiles]
        query["profile_id"] = {"$in": profile_ids}
    
    # Get logs
    logs = await db.logs.find(query).sort("detected_at", -1).limit(limit).to_list(limit)
    
    # Enrich with profile names
    result = []
    for log in logs:
        profile = await db.profiles.find_one({"_id": ObjectId(log["profile_id"])})
        log["id"] = str(log["_id"])
        log["profile_name"] = profile["name"] if profile else "Unknown"
        # Remove the _id field to avoid serialization issues
        del log["_id"]
        result.append(log)
    
    return result

@api_router.get("/logs/search", response_model=List[ContentLogResponse])
async def search_logs(
    keyword: str,
    current_user = Depends(get_current_user)
):
    # Get all profiles for user
    profiles = await db.profiles.find({"parent_id": str(current_user["_id"])}).to_list(100)
    profile_ids = [str(p["_id"]) for p in profiles]
    
    # Search in logs
    logs = await db.logs.find({
        "profile_id": {"$in": profile_ids},
        "$or": [
            {"content_snippet": {"$regex": keyword, "$options": "i"}},
            {"reasons": {"$regex": keyword, "$options": "i"}},
            {"url": {"$regex": keyword, "$options": "i"}}
        ]
    }).sort("detected_at", -1).limit(50).to_list(50)
    
    result = []
    for log in logs:
        profile = await db.profiles.find_one({"_id": ObjectId(log["profile_id"])})
        log["id"] = str(log["_id"])
        log["profile_name"] = profile["name"] if profile else "Unknown"
        # Remove the _id field to avoid serialization issues
        del log["_id"]
        result.append(log)
    
    return result

# ==================== MAIN APP SETUP ====================

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
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
