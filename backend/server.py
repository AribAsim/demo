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
import asyncio
from urllib.parse import urlparse
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


# Safe environment variable loader
def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.environ.get(name, default or "")
    if isinstance(value, str):
        value = value.strip().strip('\'"')
    return value

# MongoDB connection with safe env loading
mongo_url = _get_env('MONGO_URL', 'mongodb://localhost:27017')
db_name = _get_env('DB_NAME', 'safebrowse_db')

async def connect_to_mongo():
    try:
        temp_client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
        # Verify connection
        await temp_client.admin.command('ping')
        print(f"✅ Successfully connected to MongoDB at {mongo_url}")
        return temp_client
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("Running in limited mode (Database unavailable)")
        return AsyncIOMotorClient(mongo_url) # Fallback

client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Security setup
SECRET_KEY = _get_env('JWT_SECRET_KEY', 'safebrowse-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

from contextlib import asynccontextmanager

# ... (imports)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    try:
        # We can use the existing global 'client' or re-initialize it here
        # For simplicity with existing code, we verify the connection
        await client.admin.command('ping')
        logger.info(f"✅ Successfully connected to MongoDB at {mongo_url}")
    except Exception as e:
        logger.warning(f"❌ Failed to connect to MongoDB: {e}")
        logger.warning("Running in limited mode (Database unavailable)")
    
    yield
    
    # Shutdown: Close connection
    client.close()
    logger.info("MongoDB connection closed")

# Create the main app
app = FastAPI(title="SafeBrowse AI API", lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# ==================== IMPROVED AI MODEL INITIALIZATION ====================

logger = logging.getLogger("SafeBrowseAI")

# Multi-Model Approach for Better Accuracy
class NSFWDetector:
    def __init__(self):
        self.image_models = []
        self.text_models = []
        
        # Determine device: 0 for GPU (CUDA), -1 for CPU
        self.device = 0 if HAS_CUDA else -1
        logger.info(f"🚀 AI Inference running on: {'GPU (CUDA)' if HAS_CUDA else 'CPU'}")
        
        # Load Image Models (multiple for voting system)
        try:
            logger.info("Loading Image AI Models...")
            # Model 1: Falconsai (Fast, good accuracy)
            self.image_models.append({
                'name': 'falconsai',
                'model': pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                    device=self.device
                ),
                'weight': 1.0
            })
            
            # Model 2: AdamCodd (Higher accuracy) - GPU ACCELERATED
            try:
                self.image_models.append({
                    'name': 'adamcodd',
                    'model': pipeline(
                        "image-classification",
                        model="AdamCodd/vit-base-nsfw-detector",
                        device=self.device
                    ),
                    'weight': 1.5
                })
            except Exception as e:
                logger.warning(f"Could not load secondary image model: {e}")
            
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
                    device=self.device
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
                        device=self.device
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

class InsightItem(BaseModel):
    title: str
    description: str
    sentiment: str  # 'positive', 'neutral', 'negative'

class WellbeingResponse(BaseModel):
    period_days: int
    total_scans: int
    safety_score: float
    blocked_count: int
    top_blocked_categories: List[Dict[str, Any]]
    insights: List[InsightItem]

class DailyStat(BaseModel):
    date: str  # YYYY-MM-DD
    screen_time_minutes: int
    unsafe_count: int
    
class DigitalWellbeingResponse(BaseModel):
    profile_id: str
    period_days: int
    total_screen_time_minutes: int
    avg_daily_minutes: int
    daily_stats: List[DailyStat]
    unsafe_detections_total: int

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
    r'adult content', r'adult site', r'adult movie', r'adult video',
    # Contextual Explicit Phrases (to avoid blocking single words like 'hot')
    'hot girl', 'hot girls', 'hot babe', 'hot babes', 'hot sex', 'hot women', 
    'hot milf', 'hot teen', 'hot teens', 'hot wife', 'hot wives', 'hot mom', 'hot moms',
    'strip club', 'strip tease', 'hard core', 'hardcore'
]

VIOLENCE_KEYWORDS = [
    r'\bkill(?!er whale)\b', r'\bmurder\b', r'\bsuicide\b', 
    r'\btorture\b', r'\bgore\b', r'\bmutilat',
    r'\bterroris', r'\bbomb\b', r'\bexplosive',
    r'\bshoot(?!ing star)\b', r'\bstab\b', r'\bstrangle\b',
    r'\bbehead\b', r'\bdecapitat', r'\bdismember',
    r'\bblood\b', r'\bbleed\b', r'\bcorpse\b', r'\bcadavre\b',
    r'\bmassacre\b', r'\bslaughter\b', r'\bassault\b', r'\bbeating\b',
    r'\bfight\b', r'\battack\b', r'\bweapon\b', r'\bgun\b', r'\bknife\b',
    r'\bfirearm\b', r'\bpistol\b', r'\brifle\b', r'\bammo\b',
    r'\bwar\b', r'\bbattle\b', r'\bcombat\b'
]

DRUG_KEYWORDS = [
    r'\bcocaine\b', r'\bheroin\b', r'\bmeth\b', r'\bfentanyl\b',
    r'\bdrug deal', r'\bget high\b', r'\bsnort\b', r'\bweed\b',
    r'\bcannabis\b', r'\bmarijuana\b', r'\blsd\b', r'\bacid\b',
    r'\becstasy\b', r'\bmolly\b', r'\bpill\b', r'\boverdose\b',
    r'\baddict\b', r'\bneedle\b', r'\bsyringe\b'
]

GAMBLING_KEYWORDS = [
    r'\bcasino\b', r'\bslot machine\b', r'\bpoker\b', r'\broulette\b',
    r'\bbetting\b', r'\bjackpot\b', r'\bblackjack\b', r'\bonline bet',
    r'\bgamble', r'\bgambling\b', r'\blottery\b', r'\braffle\b',
    r'\bwager\b', r'\bbookie\b', r'\bcard game\b', r'\bno limit\b',
    r'\btexas holdem\b', r'\bbet365\b', r'\bdraftkings\b'
]

SELF_HARM_KEYWORDS = [
    r'\bcut myself\b', r'\bcutting myself\b', r'\bkill myself\b',
    r'\bwant to die\b', r'\banorexia\b', r'\bbulimia\b', r'\bpro-ana\b',
    r'\bself-harm\b', r'\bhow to commit suicide\b',
    r'\bhang myself\b', r'\bend it all\b', r'\bsolitary\b',
    r'\bdepression\b', r'\bhopeless\b', r'\bworthless\b'
]

CRIME_KEYWORDS = [
    r'\bcrime\b', r'\bcriminal\b', r'\brobbery\b', r'\btheft\b',
    r'\bsteal\b', r'\bfraud\b', r'\bscam\b', r'\bhack\b',
    r'\billegal\b', r'\bfelony\b', r'\bjail\b', r'\bprison\b',
    r'\barrest\b', r'\bgang\b', r'\bmafia\b', r'\bcartel\b',
    r'\btrafficking\b', r'\bkidnap\b', r'\babduct\b',
    r'\bshoplift\b', r'\bvandalism\b', r'\barson\b'
]

# Meta keywords that are often safe in search suggestions but unsafe on regular sites
META_KEYWORDS = []

# Helper to compile safe patterns with boundaries
def compile_safe_pattern(keywords):
    # Escape keywords just in case, and ensure word boundaries
    # We strip existing \b to avoid double application, then re-apply
    cleaned = [k.replace(r'\b', '') for k in keywords]
    pattern = r'\b(' + '|'.join(map(re.escape, cleaned)) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

# Compile patterns for efficiency with strict boundaries
EXPLICIT_RE = compile_safe_pattern(EXPLICIT_KEYWORDS)
VIOLENCE_RE = compile_safe_pattern(VIOLENCE_KEYWORDS)
DRUG_RE = compile_safe_pattern(DRUG_KEYWORDS)
GAMBLING_RE = compile_safe_pattern(GAMBLING_KEYWORDS)
SELF_HARM_RE = compile_safe_pattern(SELF_HARM_KEYWORDS)
CRIME_RE = compile_safe_pattern(CRIME_KEYWORDS)

# Ambiguous/Double-meaning words that are safe on their own
# We use these to dampen AI sensitivity when they appear without explicit context
AMBIGUOUS_KEYWORDS = [
    'hot', 'strip', 'hard', 'wet', 'dirty', 'bang', 'beaver', 'facial'
]
AMBIGUOUS_RE = compile_safe_pattern(AMBIGUOUS_KEYWORDS)

# Trusted domains that manage their own content or are common utilities
# We are more lenient with images/text from these domains
TRUSTED_DOMAINS = []

# Age-Based Social Media Restrictions
SOCIAL_MEDIA_AGE_LIMIT = 16
SOCIAL_MEDIA_DOMAINS = [
    'facebook.com', 'instagram.com', 'tiktok.com', 'twitter.com', 'x.com',
    'snapchat.com', 'reddit.com', 'discord.com', 'pinterest.com', 
    'tumblr.com', 'twitch.tv'
]

# Age-based thresholds (Adjusted to be less aggressive)
AGE_THRESHOLDS = {
    'text': {
        'strict': {'ai': 0.35, 'keyword': 1},      # Age 0-8 (Increased AI threshold)
        'moderate': {'ai': 0.55, 'keyword': 1},    # Age 9-12
        'lenient': {'ai': 0.75, 'keyword': 3},     # Age 13+
    },
    'image': {
        'strict': {'ai': 0.75, 'votes': 1},        # Greatly increased (NSFW models are noisy)
        'moderate': {'ai': 0.85, 'votes': 1},      
        'lenient': {'ai': 0.92, 'votes': 2},       
    }
}

def get_age_category(age: int) -> str:
    if age <= 8: return 'strict'
    if age <= 12: return 'moderate'
    return 'lenient'

def is_trusted_domain(url: str) -> bool:
    if not url: return False
    url_lower = url.lower()
    return any(domain in url_lower for domain in TRUSTED_DOMAINS)

def is_search_engine(url: str) -> bool:
    if not url: return False
    url_lower = url.lower()
    search_engines = ['google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com', 'baidu.com', 'chrome://newtab']
    return any(se in url_lower for se in search_engines)

# Layer 2: Text Context Helpers
REDEEMING_KEYWORDS = {
    'anatomy', 'biology', 'medical', 'health', 'diagnosis', 'surgery', 
    'education', 'research', 'scientific', 'study', 'clinical', 'patient',
    'university', 'school', 'textbook', 'journal', 'prevention'
}

# Arousing keywords that, when combined with explicit terms, confirm NSFW intent
AROUSING_KEYWORDS = {
    'hot', 'sexy', 'cam', 'video', 'live', 'gallery', 'pics', 
    'nude', 'naked', 'xxx', 'porn', 'fuck', 'adult', '18+'
}

# Ethical / Hate Speech Keywords for "Disguised" Content Detection
# Includes terms related to racial slurs, extremist ideologies, and pseudo-scientific hate
HATE_KEYWORDS = {
  'white power', 'aryan', 'jihad', 'crusade', 'incel', 'chads', 'stacy',
  'kike', 'nigger', 'faggot', 'tranny', 'dyke', 'retard', 'subhuman',
  'pure blood', 'ethnic cleansing', 'genocide hoax', 'race realism'
}
# Regex for stricter phrase matching if needed (simplified for now to keywords)

def analyze_video_metadata(url: str, title: str, description: str, age: int) -> Tuple[bool, float, List[str]]:
    """
    Analyzes video metadata (title, description) for harmful content.
    Used for YouTube, TikTok, etc. where we can't scan the video binary easily.
    """
    combined_text = f"{title} {description}"
    
    # Run standard text analysis on the metadata
    is_safe, confidence, reasons = analyze_text_content(combined_text, age, url_context=url)
    
    # Add video-specific checks here if needed (e.g. clickbait detection)
    
    if not is_safe:
        return False, confidence, [f"Video Metadata Unsafe: {r}" for r in reasons]
        
    return True, 0.0, []

def analyze_text_content(text: str, age: int, url_context: Optional[str] = None) -> Tuple[bool, float, List[str]]:
    """
    Multi-model text analysis with ensemble voting and context awareness (Layer 2)
    Updated to include Ethical/Hate Speech filtering (Layer 2.5)
    """
    if not text or not text.strip():
        return True, 0.0, []

    age_cat = get_age_category(age)
    thresholds = AGE_THRESHOLDS['text'][age_cat]
    reasons = []
    
    # Context Factors
    on_search_engine = is_search_engine(url_context)
    domain_trust = calculate_domain_trust(url_context)
    on_trusted_site = domain_trust >= 0.8
    
    # --- ETHICAL FILTERING (Override) ---
    # Detect harmful ideologies even on "Educational" sites
    text_lower = text.lower()
    hate_matches = [w for w in HATE_KEYWORDS if w in text_lower]
    
    if hate_matches:
        # If we find hate speech, we BLOCK regardless of domain trust (mostly)
        # Exception: Wikipedia usually discusses these terms in valid context.
        # But a random blog or standard site using them is likely harmful.
        
        # Heuristic: If trusted site (Wiki) AND mild usage, let it slide.
        # If untrusted OR severe usage (multiple terms), BLOCK.
        
        if on_trusted_site and len(hate_matches) == 1:
             pass # Likely a dictionary definition or history article
        else:
             return False, 1.0, [f"Ethical Violation / Hate Speech Detected: {set(hate_matches)}"]

    # 1. Layer: Keyword Match & Co-occurrence
    explicit_matches = set(EXPLICIT_RE.findall(text))
    violence_matches = VIOLENCE_RE.findall(text)
    drug_matches = DRUG_RE.findall(text)
    gambling_matches = GAMBLING_RE.findall(text)
    self_harm_matches = SELF_HARM_RE.findall(text)
    crime_matches = CRIME_RE.findall(text)
    
    # Filter explicit matches based on context (Layer 2)
    final_explicit_matches = []
    
    if explicit_matches:
        text_lower = text.lower()
        # Check for redeeming context
        redeeming_count = sum(1 for w in REDEEMING_KEYWORDS if w in text_lower)
        arousing_count = sum(1 for w in AROUSING_KEYWORDS if w in text_lower)
        
        # Heuristic: If we have redeeming words and NO arousing words, we might allow simple explicit terms
        # If term is just 'sex', 1 redeeming word is enough (e.g. 'sex education')
        min_redeeming = 1 if all(m.lower() == 'sex' for m in explicit_matches) else 2
        is_educational_context = redeeming_count >= min_redeeming and arousing_count == 0
        
        if on_trusted_site or is_educational_context:
            # Be very lenient. Only block if we see hardcore terms (which shouldn't be in edu context)
            # For now, we filter out common "biological" explicit terms if context is good
            for match in explicit_matches:
                # If term is ambiguous or mild, and context is educational, ignore it
                if match.lower() in ['sex', 'breast', 'penis', 'vagina'] and is_educational_context:
                    continue
                final_explicit_matches.append(match)
        else:
            final_explicit_matches = list(explicit_matches)
            
    if on_search_engine:
         return True, 0.0, []

    if final_explicit_matches:
        reasons.append(f"Explicit terms detected: {set(final_explicit_matches)}")
        
    if violence_matches and not on_trusted_site:
        reasons.append(f"Violence/Gore detected: {set(violence_matches)}")
    if drug_matches and not on_trusted_site:
        reasons.append(f"Drug-related terms: {set(drug_matches)}")
    if gambling_matches and not on_trusted_site:
        reasons.append(f"Gambling-related terms: {set(gambling_matches)}")
    if crime_matches and not on_trusted_site:
        reasons.append(f"Crime-related terms: {set(crime_matches)}")
    if self_harm_matches:
        reasons.append(f"Self-harm terms: {set(self_harm_matches)}")

    # 2. Layer: AI Ensemble
    # Skip AI analysis for small snippets on trusted sites to avoid false positives
    if on_trusted_site and len(text) < 100:
        return True, 0.0, []

    # Check for ambiguous words (Double meaning protection)
    # If text matches ambiguous words (like 'hot') but NO explicit phrases (like 'hot girl'),
    # we treat it as a safe context (e.g. 'hot weather').
    ambiguous_matches = AMBIGUOUS_RE.findall(text)
    is_ambiguous_only = bool(ambiguous_matches) and not final_explicit_matches
    
    ai_votes_toxic = 0.0
    ai_votes_safe = 0.0
    ai_scores = []
    
    for text_model in nsfw_detector.text_models:
        try:
            # Use longer snippet for better context if available
            res = text_model['model'](text[:800])[0]
            label = res['label'].lower()
            score = res['score']
            
            # Weighted Voting
            is_toxic_label = label in ['toxic', 'label_1', 'offensive', 'obscene']
            
            # Dynamic threshold adjustment for double-meaning words
            detected_threshold = thresholds['ai']
            if is_ambiguous_only:
                # If text contains ambiguous words (like 'hot') but no explicit keywords,
                # we significantly raise the threshold to avoid false positives.
                detected_threshold += 0.35
            
            if is_toxic_label and score > detected_threshold:
                # Even if AI thinks it's toxic, be more lenient on trusted sites
                if on_trusted_site and score < detected_threshold + 0.2:
                    ai_votes_safe += text_model['weight']
                else:
                    ai_votes_toxic += text_model['weight']
                    reasons.append(f"AI ({text_model['name']}): Toxic ({score:.2f})")
            else:
                ai_votes_safe += text_model['weight']
            
            scores.append(score)
        except:
            continue
    
    avg_ai_confidence = sum(ai_scores) / len(ai_scores) if ai_scores else 0.0
    
    # Decision Logic
    is_safe = True
    
    # Keyword Threshold Logic
    # On Trusted Sites/Search Engines: High threshold (require user to be very explicit)
    if on_trusted_site or on_search_engine:
         keyword_thresh = 3 # Need at least 3 distinct explicit words to block purely on text
    else:
         keyword_thresh = thresholds['keyword']

    # Filter out very short matches that might be noise (e.g., 'ho' vs 'hoe')
    # Since we now use \b, this is less risky, but still good hygiene for search queries
    strong_explicit_matches = [m for m in final_explicit_matches if len(m) > 2]
    
    if len(strong_explicit_matches) >= keyword_thresh:
        is_safe = False
        reasons.append("Flagged due to repeated explicit keyword patterns")
    
    # Block on AI sentiment mismatch (only if we have significant confidence)
    if ai_votes_toxic > ai_votes_safe and ai_votes_toxic >= 1.0:
        is_safe = False
        reasons.append(f"Flagged by AI sentiment analysis (Confidence: {ai_votes_toxic})")
        
    # Block on violence/drugs/gambling for younger kids (unless on trusted site)
    if not on_trusted_site:
        if age <= 12:
            # Zero tolerance for younger kids, but require at least one strong match
            if any([len(m) > 2 for m in violence_matches + drug_matches + gambling_matches + self_harm_matches + crime_matches]):
                 # Basic check, but let's be more specific
                 if any(len(m) > 2 for m in violence_matches) or \
                    any(len(m) > 2 for m in drug_matches) or \
                    any(len(m) > 2 for m in gambling_matches) or \
                    any(len(m) > 2 for m in self_harm_matches):
                    is_safe = False
        else:
            # Older kids: Block explicit gambling/self-harm/drugs, allow mild violence context
            # Increased threshold to 2 matches to avoid accidental single-word blocks
            if len(drug_matches) > 1 or len(gambling_matches) > 1 or len(self_harm_matches) > 0:
                is_safe = False
            # Still block severe violence for older kids
            if len(violence_matches) > 2:
                is_safe = False

    return is_safe, avg_ai_confidence, reasons

# Simple in-memory cache for image URLs to avoid re-processing headers/logos
# Format: {url: (is_safe, score, reasons)}
IMAGE_CACHE = {}
MAX_CACHE_SIZE = 1000

def analyze_image_content(content: str, age: int, url_context: Optional[str] = None) -> Tuple[bool, float, List[str]]:
    """
    Image analysis with multi-model voting.
    Supports both base64 strings and image URLs.
    """
    if not nsfw_detector.image_models:
        return True, 0.0, ["Image AI not ready"]

    # Check cache for URLs
    if content.startswith(('http://', 'https://')):
        if content in IMAGE_CACHE:
            return IMAGE_CACHE[content]

    # Layer 1 Context Check
    domain_trust = calculate_domain_trust(url_context)
    on_trusted_site = domain_trust >= 0.8
    
    try:
        img_data = None
        
        # Check if content is a URL
        if content.startswith(('http://', 'https://')):
            try:
                # Add check for tiny images (likely icons/trackers)
                if not content.endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')) and 'data:' not in content:
                     # If it doesn't look like a real image URL, be lenient
                     pass
                
                response = requests.get(content, timeout=5)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type or 'application/octet-stream' in content_type:
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

        if not img_data:
            return True, 0.0, []

        try:
            # First try to open to verify it's a valid image
            img = Image.open(io.BytesIO(img_data))
            img.verify() 
            
            # Re-open for processing
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception:
            return True, 0.0, []

        # Check image dimensions - skip if too small (likely icons or trackers)
        width, height = img.size
        # Lower threshold slightly to catch small thumbnails
        if width < 30 or height < 30:
            return True, 0.0, []

        # Resize for performance - Model expects 224x224 usually
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        
        age_cat = get_age_category(age)
        thresholds = AGE_THRESHOLDS['image'][age_cat]
        
        # If on a trusted site, significantly raise the bar for blocking
        current_ai_threshold = thresholds['ai']
        if on_trusted_site:
            current_ai_threshold = max(current_ai_threshold, 0.90)
            
        nsfw_votes = 0.0
        safe_votes = 0.0
        scores = []
        reasons = []
        
        for img_model in nsfw_detector.image_models:
            results = img_model['model'](img)
            nsfw_score = 0.0
            for res in results:
                label_lower = res['label'].lower()
                if any(k in label_lower for k in ['nsfw', 'porn', 'hentai', 'sexy', 'nudity', 'explicit', 'adult']):
                    nsfw_score = max(nsfw_score, res['score'])
            
            if nsfw_score > current_ai_threshold:
                nsfw_votes += img_model['weight']
                reasons.append(f"AI ({img_model['name']}): NSFW ({nsfw_score:.2f})")
            else:
                safe_votes += img_model['weight']
            
            scores.append(nsfw_score)
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Require more than one vote for blocking on trusted sites or if leniency is high
        votes_required = thresholds['votes']
        if on_trusted_site: votes_required += 1

        is_safe = nsfw_votes < votes_required
        
        # Cache safe results to prevent re-fetching (keep cache small)
        if len(IMAGE_CACHE) > MAX_CACHE_SIZE:
             IMAGE_CACHE.clear() # Primitive cleanup
        
        if content.startswith(('http://', 'https://')):
            IMAGE_CACHE[content] = (is_safe, avg_score, reasons)
            
        return is_safe, avg_score, reasons
    except Exception as e:
        logger.error(f"Image Analysis Error: {e}")
        return True, 0.0, [str(e)]

def analyze_url(url: str, age: int, blocked_sites: List[str] = None, whitelisted_sites: List[str] = None) -> Tuple[bool, float, List[str]]:
    """
    URL analysis with domain-level blocking and age thresholds
    """
    url_lower = url.lower()
    reasons = []
    score = 0
    
    # 0. User-Defined Whitelist/Blacklist Check
    if blocked_sites is None: blocked_sites = []
    if whitelisted_sites is None: whitelisted_sites = []
    
    for allowed in whitelisted_sites:
        if allowed.lower() in url_lower:
            return True, 0.0, ["Whitelisted by parent"]
            
    for blocked in blocked_sites:
        if blocked.lower() in url_lower:
             return False, 1.0, ["Blacklisted by parent"]

    # 1. Age-Based Social Media Restriction (Policy Check)
    # We check this FIRST to short-circuit and avoid unnecessary processing
    if age is not None and age < SOCIAL_MEDIA_AGE_LIMIT and SOCIAL_MEDIA_DOMAINS:
        try:
            parsed_url = urlparse(url_lower)
            hostname = parsed_url.hostname
            
            if hostname:
                for domain in SOCIAL_MEDIA_DOMAINS:
                    # Strict matching: matches "facebook.com" or "m.facebook.com"
                    # Does NOT match "fakebook.com" or "facebook.com.scam.site" (unless logic is flawed, but endswith is decent for minimal setup)
                    # Better: check if it IS the domain or ENDS WITH .domain
                    if hostname == domain or hostname.endswith('.' + domain):
                        return False, 1.0, [f"Age-restricted Social Media: {domain}"]
        except Exception:
            # If URL parsing fails, fail open (allow) for this specific check to avoid overblocking
            pass

    # 2. Domain Trust Check (Layer 1)
    # If the domain is highly trusted (e.g. .edu, .gov, medical), we skip keyword heuristics on the URL
    # deeper text analysis will still run if the page is visited, but we won't block just because the URL contains "sex" (e.g. sex-education)
    domain_trust = calculate_domain_trust(url)
    if domain_trust >= 0.8:
         return True, 0.0, ["Trusted Domain (Safe)"]
    
    # Known adult domains (Expanded)
    adult_domains = [
        'pornhub', 'xvideos', 'xnxx', 'redtube', 'youporn', 
        'xhamster', 'hentai', 'onlyfans', 'chaturbate', 'beeg', 'tnaflix',
        'liveleak', 'goregrish', 'kaotic', 'theync' # Gore sites
    ]
    # Known gambling domains
    gambling_domains = [
        'bet365', 'pokerstars', '888casino', 'draftkings', 'fanduel',
        'roobet', 'stake.com', 'bovada', 'betway', 'williamhill',
        'slots', 'casino', 'poker', 'betting', 'sportsbook'
    ]
    
    for domain in adult_domains:
        if domain in url_lower:
            reasons.append(f"Adult website: {domain}")
            score += 100
            
    for domain in gambling_domains:
        if domain in url_lower:
            reasons.append(f"Gambling website: {domain}")
            score += 100
    
    # Keyword detection in URL (Using regex)
    target_text = url_lower
    
    # Check if this is a search engine URL to extract the query
    parsed_u = urlparse(url_lower)
    is_search = is_search_engine(url_lower)
    
    query_text = ""
    # Very basic query extraction for common engines
    if is_search:
        if 'q=' in parsed_u.query:
            query_parts = parsed_u.query.split('q=')[1].split('&')[0]
            query_text = requests.utils.unquote(query_parts).replace('+', ' ')
        elif 'p=' in parsed_u.query: # Yahoo uses p=
            query_parts = parsed_u.query.split('p=')[1].split('&')[0]
            query_text = requests.utils.unquote(query_parts).replace('+', ' ')
            
        # Short query rule: If user is typing 'lo' (for 'lottery' or 'love'), don't block yet.
        # Wait for meaningful input.
        if query_text and len(query_text) < 3:
             # Basic navigation or incomplete typing on SE
             return True, 0.0, ["Search query too short or ambiguous"]
             
        if query_text:
            target_text = query_text
    elif is_trusted_domain(url_lower):
        # Extract path for deeper check
        target_text = parsed_u.path
        
    url_keywords = EXPLICIT_RE.findall(target_text)
    if url_keywords:
        # Check against meta keywords that might appear innocently
        filtered_keywords = [k for k in url_keywords if k.lower() not in META_KEYWORDS]
        
        should_block = False
        if filtered_keywords:
             if is_search:
                 # Search engines need STRICTER evidence. 
                 # One explicit word might be a health search or news.
                 # Block only if multiple explicit words OR if the word is long/unambiguous
                 strong_matches = [k for k in filtered_keywords if len(k) >= 4]
                 if len(strong_matches) > 0 or len(filtered_keywords) > 1:
                     should_block = True
             else:
                 should_block = True
                 
        if should_block:
             reasons.append(f"Explicit keywords in URL: {set(filtered_keywords)}")
             score += 100
    
    # Violence/Crime detection in URL
    # Use target_text (query or path) instead of full url_lower to avoid matching domain names/params incorrectly
    val_keywords = VIOLENCE_RE.findall(target_text)
    crime_keywords = CRIME_RE.findall(target_text)
    
    if (val_keywords or crime_keywords) and not is_trusted_domain(url_lower):
        if val_keywords:
            reasons.append(f"Violence/Gore in URL: {set(val_keywords)}")
        if crime_keywords:
            reasons.append(f"Crime in URL: {set(crime_keywords)}")
        score += 80

    # Gambling & Self-Harm in URL
    # Use target_text (search query or path) for better accuracy
    if GAMBLING_RE.findall(target_text) and not is_trusted_domain(url_lower):
         reasons.append("Gambling in URL")
         score += 80
    if DRUG_RE.findall(target_text) and not is_trusted_domain(url_lower):
         reasons.append("Drug content in URL")
         score += 80
    if SELF_HARM_RE.findall(target_text) and not is_trusted_domain(url_lower):
         reasons.append("Self-harm content in URL")
         score += 100

    confidence = min(score / 100.0, 1.0)
    
    # Age-based threshold for URL risk
    age_cat = get_age_category(age)
    limit = 35 if age_cat == 'strict' else (55 if age_cat == 'moderate' else 75)
    
    is_safe = score < limit
    
    return is_safe, confidence, reasons

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/signup", response_model=Token)
async def signup(user_data: UserCreate):
    try:
        existing_user = await db.users.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(user_data.password)

        user_dict = {
            "email": user_data.email,
            "password": hashed_password,
            "name": user_data.name,
            "pin": None,
            "created_at": datetime.utcnow()
        }

        result = await db.users.insert_one(user_dict)
        user_id = str(result.inserted_id)

        access_token = create_access_token({"sub": user_id})

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": user_data.email,
                "name": user_data.name
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Signup failed: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")


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

# ==================== ADVANCED CONTEXT ANALYSIS (LAYER 1 & 2) ====================

# Layer 1: Domain Trust
TRUSTED_TLDS = ['.edu', '.gov', '.mil', '.ac.uk', '.edu.au', '.gov.au']
EDUCATIONAL_DOMAINS = [
    'wikipedia.org', 'webmd.com', 'mayoclinic.org', 'nih.gov', 'cdc.gov',
    'who.int', 'britannica.com', 'khanacademy.org', 'coursera.org', 'edx.org',
    'stackoverflow.com', 'github.com', 'w3schools.com', 'mdn.io', 'mozilla.org'
]

def calculate_domain_trust(url: str) -> float:
    """
    Returns a trust score between -1.0 (Bad) and 1.0 (Trusted).
    0.0 is neutral.
    """
    if not url:
        return 0.0
        
    try:
        parsed = urlparse(url.lower())
        hostname = parsed.hostname
        if not hostname:
             return 0.0
             
        # Check TLDs
        for tld in TRUSTED_TLDS:
            if hostname.endswith(tld):
                return 1.0 # Highly trusted
                
        # Check Educational Domains
        for domain in EDUCATIONAL_DOMAINS:
            if hostname == domain or hostname.endswith('.' + domain):
                return 0.8 # Trusted educational/medical
                
        return 0.0
    except:
        return 0.0


@api_router.post("/content/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(request: ContentAnalysisRequest):
    # Get profile to determine age-based thresholds
    profile = await db.profiles.find_one({"_id": ObjectId(request.profile_id)})
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    age = profile["age"]
    
    # Run heavy analysis in a separate thread to avoid blocking the async event loop
    loop = asyncio.get_event_loop()
    
    try:
        if request.content_type == "text":
            is_safe, confidence, reasons = await loop.run_in_executor(
                None, analyze_text_content, request.content, age, request.context
            )
        elif request.content_type == "url":
            is_safe, confidence, reasons = await loop.run_in_executor(
                None, analyze_url, request.content, age, profile.get("blocked_sites", []), profile.get("whitelisted_sites", [])
            )
        elif request.content_type == "video_metadata":
             # New support based on plan B1
             video_url = ""
             description = ""
             title = request.content
             
             if request.context:
                 parts = request.context.split("||", 1)
                 video_url = parts[0].strip()
                 if len(parts) > 1:
                     description = parts[1].strip()
                     
             is_safe, confidence, reasons = await loop.run_in_executor(
                None, analyze_video_metadata, video_url, title, description, age
            )     
        elif request.content_type == "image":
            is_safe, confidence, reasons = await loop.run_in_executor(
                 None, analyze_image_content, request.content, age, request.context
            )
        else:
            return ContentAnalysisResponse(is_safe=True, confidence=0.0, reasons=[], blocked=False)
            
        # Log if harmful
        if not is_safe:
            log_dict = {
                "profile_id": request.profile_id,
                "content_type": request.content_type,
                "detected_at": datetime.utcnow(),
                "is_safe": is_safe,
                "confidence": confidence,
                "reasons": reasons,
                "content_snippet": request.content[:200] if request.content_type == "text" else "[Content blocked]",
                "url": request.context or ""
            }
            await db.logs.insert_one(log_dict)

        return ContentAnalysisResponse(
            is_safe=is_safe,
            confidence=confidence,
            reasons=reasons,
            blocked=not is_safe
        )

    except Exception as e:
        print(f"Analysis Failed: {e}") 
        # Fail SAFE (Allow) so browser doesn't break.
        return ContentAnalysisResponse(is_safe=True, confidence=0.0, reasons=["Scanner Error"], blocked=False)

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

@api_router.get("/insights/{profile_id}", response_model=WellbeingResponse)
async def get_wellbeing_insights(
    profile_id: str,
    days: int = 7,
    current_user = Depends(get_current_user)
):
    # Verify profile ownership
    profile = await db.profiles.find_one({
        "_id": ObjectId(profile_id),
        "parent_id": str(current_user["_id"])
    })
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    # Calculate date range
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Fetch logs
    logs = await db.logs.find({
        "profile_id": profile_id,
        "detected_at": {"$gte": start_date}
    }).to_list(1000) # Limit reasonable analysis size
    
    total_scans = len(logs)
    
    if total_scans == 0:
        return {
            "period_days": days,
            "total_scans": 0,
            "safety_score": 100.0,
            "blocked_count": 0,
            "top_blocked_categories": [],
            "insights": [
                {
                    "title": "No Activity Detected",
                    "description": "No browsing activity recorded in this period.",
                    "sentiment": "neutral"
                }
            ]
        }
        
    # Aggregation
    safe_count = sum(1 for log in logs if log["is_safe"])
    unsafe_count = total_scans - safe_count
    safety_score = (safe_count / total_scans) * 100.0
    
    # Categorize Block Reasons
    categories = {
        "Social Media": 0,
        "Violence & Gore": 0,
        "Adult Content": 0,
        "Gambling": 0,
        "Drugs": 0,
        "Self-Harm": 0,
        "Other": 0
    }
    
    for log in logs:
        if not log["is_safe"]:
            reasons_text = " ".join(log["reasons"]).lower()
            
            matched = False
            if "social media" in reasons_text:
                categories["Social Media"] += 1
                matched = True
            if any(x in reasons_text for x in ["violence", "gore", "weapon", "kill", "blood"]):
                categories["Violence & Gore"] += 1
                matched = True
            if any(x in reasons_text for x in ["explicit", "porn", "nude", "adult", "nsfw", "sexy"]):
                categories["Adult Content"] += 1
                matched = True
            if "gambling" in reasons_text:
                categories["Gambling"] += 1
                matched = True
            if "drug" in reasons_text:
                categories["Drugs"] += 1
                matched = True
            if any(x in reasons_text for x in ["self-harm", "suicide", "depression"]):
                categories["Self-Harm"] += 1
                matched = True
                
            if not matched:
                categories["Other"] += 1
                
    # Sort categories
    sorted_cats = sorted(
        [{"category": k, "count": v} for k, v in categories.items() if v > 0],
        key=lambda x: x["count"],
        reverse=True
    )
    
    # Generate Insights
    insights = []
    
    # Insight 1: General Safety
    if safety_score >= 90:
        insights.append({
            "title": "Healthy Habits",
            "description": "Browsing activity is largely safe and constructive.",
            "sentiment": "positive"
        })
    elif safety_score >= 70:
        insights.append({
            "title": "Moderate Risks",
            "description": "Some restricted content is being accessed occasionally.",
            "sentiment": "neutral"
        })
    else:
        insights.append({
            "title": "High Risk Activity",
            "description": "Frequent attempts to access blocked content detected.",
            "sentiment": "negative"
        })
        
    # Insight 2: Top Category
    if sorted_cats:
        top_cat = sorted_cats[0]
        if top_cat["count"] > 2:
            insights.append({
                "title": f"Main Concern: {top_cat['category']}",
                "description": f"Most blocked attempts are related to {top_cat['category']}.",
                "sentiment": "neutral" if top_cat["category"] == "Social Media" else "negative"
            })
            
    # Insight 3: Safe Streak (if applicable)
    if unsafe_count == 0 and total_scans > 10:
        insights.append({
            "title": "Perfect Week",
            "description": "No harmful content was accessed in the last 7 days.",
            "sentiment": "positive"
        })

    return {
        "period_days": days,
        "total_scans": total_scans,
        "safety_score": round(safety_score, 1),
        "blocked_count": unsafe_count,
        "top_blocked_categories": sorted_cats[:3],
        "insights": insights
    }

@api_router.get("/parent/digital-wellbeing/{profile_id}", response_model=DigitalWellbeingResponse)
async def get_digital_wellbeing(
    profile_id: str,
    days: int = 7,
    current_user = Depends(get_current_user)
):
    # Verify profile ownership
    profile = await db.profiles.find_one({
        "_id": ObjectId(profile_id),
        "parent_id": str(current_user["_id"])
    })
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Fetch logs sorted chronologically (ascending) for time calculation
    logs = await db.logs.find({
        "profile_id": profile_id,
        "detected_at": {"$gte": start_date}
    }).sort("detected_at", 1).to_list(2000)
    
    # Initialize daily stats structure
    daily_map = {}
    
    # We want the last 'days' days INCLUDING today.
    # If days=7, we want [Today-6, Today-5, ..., Today].
    today_date = datetime.utcnow().date()
    
    for i in range(days):
        d = (today_date - timedelta(days=days - 1 - i)).strftime('%Y-%m-%d')
        daily_map[d] = {"screen_time": 0, "unsafe": 0}
        
    last_time = None
    
    # Process logs
    for log in logs:
        # Normalize date
        log_date = log["detected_at"].strftime('%Y-%m-%d')
        
        # If logs are slightly outside range (due to utc/local drift), skip or clamp
        # basic clamping to avoid key errors
        if log_date not in daily_map:
            # Try to map to nearest or skip
             continue

        # Count Unsafe
        if not log["is_safe"]:
            daily_map[log_date]["unsafe"] += 1
            
        # Screen Time Algorithm
        # Only time gaps < 5 minutes are considered continuous browsing session
        # If gap > 5 mins, we assume it's a new interaction and credit just 1 minute
        if last_time:
            gap = (log["detected_at"] - last_time).total_seconds()
            if gap < 300: # 5 minutes
                 daily_map[log_date]["screen_time"] += gap
            else:
                 daily_map[log_date]["screen_time"] += 60 # credit 1 minute for new activity
        else:
            daily_map[log_date]["screen_time"] += 60 # First action
            
        last_time = log["detected_at"]
        
    # Format Response
    daily_stats = []
    total_time = 0
    total_unsafe = 0
    
    # Ensure sorted order for graph
    sorted_dates = sorted(daily_map.keys())
    
    for d in sorted_dates:
        minutes = int(daily_map[d]["screen_time"] / 60)
        daily_stats.append({
            "date": d,
            "screen_time_minutes": minutes,
            "unsafe_count": daily_map[d]["unsafe"]
        })
        total_time += minutes
        total_unsafe += daily_map[d]["unsafe"]
        
    return {
        "profile_id": profile_id,
        "period_days": days,
        "total_screen_time_minutes": total_time,
        "avg_daily_minutes": int(total_time / days) if days > 0 else 0,
        "daily_stats": daily_stats,
        "unsafe_detections_total": total_unsafe
    }

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

