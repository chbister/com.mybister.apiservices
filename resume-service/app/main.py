import os
import socket
import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .parser.resume import ResumeExtractor, NER_MODEL_ID, NER_ENABLED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- GlitchTip (Sentry-kompat): Errors + Traces -----------------------------
# Nur aktiv, wenn SENTRY_DSN gesetzt ist (siehe .env.example / docker-compose).
# Ohne DSN bzw. ohne installiertes sentry-sdk verhält sich der Service wie bisher.
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

if sentry_sdk is not None and (_sentry_dsn := os.getenv("SENTRY_DSN")):
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "dev"),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        integrations=[FastApiIntegration()],
        send_default_pii=False,
    )
    sentry_sdk.set_tag("service", "resume-service")
    if os.getenv("SENTRY_SEND_TEST_EVENT"):
        sentry_sdk.capture_message("resume-service: GlitchTip test event")

app = FastAPI(title="Resume Service")

extractor = ResumeExtractor()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Resume Service...")
    extractor.preload()
    logger.info("Resume extractor ready (ner_enabled=%s).", extractor.ner_enabled)


class ResumeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    lines: List[str] = []
    locale: str = "de"


class ResumeProfile(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None


class ResumeExperience(BaseModel):
    organization: str
    role: str
    employment_type: Optional[str] = None
    location: Optional[str] = None
    started_on: Optional[str] = None
    ended_on: Optional[str] = None
    description: Optional[str] = None


class ResumeEducation(BaseModel):
    institution: str
    qualification: str
    field: Optional[str] = None
    started_on: Optional[str] = None
    ended_on: Optional[str] = None


class ResumeCertification(BaseModel):
    name: str
    issuer: Optional[str] = None
    issued_on: Optional[str] = None
    expires_on: Optional[str] = None


class ResumeProject(BaseModel):
    name: str
    role: Optional[str] = None
    client_name: Optional[str] = None
    started_on: Optional[str] = None
    ended_on: Optional[str] = None
    description: Optional[str] = None


class ResumeMetadata(BaseModel):
    processing_time: float
    model: str
    locale: str


class ResumeResponse(BaseModel):
    profile: ResumeProfile
    experiences: List[ResumeExperience] = []
    educations: List[ResumeEducation] = []
    certifications: List[ResumeCertification] = []
    projects: List[ResumeProject] = []
    warnings: List[str] = []
    count: int
    hostname: str
    metadata: ResumeMetadata


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "resume",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat(),
        "config": {
            "supported_locales": ["de", "en"],
            "ner_enabled": extractor.ner_enabled,
            "ner_model": NER_MODEL_ID,
        },
    }


@app.post("/v1/resume/extract", response_model=ResumeResponse)
async def extract(req: ResumeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    start_time = time.time()
    try:
        result = extractor.extract(req.text, req.lines, req.locale)
    except Exception as exc:
        logger.error("Resume extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Resume extraction failed.")

    processing_time = time.time() - start_time
    count = (
        len(result.experiences)
        + len(result.educations)
        + len(result.certifications)
        + len(result.projects)
    )

    return ResumeResponse(
        profile=ResumeProfile(**result.profile),
        experiences=[ResumeExperience(**e) for e in result.experiences],
        educations=[ResumeEducation(**e) for e in result.educations],
        certifications=[ResumeCertification(**c) for c in result.certifications],
        projects=[ResumeProject(**p) for p in result.projects],
        warnings=result.warnings,
        count=count,
        hostname=socket.gethostname(),
        metadata=ResumeMetadata(
            processing_time=round(processing_time, 4),
            model=NER_MODEL_ID if extractor.ner_enabled else "rule-based",
            locale=req.locale,
        ),
    )
