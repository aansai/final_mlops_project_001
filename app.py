#🔄 PredictRequest fields     ← your new features
#🔄 PredictResponse field     ← predicted_profit → predicted_churn etc
#🔄 input_dict keys           ← match your new column names
#🔄 VALID_PRICE_TIERS         ← your new categorical validators
#🔄 App title & description   ← your new project name

from __future__ import annotations

import sys
import uuid
import time
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic import ConfigDict

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from MLOPS_Clothes_project.config import ConfigLoader, get_logger


logger = get_logger(__name__)
cfg    = ConfigLoader()

MODEL_PATH     = Path(cfg.get("paths", "models")) / "model.pkl"
PROCESSOR_PATH = Path(cfg.get("paths", "processor"))


_raw_origins = cfg.get("api", "allowed_origins", default="")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["http://localhost:3000", "http://localhost:8080"]
)


VALID_PRICE_TIERS = {"budget", "mid", "upper-mid", "premium"}

MODEL_STORE: dict = {}


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency_ms)

        logger.info(
            f"method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={latency_ms} "
            f"request_id={request_id}"
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup — loading model and processor ...")
    try:
        MODEL_STORE["model"]      = joblib.load(MODEL_PATH)
        MODEL_STORE["processor"]  = joblib.load(PROCESSOR_PATH)
        MODEL_STORE["model_name"] = type(MODEL_STORE["model"]).__name__
        logger.info(
            f"Loaded model='{MODEL_STORE['model_name']}' "
            f"from '{MODEL_PATH}'"
        )
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        raise RuntimeError(f"Startup failed — artifact missing: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected startup error: {e}")
        raise RuntimeError(f"Startup failed: {e}") from e

    yield

    MODEL_STORE.clear()
    logger.info("Model store cleared — shutdown complete.")


app = FastAPI(
    title       = "MLOPS Clothes Project",
    description = "Predict Profit from Clothing Sales Data",
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

if PROMETHEUS_AVAILABLE:
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus metrics exposed at /metrics")


def require_model():
    if "model" not in MODEL_STORE or "processor" not in MODEL_STORE:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Service temporarily unavailable.",
        )


# SCHEMAS  ← only this section changes per project
class PredictRequest(BaseModel):
    # ── Categorical (cat_columns in config.yaml) ──
    Product_Category : str = Field(..., json_schema_extra={"example": "Shirts"})
    Product_Name     : str = Field(..., example="Polo Shirt")
    City             : str = Field(..., example="Mumbai")
    Segment          : str = Field(..., example="Consumer")
    price_tier       : str = Field(..., example="mid")

    # ── Numerical (num_columns in config.yaml) ────
    Units_Sold       : float = Field(..., example=10.0)
    Unit_Price       : float = Field(..., gt=0, example=499.0)
    Discount_pct     : float = Field(..., ge=0.0, le=1.0, example=0.1, alias="Discount_%")
    Sales_Amount     : float = Field(..., ge=0.0, example=4491.0)
    is_return        : int   = Field(..., ge=0, le=1, example=0)
    is_bulk_order    : int   = Field(..., ge=0, le=1, example=1)
    order_month      : int   = Field(..., ge=1, le=12, example=6)
    order_quarter    : int   = Field(..., ge=1, le=4, example=2)
    order_dayofweek  : int   = Field(..., ge=0, le=6, example=3)
    is_zero_sale     : int   = Field(..., ge=0, le=1, example=0)
    discount_applied : int   = Field(..., ge=0, le=1, example=1)

    # ── Validators ────────────────────────────────
    @field_validator("price_tier")
    @classmethod
    def validate_price_tier(cls, v: str) -> str:
        if v.lower() not in VALID_PRICE_TIERS:
            raise ValueError(
                f"price_tier must be one of {VALID_PRICE_TIERS}, got '{v}'"
            )
        return v.lower()

    @field_validator("Product_Category", "Product_Name", "City", "Segment")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("String fields must not be blank.")
        return v.strip()

    model_config = ConfigDict(populate_by_name=True)


class PredictResponse(BaseModel):
    # ── Keep these 4 fields in every project ──────
    request_id  : str
    model_name  : str
    latency_ms  : float | None = None
    status      : str
    # ── Change this field name per project ────────
    predicted_profit : float


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    model_name   : str | None = None
    model_path   : str


class ModelInfoResponse(BaseModel):
    model_name     : str
    model_path     : str
    processor_path : str
    version        : str


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"Unhandled exception request_id={request_id} error={exc}")
    return JSONResponse(
        status_code = 500,
        content     = {
            "status"     : "error",
            "detail"     : "Internal server error.",
            "request_id" : request_id,
        },
    )



@app.get("/", tags=["Root"], include_in_schema=False)
def root():
    return {
        "service" : "MLOPS Clothes Project",
        "version" : "2.0.0",
        "docs"    : "/docs",
        "health"  : "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    loaded = "model" in MODEL_STORE
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return HealthResponse(
        status       = "ok",
        model_loaded = True,
        model_name   = MODEL_STORE.get("model_name"),
        model_path   = str(MODEL_PATH),
    )


@app.get(
    "/model-info",
    response_model = ModelInfoResponse,
    tags           = ["Ops"],
    dependencies   = [Depends(require_model)],
)
def model_info():
    return ModelInfoResponse(
        model_name     = MODEL_STORE.get("model_name", "unknown"),
        model_path     = str(MODEL_PATH),
        processor_path = str(PROCESSOR_PATH),
        version        = "2.0.0",
    )


@app.post(
    "/predict",
    response_model = PredictResponse,
    tags           = ["Prediction"],
    dependencies   = [Depends(require_model)],
)
def predict(request: PredictRequest, req: Request):
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    t0 = time.perf_counter()

    # ── Column names must match config.yaml features section exactly ──────
    input_dict = {
        "Product_Category" : request.Product_Category,
        "Product_Name"     : request.Product_Name,
        "City"             : request.City,
        "Segment"          : request.Segment,
        "price_tier"       : request.price_tier,
        "Units_Sold"       : request.Units_Sold,
        "Unit_Price"       : request.Unit_Price,
        "Discount_%"       : request.Discount_pct,  
        "Sales_Amount"     : request.Sales_Amount,
        "is_return"        : request.is_return,
        "is_bulk_order"    : request.is_bulk_order,
        "order_month"      : request.order_month,
        "order_quarter"    : request.order_quarter,
        "order_dayofweek"  : request.order_dayofweek,
        "is_zero_sale"     : request.is_zero_sale,
        "discount_applied" : request.discount_applied,
    }

    try:
        df          = pd.DataFrame([input_dict])
        processor   = MODEL_STORE["processor"]   
        model       = MODEL_STORE["model"]
        transformed = processor.transform(df)   
        prediction  = float(model.predict(transformed)[0])
    except Exception as e:
        logger.exception(f"Prediction failed request_id={request_id} error={e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        f"PREDICTION request_id={request_id} "
        f"model={MODEL_STORE['model_name']} "
        f"predicted_profit={round(prediction, 2)} "
        f"latency_ms={latency_ms} "
        f"input={input_dict}"
    )

    return PredictResponse(
        request_id       = request_id,
        predicted_profit = round(prediction, 2),
        model_name       = MODEL_STORE["model_name"],
        latency_ms       = latency_ms,
        status           = "success",
    )



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = False,  
    )

    # ── Production serving command (Dockerfile) ───────────────────────────
    # gunicorn app:app \
    #   --workers 4 \
    #   --worker-class uvicorn.workers.UvicornWorker \
    #   --bind 0.0.0.0:8000 \
    #   --timeout 120 \
    #   --access-logfile - \
    #   --error-logfile  -