"""FastAPI app for CATE prediction + ATE comparison + ops metrics.

Routes (v1):
- POST /api/v1/cate            — single CATE prediction + 95% CI + NL recommendation
- GET  /api/v1/ate_comparison  — OLS-CPS-bias vs DML vs RCT-truth table (the headline)
- GET  /api/v1/health          — uptime + memory + counters
- GET  /api/v1/model_info      — full reports/results.json
- GET  /metrics                — Prometheus exposition (incl. CATE drift instruments)

Hardening: slowapi rate limiting, CORS, TrustedHost, PredictionError -> 422
mapping, prometheus_fastapi_instrumentator for default HTTP metrics.
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.cate_monitor import CATEDriftMonitor
from src.api.nl_translator import translate_cate
from src.config import load_config
from src.data.schemas import ATEReport, CATEPrediction, LalondeProfile
from src.data.skew_check import check_skew
from src.exceptions import PredictionError
from src.logger import get_logger
from src.models.causal_forest_model import CausalForestEstimator

logger = get_logger(__name__)

START_TIME = time.time()
PRED_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model + drift monitor at startup; release on shutdown."""
    cfg = load_config("config/config.yaml")
    app.state.cfg = cfg
    app.state.cf = CausalForestEstimator.load(
        Path(cfg["paths"]["models_dir"]) / "causal_forest"
    )
    baseline_cates = pd.read_parquet(
        Path(cfg["paths"]["reports_dir"]) / "cate_per_row.parquet"
    )["cate"].to_numpy()
    app.state.monitor = CATEDriftMonitor(
        window_size=cfg["monitoring"]["cate_drift_window_size"],
        psi_threshold=cfg["monitoring"]["cate_psi_threshold"],
        baseline_cates=baseline_cates,
    )
    logger.info(
        "API ready: causal_forest loaded, drift monitor warm with %d baseline CATEs",
        len(baseline_cates),
    )
    yield
    logger.info("API shutdown")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Causal ML . HTE", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
Instrumentator().instrument(app).expose(app)


@app.exception_handler(PredictionError)
async def _prediction_error_handler(_: Request, exc: PredictionError) -> HTTPException:
    """Map PredictionError -> HTTP 422 so input-validation failures are explicit."""
    raise HTTPException(status_code=422, detail=str(exc))


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_: Request, exc: RateLimitExceeded) -> HTTPException:
    """Map slowapi rate-limit overflow -> HTTP 429."""
    raise HTTPException(status_code=429, detail=str(exc))


@app.post("/api/v1/cate", response_model=CATEPrediction)
@limiter.limit("30/minute")
async def predict_cate(request: Request, profile: LalondeProfile) -> CATEPrediction:
    """Predict CATE + 95% bootstrap CI + TREAT/DEFER recommendation."""
    global PRED_COUNT
    cf: CausalForestEstimator = request.app.state.cf
    cfg = request.app.state.cfg

    X = np.array(
        [
            [
                profile.age,
                profile.education,
                profile.black,
                profile.hispanic,
                profile.married,
                profile.nodegree,
                profile.re74,
                profile.re75,
            ]
        ],
        dtype=float,
    )

    skew = check_skew(X[0], Path(cfg["paths"]["models_dir"]) / "training_stats.json")
    if any(skew.values()):
        logger.warning(
            "Input outside training distribution: %s",
            [k for k, v in skew.items() if v],
        )

    cate_vec = cf.safe_predict(X)
    if cf.estimator is None:  # pragma: no cover - guarded by safe_predict
        raise PredictionError("CausalForestEstimator not fitted.")
    lo, hi = cf.estimator.effect_interval(X, alpha=0.05)
    cate = float(cate_vec[0])
    ci_lower = float(np.asarray(lo).reshape(-1)[0])
    ci_upper = float(np.asarray(hi).reshape(-1)[0])
    rec, nl = translate_cate(cate, ci_lower, ci_upper)

    request.app.state.monitor.record(cate)
    PRED_COUNT += 1

    rec_literal: Literal["TREAT", "DEFER"] = "TREAT" if rec == "TREAT" else "DEFER"
    return CATEPrediction(
        cate=cate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        recommendation=rec_literal,
        nl_summary=nl,
    )


@app.get("/api/v1/ate_comparison", response_model=ATEReport)
async def ate_comparison(request: Request) -> ATEReport:
    """Headline OLS-vs-DML-vs-RCT-truth table (rule C40 narrative)."""
    results = json.loads(
        Path(request.app.state.cfg["paths"]["results_json"]).read_text()
    )
    t = results["ate_table"]
    return ATEReport(
        ols_cps_ate=t["ols_cps_unadjusted"]["ate"],
        ols_cps_ci=(
            t["ols_cps_unadjusted"]["ci_lower"],
            t["ols_cps_unadjusted"]["ci_upper"],
        ),
        dml_cps_ate=t["dml_cps"]["ate"],
        dml_cps_ci=(t["dml_cps"]["ci_lower"], t["dml_cps"]["ci_upper"]),
        dml_nsw_ate=t["dml_nsw"]["ate"],
        dml_nsw_ci=(t["dml_nsw"]["ci_lower"], t["dml_nsw"]["ci_upper"]),
        rcl_ground_truth_ate=t["rcl_ground_truth"]["ate"],
        rcl_ground_truth_ci=(
            t["rcl_ground_truth"]["ci_lower"],
            t["rcl_ground_truth"]["ci_upper"],
        ),
    )


@app.get("/api/v1/health")
async def health(request: Request) -> dict[str, str | int | bool]:
    """Detailed health: uptime, memory, version, prediction counter."""
    return {
        "status": "ok",
        "model_loaded": request.app.state.cf is not None,
        "uptime_seconds": int(time.time() - START_TIME),
        "memory_mb": int(psutil.Process().memory_info().rss / 1024 / 1024),
        "version": app.version,
        "n_cate_predictions_served": PRED_COUNT,
    }


@app.get("/api/v1/model_info")
async def model_info(request: Request) -> dict[str, object]:
    """Full reports/results.json (ATE table, heterogeneity, policy, SHAP)."""
    p = Path(request.app.state.cfg["paths"]["results_json"])
    if not p.exists():
        return {"error": "results.json missing"}
    return json.loads(p.read_text())
