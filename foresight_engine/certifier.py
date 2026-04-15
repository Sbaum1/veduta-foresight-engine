# ==================================================
# FILE: foresight_engine/certifier.py
# VERSION: 3.0.0
# ROLE: FORECAST CERTIFICATION LAYER
# ENGINE: Foresight Engine v3.0.0
# ==================================================
#
# v3.0.0 FIXES:
#   - _print_report() now outputs "FORESIGHT ENGINE v3.0.0"
#   - VARIANCE_CAP replaced with CV-normalized relative threshold.
#     Raw variance of first-differences on revenue series in millions
#     will always be astronomically larger than a fixed 0.05 cap.
#     New approach: compare normalised coefficient of variation of
#     first-differences to a relative threshold. Meaningful on any scale.
#
# CERTIFICATION GATES (all four must pass):
#   GATE 1 — SHA-256 Reproducibility
#   GATE 2 — MASE vs Seasonal Naïve (primary accuracy gate)
#   GATE 3 — RMSE improvement vs Naïve baseline
#   GATE 4 — Stability diagnostics (CV-normalized variance)
# ==================================================

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .contracts import ENGINE_VERSION

# ==================================================
# CERTIFICATION THRESHOLDS
# ==================================================

MASE_ELITE_THRESHOLD    = 0.70
MASE_STRONG_THRESHOLD   = 0.85
MASE_PASS_THRESHOLD     = 1.00
RMSE_IMPROVEMENT_MIN    = 0.10

# v3.0.0: CV-normalized stability threshold
# Measures: std(diff(forecast)) / mean(|forecast|)
# A value > 0.30 means forecast first-differences have std > 30% of
# the forecast level — indicating instability, not just large values.
VARIANCE_CV_CAP         = 0.30

HASH_DECIMAL_PLACES     = 8


# ==================================================
# RESULT DATACLASSES
# ==================================================

@dataclass
class ModelCertResult:
    model_name:             str
    sha256_hash:            str
    sha256_passed:          Optional[bool]
    mase:                   Optional[float]
    mase_tier:              str
    mase_passed:            bool
    rmse_improvement_pct:   Optional[float]
    rmse_passed:            bool
    variance_metric:        Optional[float]
    stability_passed:       bool
    all_gates_passed:       bool
    notes:                  List[str] = field(default_factory=list)


@dataclass
class CertificationReport:
    report_id:              str
    engine_version:         str
    timestamp:              str
    models_certified:       int
    models_attempted:       int
    overall_passed:         bool
    certification_tier:     str
    gate_summary:           Dict[str, Any]
    model_results:          List[ModelCertResult]
    notes:                  List[str] = field(default_factory=list)


# ==================================================
# SHA-256 HASHING
# ==================================================

def _canonical_bytes(forecast_array: np.ndarray) -> bytes:
    rounded   = np.round(np.asarray(forecast_array, dtype=float), HASH_DECIMAL_PLACES)
    canonical = ",".join(f"{v:.{HASH_DECIMAL_PLACES}f}" for v in rounded.tolist())
    return canonical.encode("utf-8")


def hash_forecast(forecast_array: np.ndarray) -> str:
    return hashlib.sha256(_canonical_bytes(forecast_array)).hexdigest()


def hash_dataframe(df: pd.DataFrame, column: str = "forecast") -> str:
    sorted_df = df.sort_values("date").reset_index(drop=True)
    return hash_forecast(sorted_df[column].values)


# ==================================================
# GOLDEN HASH MANAGEMENT
# ==================================================

def generate_certificates(
    results:   Dict[str, Any],
    cert_file: str = "certified_hashes.json",
) -> Dict[str, str]:
    certificates: Dict[str, Any] = {}

    for name, result in results.items():
        if name.startswith("_"):
            continue
        if not isinstance(result, dict) or result.get("status") != "success":
            continue
        df = result.get("forecast_df")
        if df is None or df.empty:
            continue
        future_only = df[df["actual"].isna()]
        if future_only.empty:
            future_only = df
        sha256 = hash_dataframe(future_only)
        certificates[name] = {
            "sha256":         sha256,
            "rows":           len(future_only),
            "engine_version": ENGINE_VERSION,
            "generated_at":   datetime.now(timezone.utc).isoformat(),
        }

    output = {
        "engine_version": ENGINE_VERSION,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "cert_id":        str(uuid.uuid4()),
        "models":         certificates,
    }

    Path(cert_file).write_text(json.dumps(output, indent=2))
    print(f"🔐 Golden hashes saved → {cert_file}  ({len(certificates)} models)")
    return {name: v["sha256"] for name, v in certificates.items()}


def load_certificates(cert_file: str = "certified_hashes.json") -> Dict[str, str]:
    data = json.loads(Path(cert_file).read_text())
    return {name: entry["sha256"] for name, entry in data.get("models", {}).items()}


# ==================================================
# MASE SCORING
# ==================================================

def _compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray,
    seasonal_period: int = 12,
) -> Optional[float]:
    n = len(train)
    if n <= seasonal_period:
        return None
    naive_errors = np.abs(train[seasonal_period:] - train[:n - seasonal_period])
    scale        = np.mean(naive_errors)
    if scale == 0 or not np.isfinite(scale):
        return None
    return float(np.mean(np.abs(actual - forecast)) / scale)


def _mase_tier(mase: Optional[float]) -> str:
    if mase is None:
        return "Unscored"
    if mase < MASE_ELITE_THRESHOLD:
        return "Elite (7–8/10)"
    if mase < MASE_STRONG_THRESHOLD:
        return "Strong (6–7/10)"
    if mase < MASE_PASS_THRESHOLD:
        return "Pass (5–6/10)"
    return "Fail — Does Not Beat Seasonal Naïve"


# ==================================================
# RMSE SCORING
# ==================================================

def _rmse_improvement(
    actual: np.ndarray,
    forecast: np.ndarray,
    train: np.ndarray,
) -> Optional[float]:
    naive    = np.full(len(actual), train[-1])
    rmse_mdl = np.sqrt(np.mean((actual - forecast) ** 2))
    rmse_nve = np.sqrt(np.mean((actual - naive)    ** 2))
    if rmse_nve == 0 or not np.isfinite(rmse_nve):
        return None
    return float((rmse_nve - rmse_mdl) / rmse_nve * 100)


# ==================================================
# STABILITY SCORING — v3.0.0 CV-NORMALIZED
# ==================================================

def _compute_stability_cv(forecast_values: np.ndarray) -> Optional[float]:
    """
    CV-normalized stability metric.

    Computes: std(diff(forecast)) / mean(|forecast|)

    This is scale-independent — works correctly on series in dollars,
    units, millions, or any other scale. A value above VARIANCE_CV_CAP
    (0.30) indicates the forecast first-differences have std > 30% of
    the forecast level, suggesting instability.
    """
    if len(forecast_values) < 2:
        return None
    mean_abs = float(np.mean(np.abs(forecast_values)))
    if mean_abs < 1e-8:
        return None
    std_diff = float(np.std(np.diff(forecast_values)))
    return float(std_diff / mean_abs)


# ==================================================
# GATE EVALUATION
# ==================================================

def _evaluate_model(
    name:            str,
    result:          Dict[str, Any],
    golden_hashes:   Optional[Dict[str, str]],
    train_series:    np.ndarray,
    actual_series:   np.ndarray,
    seasonal_period: int,
) -> ModelCertResult:

    notes: List[str] = []

    df = result.get("forecast_df")
    if df is None or df.empty:
        return ModelCertResult(
            model_name           = name,
            sha256_hash          = "",
            sha256_passed        = False,
            mase                 = None,
            mase_tier            = "Unscored",
            mase_passed          = False,
            rmse_improvement_pct = None,
            rmse_passed          = False,
            variance_metric      = None,
            stability_passed     = False,
            all_gates_passed     = False,
            notes                = ["No forecast DataFrame available"],
        )

    future_only = df[df["actual"].isna()].copy()
    if future_only.empty:
        future_only = df.copy()

    forecast_values = future_only["forecast"].values

    # GATE 1: SHA-256
    current_hash = hash_dataframe(future_only)
    sha256_passed: Optional[bool] = None
    if golden_hashes is not None:
        expected_hash = golden_hashes.get(name)
        if expected_hash is None:
            sha256_passed = None
            notes.append("No golden hash on file — run generate_certificates() first")
        else:
            sha256_passed = current_hash == expected_hash
            if not sha256_passed:
                notes.append("SHA-256 mismatch — output is not reproducible")

    # GATE 2: MASE
    mase_value: Optional[float] = None
    mase_passed = False
    if len(actual_series) > 0 and len(forecast_values) > 0:
        min_len    = min(len(actual_series), len(forecast_values))
        mase_value = _compute_mase(
            actual          = actual_series[:min_len],
            forecast        = forecast_values[:min_len],
            train           = train_series,
            seasonal_period = seasonal_period,
        )
        if mase_value is not None:
            mase_passed = mase_value < MASE_PASS_THRESHOLD
        else:
            notes.append("MASE could not be computed — insufficient training data")
    else:
        notes.append("No actuals available for MASE computation")

    mase_tier = _mase_tier(mase_value)

    # GATE 3: RMSE improvement
    rmse_imp: Optional[float] = None
    rmse_passed = False
    if len(actual_series) > 0 and len(forecast_values) > 0:
        min_len = min(len(actual_series), len(forecast_values))
        rmse_imp = _rmse_improvement(
            actual   = actual_series[:min_len],
            forecast = forecast_values[:min_len],
            train    = train_series,
        )
        if rmse_imp is not None:
            rmse_passed = rmse_imp >= (RMSE_IMPROVEMENT_MIN * 100)
        else:
            notes.append("RMSE improvement could not be computed")

    # GATE 4: Stability (v3.0.0 CV-normalized)
    variance_metric: Optional[float] = None
    stability_passed = False

    diagnostics = result.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        variance_metric = diagnostics.get("variance_metric_future")

    if variance_metric is None:
        variance_metric = _compute_stability_cv(forecast_values)

    if variance_metric is not None:
        stability_passed = variance_metric <= VARIANCE_CV_CAP
        if not stability_passed:
            notes.append(
                f"Stability CV {variance_metric:.4f} exceeds cap {VARIANCE_CV_CAP} "
                f"(std_diff/mean_forecast > 30%)"
            )
    else:
        notes.append("Variance metric unavailable — stability unscored")
        stability_passed = True

    sha256_gate  = sha256_passed is True or sha256_passed is None
    all_gates    = sha256_gate and mase_passed and rmse_passed and stability_passed

    return ModelCertResult(
        model_name           = name,
        sha256_hash          = current_hash,
        sha256_passed        = sha256_passed,
        mase                 = round(mase_value, 4) if mase_value is not None else None,
        mase_tier            = mase_tier,
        mase_passed          = mase_passed,
        rmse_improvement_pct = round(rmse_imp, 2) if rmse_imp is not None else None,
        rmse_passed          = rmse_passed,
        variance_metric      = round(variance_metric, 6) if variance_metric is not None else None,
        stability_passed     = stability_passed,
        all_gates_passed     = all_gates,
        notes                = notes,
    )


# ==================================================
# CERTIFICATION TIER ASSIGNMENT
# ==================================================

def _assign_certification_tier(
    model_results: List[ModelCertResult],
    n_attempted:   int,
) -> str:
    n_passed = sum(1 for r in model_results if r.all_gates_passed)
    n_elite  = sum(1 for r in model_results if r.mase is not None
                   and r.mase < MASE_ELITE_THRESHOLD)

    if n_passed == 0:
        return "Uncertified"
    if n_elite >= max(1, int(n_attempted * 0.6)) and n_passed == n_attempted:
        return "MASE < 0.70 — Beats seasonal naive by 30%+ (M-Competition standard)"
    if n_passed == n_attempted:
        return "6–7 / 10  — Strong (All models beat seasonal naïve)"
    if n_passed >= int(n_attempted * 0.5):
        return "5–6 / 10  — Moderate (Majority beat seasonal naïve)"
    return "< 5 / 10  — Does Not Consistently Beat Seasonal Naïve"


# ==================================================
# MAIN CERTIFICATION ENTRY POINT
# ==================================================

def certify(
    results:          Dict[str, Any],
    historical_df:    pd.DataFrame,
    cert_file:        Optional[str] = "certified_hashes.json",
    seasonal_period:  int           = 12,
    generate_hashes:  bool          = False,
) -> CertificationReport:

    report_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    hist = historical_df.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)
    train_series = hist["value"].values

    golden_hashes: Optional[Dict[str, str]] = None
    if cert_file is not None:
        if generate_hashes:
            golden_hashes = generate_certificates(results, cert_file)
        elif Path(cert_file).exists():
            golden_hashes = load_certificates(cert_file)
        else:
            print(f"⚠️  No cert file found at '{cert_file}'. Run with generate_hashes=True to create it.")

    production_results = {
        name: result
        for name, result in results.items()
        if not name.startswith("_")
        and isinstance(result, dict)
        and result.get("status") == "success"
        and not result.get("diagnostic_only", False)
    }

    n_attempted = len(production_results)

    if len(train_series) > seasonal_period:
        actual_series  = train_series[-seasonal_period:]
        train_for_mase = train_series[:-seasonal_period]
    else:
        actual_series  = train_series
        train_for_mase = train_series

    model_results: List[ModelCertResult] = []
    for name, result in production_results.items():
        cert_result = _evaluate_model(
            name            = name,
            result          = result,
            golden_hashes   = golden_hashes,
            train_series    = train_for_mase,
            actual_series   = actual_series,
            seasonal_period = seasonal_period,
        )
        model_results.append(cert_result)

    n_sha256_passed    = sum(1 for r in model_results if r.sha256_passed is True)
    n_mase_passed      = sum(1 for r in model_results if r.mase_passed)
    n_rmse_passed      = sum(1 for r in model_results if r.rmse_passed)
    n_stability_passed = sum(1 for r in model_results if r.stability_passed)
    n_all_passed       = sum(1 for r in model_results if r.all_gates_passed)
    n_elite            = sum(1 for r in model_results
                             if r.mase is not None and r.mase < MASE_ELITE_THRESHOLD)

    overall_passed = n_all_passed == n_attempted and n_attempted > 0
    cert_tier      = _assign_certification_tier(model_results, n_attempted)

    gate_summary = {
        "gate_1_sha256":    {"passed": n_sha256_passed,    "of": n_attempted},
        "gate_2_mase":      {"passed": n_mase_passed,      "of": n_attempted},
        "gate_3_rmse":      {"passed": n_rmse_passed,      "of": n_attempted},
        "gate_4_stability": {"passed": n_stability_passed, "of": n_attempted},
        "elite_models":     n_elite,
        "all_gates_passed": n_all_passed,
    }

    _print_report(
        cert_tier     = cert_tier,
        overall       = overall_passed,
        gate_summary  = gate_summary,
        model_results = model_results,
        n_attempted   = n_attempted,
    )

    return CertificationReport(
        report_id          = report_id,
        engine_version     = ENGINE_VERSION,
        timestamp          = timestamp,
        models_certified   = n_all_passed,
        models_attempted   = n_attempted,
        overall_passed     = overall_passed,
        certification_tier = cert_tier,
        gate_summary       = gate_summary,
        model_results      = model_results,
    )


# ==================================================
# REPORT PRINTER
# ==================================================

def _print_report(
    cert_tier:     str,
    overall:       bool,
    gate_summary:  Dict[str, Any],
    model_results: List[ModelCertResult],
    n_attempted:   int,
) -> None:

    w = 72
    print("\n" + "=" * w)
    print(f"  FORESIGHT ENGINE v{ENGINE_VERSION} — CERTIFICATION REPORT")
    print("=" * w)

    for result in sorted(model_results, key=lambda r: r.model_name):
        sha  = "✅" if result.sha256_passed is True  else ("➖" if result.sha256_passed is None else "❌")
        mase = "✅" if result.mase_passed             else "❌"
        rmse = "✅" if result.rmse_passed             else "❌"
        stab = "✅" if result.stability_passed        else "❌"
        all_ = "✅ CERTIFIED" if result.all_gates_passed else "❌ NOT CERTIFIED"

        mase_str = f"{result.mase:.4f}" if result.mase is not None else "N/A"
        rmse_str = f"{result.rmse_improvement_pct:.1f}%" if result.rmse_improvement_pct is not None else "N/A"

        print(f"\n  {result.model_name}")
        print(f"    SHA-256 {sha}  |  MASE {mase} {mase_str}  |  "
              f"RMSE↑ {rmse} {rmse_str}  |  Stability {stab}  →  {all_}")
        print(f"    Tier: {result.mase_tier}")
        for note in result.notes:
            print(f"    ⚠️  {note}")

    print("\n" + "─" * w)
    gs = gate_summary
    print(f"  Gate 1 SHA-256    : {gs['gate_1_sha256']['passed']}/{n_attempted}")
    print(f"  Gate 2 MASE       : {gs['gate_2_mase']['passed']}/{n_attempted}")
    print(f"  Gate 3 RMSE       : {gs['gate_3_rmse']['passed']}/{n_attempted}")
    print(f"  Gate 4 Stability  : {gs['gate_4_stability']['passed']}/{n_attempted}")
    print(f"  Elite models      : {gs['elite_models']}")
    print(f"\n  CERTIFICATION TIER: {cert_tier}")
    print(f"  OVERALL           : {'✅ CERTIFIED' if overall else '❌ NOT CERTIFIED'}")
    print("=" * w + "\n")


# ==================================================
# REPORT SERIALIZATION
# ==================================================

def save_report(report: CertificationReport, filepath: str = "certification_report.json") -> None:
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    data = asdict(report)
    Path(filepath).write_text(json.dumps(data, indent=2, default=_serialize))
    print(f"📄 Certification report saved → {filepath}")


def verify_certificates(results: Dict[str, Any], cert_file: str = "certified_hashes.json") -> bool:
    if not Path(cert_file).exists():
        print(f"❌ Cert file not found: {cert_file}")
        return False

    golden    = load_certificates(cert_file)
    all_passed = True

    print("\n── SHA-256 Verification ─────────────────────────────────────")

    for name, result in results.items():
        if name.startswith("_") or not isinstance(result, dict):
            continue
        if result.get("status") != "success":
            continue
        df = result.get("forecast_df")
        if df is None or df.empty:
            continue
        future_only = df[df["actual"].isna()]
        if future_only.empty:
            future_only = df
        current  = hash_dataframe(future_only)
        expected = golden.get(name)
        if expected is None:
            print(f"  ➖ {name:30s} — no golden hash on file")
            continue
        passed = current == expected
        icon   = "✅" if passed else "❌"
        print(f"  {icon} {name:30s}")
        if not passed:
            all_passed = False
            print(f"     Expected: {expected}")
            print(f"     Got:      {current}")

    print(f"\n  Result: {'✅ ALL PASS' if all_passed else '❌ FAILURES DETECTED'}\n")
    return all_passed
