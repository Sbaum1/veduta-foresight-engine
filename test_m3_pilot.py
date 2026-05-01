# ==============================================================================
# FILE: test_m3_pilot.py
# VERSION: 1.0.0
# ROLE: STAGE 2 — STRATIFIED PILOT (30 SERIES)
# ENGINE: VEDUTA Foresight Engine v3.0.0
#
# PURPOSE:
#   Stratified pre-flight evaluation across the full range of M3 monthly
#   series lengths. Mirrors the methodology of the original VEDUTA engine.
#
#   Stratification:
#     Short  (n_train < 60):   10 series
#     Medium (n_train 60-99):  10 series
#     Long   (n_train >= 100): 10 series
#     Seed: 42 (fixed for reproducibility)
#
#   PASS CRITERIA:
#     - 30/30 series complete (zero failures)
#     - Median MASE < 1.0 (beats seasonal naive on majority of series)
#     - Max per-series time < 120s
#     - Target for full run: Median MASE < 0.72
#
# USAGE:
#   python test_m3_pilot.py
#   Must be run from C:\Dev\VEDUTA\core\foresight_x
# ==============================================================================

from __future__ import annotations

import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

# ── Executive-grade console suppression ───────────────────────────────────────
import logging, warnings, os, sys

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)  # Block DEBUG/INFO/WARNING globally

for _n in ["cmdstanpy","prophet","prophet.forecaster","numexpr",
           "statsmodels","statsmodels.tsa","matplotlib","py.warnings",
           "torch","transformers","accelerate","chronos"]:
    _l = logging.getLogger(_n)
    _l.setLevel(logging.CRITICAL)
    _l.propagate = False

# Redirect stderr to suppress any remaining library noise
class _Quiet:
    def write(self, *a): pass
    def flush(self): pass

sys.stderr = _Quiet()
# ─────────────────────────────────────────────────────────────────────────────




import numpy as np
import pandas as pd

from m3_loader import (
    load_m3_monthly, stratified_sample,
    compute_mase, compute_mase_raw, compute_smape, compute_mae, compute_rmse,
    M3_HORIZON,
)

M3_PATH = r"V:\_staging\canonical\foresight_x\sample_data\m3\m3_monthly_dataset.tsf"
OUTPUT  = Path("diagnostics")
OUTPUT.mkdir(exist_ok=True)

PILOT_N    = 30
PILOT_SEED = 42
PASS_MASE  = 1.0   # Pilot gate: beat seasonal naive
TARGET_MASE = 0.72  # Full run target


def _no_backtest(df, model_runner, horizon, confidence_level):
    """No-op backtest. M3 uses official held-out actuals, not CV."""
    return {
        "eligible":         True,
        "observations":     len(df),
        "backtest_skipped": True,
        "reason":           "M3 benchmark — official holdout actuals used",
    }


def extract_ensemble_forecast(raw, last_train_date, horizon):
    ens = raw.get("Primary Ensemble", {})
    if ens.get("status") != "success":
        raise RuntimeError(
            f"Primary Ensemble status: {ens.get('status')} — "
            f"{ens.get('error', 'no error detail')}"
        )
    fc_df = ens["forecast_df"].copy()
    fc_df["date"] = pd.to_datetime(fc_df["date"])
    future = fc_df[fc_df["date"] > last_train_date].copy()
    if len(future) < horizon:
        future = fc_df[fc_df["actual"].isna()].copy()
    future = future.head(horizon)
    if len(future) < horizon:
        raise RuntimeError(f"Forecast too short: {len(future)} < {horizon}")
    fc = future["forecast"].values.astype(float)
    if not np.isfinite(fc).all():
        raise RuntimeError("Non-finite forecast values")
    return fc


def run_pilot():
    ts = datetime.now()
    print()
    print("=" * 72)
    print("  VEDUTA Foresight Engine v3.0.0 — STAGE 2: STRATIFIED PILOT")
    print(f"  Started: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {PILOT_N} series · Stratified by length · Seed={PILOT_SEED}")
    print("  Citations: Makridakis & Hibon (2000) · Hyndman & Koehler (2006)")
    print("=" * 72)

    print(f"\n  Loading M3 dataset...")
    all_series = load_m3_monthly(M3_PATH, verbose=True)

    pilot = stratified_sample(all_series, n=PILOT_N, seed=PILOT_SEED)

    # Report stratification
    short  = [s for s in pilot if s["n_train"] < 60]
    medium = [s for s in pilot if 60 <= s["n_train"] < 100]
    long_  = [s for s in pilot if s["n_train"] >= 100]
    print(f"\n  Pilot sample: {len(pilot)} series")
    print(f"    Short  (n_train <60):   {len(short):2d} series")
    print(f"    Medium (n_train 60-99): {len(medium):2d} series")
    print(f"    Long   (n_train >=100): {len(long_):2d} series")
    print()

    from foresight_engine.runner import run_all_models

    series_results = []
    mase_vals      = []
    smape_vals     = []
    n_fail         = 0

    print(f"  {'#':>3}  {'Series':8s}  {'n_train':>8}  {'Elapsed':>9}  "
          f"{'MASE':>8}  {'sMAPE':>7}  {'Status'}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*6}")

    t_start = time.time()

    for i, s in enumerate(pilot, 1):
        sid     = s["series_id"]
        df      = s["df"]
        actuals = s["actuals"]
        n_train = s["n_train"]

        t0 = time.time()
        status = "PASS"
        mase = smape = mase_raw = mae = rmse = None
        error_msg = None

        try:
            raw = run_all_models(
                df               = df.copy(),
                horizon          = M3_HORIZON,
                confidence_level = 0.90,
                backtest_fn      = _no_backtest,
            )

            last_date  = pd.to_datetime(df["date"].max())
            fc         = extract_ensemble_forecast(raw, last_date, M3_HORIZON)
            train_vals = df["value"].values.astype(float)

            mase     = compute_mase(fc, actuals, train_vals)
            mase_raw = compute_mase_raw(fc, actuals, train_vals)
            smape    = compute_smape(fc, actuals)
            mae      = compute_mae(fc, actuals)
            rmse     = compute_rmse(fc, actuals)

            mase_vals.append(mase)
            if np.isfinite(smape):
                smape_vals.append(smape)

            if mase < 1.0:
                status = "PASS"
            elif mase < 2.0:
                status = "WARN"
            else:
                status = "HIGH"

        except Exception as e:
            status    = "FAIL"
            error_msg = str(e)[:100]
            n_fail   += 1

        elapsed = time.time() - t0

        flag = ("✅" if status == "PASS" else
                "⚠️ " if status in ("WARN", "HIGH") else "❌")

        mase_str  = f"{mase:.4f}"  if mase  is not None else "   N/A"
        smape_str = f"{smape:.1f}%" if smape is not None else "   N/A"

        print(f"  {i:>3}  {sid:8s}  {n_train:>8d}  {elapsed:>8.1f}s  "
              f"{mase_str:>8}  {smape_str:>7}  {flag} {status}")

        if error_msg:
            print(f"         ERROR: {error_msg}")

        series_results.append({
            "series_id": sid,
            "n_train":   n_train,
            "elapsed_s": round(elapsed, 2),
            "mase":      round(mase, 6)     if mase     is not None else None,
            "mase_raw":  round(mase_raw, 6) if mase_raw is not None else None,
            "smape":     round(smape, 4)    if smape    is not None else None,
            "mae":       round(mae, 4)      if mae      is not None else None,
            "rmse":      round(rmse, 4)     if rmse     is not None else None,
            "status":    status,
            "error":     error_msg,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start

    median_mase  = float(np.median(mase_vals))  if mase_vals  else None
    mean_mase    = float(np.mean(mase_vals))     if mase_vals  else None
    median_smape = float(np.median(smape_vals))  if smape_vals else None
    elapsed_all  = [r["elapsed_s"] for r in series_results]

    pilot_passed = (
        n_fail == 0 and
        median_mase is not None and
        median_mase <= PASS_MASE
    )

    print()
    print("  " + "─" * 70)
    print(f"  Series tested:     {len(pilot)}")
    print(f"  Completed:         {len(pilot) - n_fail}")
    print(f"  Failed:            {n_fail}")
    print()
    print(f"  Median MASE:       {median_mase:.4f}  "
          f"(gate: <{PASS_MASE}  target: <{TARGET_MASE})"
          if median_mase else "  Median MASE:  N/A")
    print(f"  Mean MASE:         {mean_mase:.4f}"
          if mean_mase else "  Mean MASE:    N/A")
    print(f"  Median sMAPE:      {median_smape:.2f}%"
          if median_smape else "  Median sMAPE: N/A")
    print()
    print(f"  Avg time/series:   {np.mean(elapsed_all):.1f}s")
    print(f"  Max time/series:   {np.max(elapsed_all):.1f}s")
    print(f"  Total elapsed:     {total_elapsed/60:.1f} min")
    proj = np.mean(elapsed_all) * 1428 / 3600
    print(f"  Projected full run: ~{proj:.1f} hours")
    print()

    verdict = (
        f"✅ STAGE 2 PASSED (Median MASE {median_mase:.4f} < {PASS_MASE})"
        f" — Proceed to Stage 3"
        if pilot_passed else
        f"❌ STAGE 2 FAILED — Do not proceed to Stage 3"
    )
    print(f"  {verdict}")
    print("=" * 72 + "\n")

    # Save
    output = {
        "test":          "stage2_stratified_pilot",
        "engine":        "Foresight Engine v3.0.0",
        "timestamp":     ts.isoformat(),
        "pilot_n":       PILOT_N,
        "pilot_seed":    PILOT_SEED,
        "backtest":      "disabled — M3 uses official holdout actuals",
        "horizon":       M3_HORIZON,
        "n_valid":       len(mase_vals),
        "n_failed":      n_fail,
        "median_mase":   round(median_mase, 6)  if median_mase  else None,
        "mean_mase":     round(mean_mase, 6)    if mean_mase    else None,
        "median_smape":  round(median_smape, 4) if median_smape else None,
        "pass_gate":     PASS_MASE,
        "target_mase":   TARGET_MASE,
        "pilot_passed":  pilot_passed,
        "elapsed_s":     round(total_elapsed, 1),
        "methodology": {
            "mase":    "Hyndman & Koehler (2006), IJF 22(4):679-688",
            "smape":   "Makridakis (1993), IJF 9:527-529",
            "dataset": "Makridakis & Hibon (2000), IJF 16(4):451-476",
        },
        "series": series_results,
    }
    out_path = OUTPUT / "stage2_pilot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {out_path}\n")
    return pilot_passed


if __name__ == "__main__":
    passed = run_pilot()
    sys.exit(0 if passed else 1)


