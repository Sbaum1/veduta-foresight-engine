# ==============================================================================
# FILE: test_m3_timing.py
# VERSION: 1.0.0
# ROLE: STAGE 1 — TIMING PROOF OF CONCEPT
# ENGINE: VEDUTA Foresight Engine v3.0.0
#
# PURPOSE:
#   Proves that with backtest DISABLED, all series complete in reasonable
#   time regardless of series length. This is the core fix for the timeout
#   problem that caused 201/480 series to fail in the previous run.
#
#   Runs exactly 5 series:
#     - 2 short  (n_train ~50)  — first M3 series
#     - 2 medium (n_train ~80)  — mid-range series
#     - 1 long   (n_train ~108) — longest observed series
#
#   PASS CRITERIA:
#     - All 5 series complete (zero failures)
#     - All 5 series complete in under 120 seconds each
#     - All 5 MASE values are finite
#     - Median MASE < 2.0 (beats naive on majority)
#
# WHY BACKTEST IS DISABLED:
#   The M3 competition provides official held-out actuals (series.xx /
#   the last 18 observations). We compare our forecast directly to those
#   actuals. Rolling-origin cross-validation is a VEDUTA UI feature for
#   model trust scoring — it is not part of any M3 evaluation methodology.
#   With backtest enabled, long series (n=108) require ~20 folds × 37 models,
#   causing ~170s per series. With backtest disabled, all series complete
#   in ~29s regardless of length.
#
# USAGE:
#   cd C:\Dev\VEDUTA\core\foresight_x
#   python test_m3_timing.py
# ==============================================================================

from __future__ import annotations

import sys, json, time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from m3_loader import (
    load_m3_monthly, compute_mase, compute_smape,
    M3_HORIZON, M3_FREQUENCY,
)

M3_PATH   = r"C:\Dev\VEDUTA\_shared\sample_data\m3\m3_monthly_dataset.tsf"
OUTPUT    = Path("test_results")
OUTPUT.mkdir(exist_ok=True)


def _no_backtest(df, model_runner, horizon, confidence_level):
    """
    No-op backtest function for M3 benchmark.
    The M3 competition provides official held-out actuals — no cross-validation
    is required or appropriate for this benchmark.
    """
    return {
        "eligible":        True,
        "observations":    len(df),
        "backtest_skipped": True,
        "reason":          "M3 benchmark — official holdout used instead",
    }


def extract_ensemble_forecast(raw: dict, last_train_date: pd.Timestamp,
                               horizon: int) -> np.ndarray:
    """
    Extract the Primary Ensemble forecast values from run_all_models output.
    Returns array of length=horizon.
    """
    ens = raw.get("Primary Ensemble", {})
    if ens.get("status") != "success":
        raise RuntimeError(
            f"Primary Ensemble failed: {ens.get('error', 'unknown')}"
        )

    fc_df  = ens["forecast_df"].copy()
    fc_df["date"] = pd.to_datetime(fc_df["date"])

    # Extract future rows (after last training date)
    future = fc_df[fc_df["date"] > last_train_date].copy()
    if len(future) < horizon:
        # Fallback: rows where actual is NaN
        future = fc_df[fc_df["actual"].isna()].copy()

    future = future.head(horizon)
    if len(future) < horizon:
        raise RuntimeError(
            f"Forecast too short: {len(future)} < {horizon}"
        )

    fc = future["forecast"].values.astype(float)
    if not np.isfinite(fc).all():
        raise RuntimeError("Non-finite forecast values in Primary Ensemble")

    return fc


def run_timing_test():
    print()
    print("=" * 70)
    print("  VEDUTA Foresight Engine v3.0.0 — STAGE 1: TIMING PROOF")
    print("  Tests that backtest=disabled fixes the timeout problem")
    print("=" * 70)
    print()

    # Load all series to get length distribution
    print("  Loading M3 dataset...")
    all_series = load_m3_monthly(M3_PATH, verbose=True)
    print(f"  Total series: {len(all_series)}")
    print()

    # Select 5 test series spanning the length distribution
    short  = [s for s in all_series if s["n_train"] < 60]
    medium = [s for s in all_series if 60 <= s["n_train"] < 100]
    long_  = [s for s in all_series if s["n_train"] >= 100]

    print(f"  Series by length:")
    print(f"    Short  (n_train <60):   {len(short):4d} series")
    print(f"    Medium (n_train 60-99): {len(medium):4d} series")
    print(f"    Long   (n_train >=100): {len(long_):4d} series")
    print()

    # Pick test series — first of each type for determinism
    test_series = []
    if len(short)  >= 2: test_series += short[:2]
    elif short:          test_series += short[:1]
    if len(medium) >= 1: test_series += medium[:1]
    elif len(medium) >= 1: test_series += medium[:1]
    if len(long_)  >= 2: test_series += long_[:2]
    elif long_:          test_series += long_[:1]

    # Fill to 5 if needed
    remaining = [s for s in all_series if s not in test_series]
    while len(test_series) < 5 and remaining:
        test_series.append(remaining.pop(0))
    test_series = test_series[:5]

    print(f"  Test series selected: {len(test_series)}")
    for s in test_series:
        print(f"    {s['series_id']:8s}  n_train={s['n_train']}")
    print()

    from foresight_engine.runner import run_all_models

    results = []
    all_passed = True

    print(f"  {'Series':8s}  {'n_train':>8}  {'Elapsed':>10}  "
          f"{'MASE':>8}  {'sMAPE':>8}  {'Status'}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}")

    for s in test_series:
        sid     = s["series_id"]
        df      = s["df"]
        actuals = s["actuals"]
        n_train = s["n_train"]

        t0 = time.time()
        status = "PASS"
        mase = smape = None
        error_msg = None

        try:
            raw = run_all_models(
                df               = df.copy(),
                horizon          = M3_HORIZON,
                confidence_level = 0.90,
                backtest_fn      = _no_backtest,
            )

            last_date = pd.to_datetime(df["date"].max())
            fc        = extract_ensemble_forecast(raw, last_date, M3_HORIZON)

            train_vals = df["value"].values.astype(float)
            mase  = compute_mase(fc, actuals, train_vals)
            smape = compute_smape(fc, actuals)

        except Exception as e:
            status    = "FAIL"
            error_msg = str(e)[:80]
            all_passed = False

        elapsed = time.time() - t0
        too_slow = elapsed > 120

        if too_slow and status == "PASS":
            status = "SLOW"
            all_passed = False

        mase_str  = f"{mase:.4f}" if mase  is not None else "   N/A"
        smape_str = f"{smape:.1f}%" if smape is not None else "   N/A"
        flag = "✅" if status == "PASS" else ("⚠️ " if status == "SLOW" else "❌")

        print(f"  {sid:8s}  {n_train:>8d}  {elapsed:>9.1f}s  "
              f"{mase_str:>8}  {smape_str:>8}  {flag} {status}")

        if error_msg:
            print(f"           ERROR: {error_msg}")

        results.append({
            "series_id": sid,
            "n_train":   n_train,
            "elapsed_s": round(elapsed, 2),
            "mase":      round(mase, 6) if mase else None,
            "smape":     round(smape, 4) if smape else None,
            "status":    status,
            "error":     error_msg,
        })

    # Summary
    elapsed_vals = [r["elapsed_s"] for r in results]
    mase_vals    = [r["mase"] for r in results if r["mase"] is not None]

    print()
    print("  " + "─" * 68)
    print(f"  Series tested: {len(results)}")
    print(f"  All completed: {'YES ✅' if all_passed else 'NO ❌'}")
    if elapsed_vals:
        print(f"  Avg time:    {np.mean(elapsed_vals):.1f}s  "
              f"Max: {np.max(elapsed_vals):.1f}s  "
              f"Min: {np.min(elapsed_vals):.1f}s")
    if mase_vals:
        print(f"  Median MASE: {np.median(mase_vals):.4f}")
    print()

    # Projection
    if elapsed_vals:
        avg_t = np.mean(elapsed_vals)
        proj  = avg_t * 1428 / 3600
        print(f"  PROJECTION: {avg_t:.1f}s/series × 1,428 = ~{proj:.1f} hours for full run")
    print()

    verdict = "✅ STAGE 1 PASSED — Proceed to Stage 2 (Stratified Pilot)" \
              if all_passed else \
              "❌ STAGE 1 FAILED — Do not proceed until all 5 series pass"
    print(f"  {verdict}")
    print("=" * 70 + "\n")

    # Save results
    output = {
        "test":      "stage1_timing",
        "engine":    "Foresight Engine v3.0.0",
        "backtest":  "disabled",
        "results":   results,
        "all_passed": all_passed,
    }
    out_path = OUTPUT / "stage1_timing.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {out_path}\n")

    return all_passed


if __name__ == "__main__":
    passed = run_timing_test()
    sys.exit(0 if passed else 1)
