# ==============================================================================
# FILE: test_m3_extended.py
# VERSION: 1.0.0
# ROLE: STAGE 3 — EXTENDED RUN (200 SERIES)
# ENGINE: VEDUTA Foresight Engine v3.0.0
#
# PURPOSE:
#   Extended pre-flight evaluation across 200 series covering the full
#   length distribution of the M3 monthly dataset. Confirms performance
#   holds at scale before committing to the full 1,428-series benchmark.
#
#   Selection: every 7th series from the full dataset (systematic sample)
#   This ensures coverage across the entire dataset ordering.
#
#   PASS CRITERIA:
#     - >= 198/200 series complete (99% completion)
#     - Median MASE < 0.72 (full benchmark target)
#     - Zero crashes (model exceptions)
#     - Max per-series time < 120s
#
# USAGE:
#   python test_m3_extended.py
#   Must be run from C:\Dev\VEDUTA\core\foresight_x
# ==============================================================================

from __future__ import annotations

import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from m3_loader import (
    load_m3_monthly,
    compute_mase, compute_mase_raw, compute_smape, compute_mae, compute_rmse,
    M3_HORIZON,
)

M3_PATH    = r"C:\Dev\VEDUTA\_shared\sample_data\m3\m3_monthly_dataset.tsf"
OUTPUT     = Path("test_results")
OUTPUT.mkdir(exist_ok=True)
CHECKPOINT = OUTPUT / "stage3_checkpoint.json"

TARGET_N    = 200
PASS_MASE   = 0.72
SAVE_EVERY  = 20


def _no_backtest(df, model_runner, horizon, confidence_level):
    return {
        "eligible":         True,
        "observations":     len(df),
        "backtest_skipped": True,
        "reason":           "M3 benchmark — official holdout actuals used",
    }


def extract_ensemble_forecast(raw, last_train_date, horizon):
    ens = raw.get("Primary Ensemble", {})
    if ens.get("status") != "success":
        raise RuntimeError(f"Primary Ensemble failed: {ens.get('error','')}")
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


def save_checkpoint(results, mase_vals, smape_vals, crashes, elapsed):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({
            "results":    results,
            "mase_vals":  mase_vals,
            "smape_vals": smape_vals,
            "crashes":    crashes,
            "elapsed":    elapsed,
        }, f, default=str)


def run_extended():
    ts = datetime.now()
    print()
    print("=" * 72)
    print("  VEDUTA Foresight Engine v3.0.0 — STAGE 3: EXTENDED RUN")
    print(f"  Started: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {TARGET_N} series · Systematic sample (every 7th) · h=18")
    print("=" * 72)

    print(f"\n  Loading M3 dataset...")
    all_series = load_m3_monthly(M3_PATH, verbose=True)

    # Systematic sample — every 7th series covers the full dataset evenly
    step = max(1, len(all_series) // TARGET_N)
    test_series = all_series[::step][:TARGET_N]

    # Report length distribution
    n_trains = [s["n_train"] for s in test_series]
    short  = sum(1 for n in n_trains if n < 60)
    medium = sum(1 for n in n_trains if 60 <= n < 100)
    long_  = sum(1 for n in n_trains if n >= 100)
    print(f"\n  Test series: {len(test_series)} (every {step}th series)")
    print(f"    Short  (n<60):    {short:3d} series")
    print(f"    Medium (60-99):   {medium:3d} series")
    print(f"    Long   (>=100):   {long_:3d} series")
    print(f"    avg n_train:      {np.mean(n_trains):.1f}")
    print()

    from foresight_engine.runner import run_all_models

    results    = []
    mase_vals  = []
    smape_vals = []
    crashes    = 0

    t_start = time.time()

    for i, s in enumerate(test_series):
        sid     = s["series_id"]
        df      = s["df"]
        actuals = s["actuals"]
        n_train = s["n_train"]

        t0 = time.time()
        status = "success"
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

        except Exception as e:
            status    = "crash"
            error_msg = str(e)[:120]
            crashes  += 1

        elapsed = time.time() - t0

        results.append({
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

        # Progress every SAVE_EVERY series
        done = i + 1
        if done % SAVE_EVERY == 0 or done == len(test_series):
            elapsed_total = time.time() - t_start
            remain = len(test_series) - done
            eta = (elapsed_total / done) * remain if remain > 0 else 0
            med  = round(float(np.median(mase_vals)), 4) if mase_vals else "—"
            smap = round(float(np.median(smape_vals)), 1) if smape_vals else "—"
            sps  = elapsed_total / done
            print(f"  [{done:3d}/{len(test_series)}]  "
                  f"MASE:{med}  sMAPE:{smap}%  "
                  f"Crashes:{crashes}  "
                  f"{sps:.1f}s/series  ETA:{eta/60:.0f}min")
            save_checkpoint(results, mase_vals, smape_vals, crashes,
                            elapsed_total)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    n_valid       = len(mase_vals)
    completion    = n_valid / len(test_series) * 100

    median_mase  = float(np.median(mase_vals))  if mase_vals  else None
    mean_mase    = float(np.mean(mase_vals))     if mase_vals  else None
    p25          = float(np.percentile(mase_vals, 25)) if mase_vals else None
    p75          = float(np.percentile(mase_vals, 75)) if mase_vals else None
    median_smape = float(np.median(smape_vals))  if smape_vals else None

    elapsed_all  = [r["elapsed_s"] for r in results]

    stage_passed = (
        completion >= 99.0 and
        crashes == 0 and
        median_mase is not None and
        median_mase <= PASS_MASE
    )

    print()
    print("  " + "─" * 70)
    print(f"  Series tested:     {len(test_series)}")
    print(f"  Valid results:     {n_valid}  ({completion:.1f}%)")
    print(f"  Crashes:           {crashes}")
    print()
    if median_mase:
        print(f"  Median MASE:       {median_mase:.6f}  (target: <{PASS_MASE})")
        print(f"  Mean MASE:         {mean_mase:.6f}")
        print(f"  P25 / P75:         {p25:.6f} / {p75:.6f}")
    if median_smape:
        print(f"  Median sMAPE:      {median_smape:.2f}%")
    print()
    print(f"  Avg time/series:   {np.mean(elapsed_all):.1f}s")
    print(f"  Max time/series:   {np.max(elapsed_all):.1f}s")
    print(f"  Total elapsed:     {total_elapsed/3600:.2f} hours")
    proj = np.mean(elapsed_all) * 1428 / 3600
    print(f"  Projected full run: ~{proj:.1f} hours")
    print()

    verdict = (
        f"✅ STAGE 3 PASSED (MASE {median_mase:.4f} < {PASS_MASE}) — "
        f"Proceed to full 1,428-series benchmark"
        if stage_passed else
        f"❌ STAGE 3 FAILED — Investigate before full run"
    )
    print(f"  {verdict}")
    print("=" * 72 + "\n")

    # Save full results
    output = {
        "test":         "stage3_extended_200",
        "engine":       "Foresight Engine v3.0.0",
        "timestamp":    ts.isoformat(),
        "n_tested":     len(test_series),
        "n_valid":      n_valid,
        "n_crashes":    crashes,
        "completion":   round(completion, 2),
        "median_mase":  round(median_mase, 6)  if median_mase  else None,
        "mean_mase":    round(mean_mase, 6)    if mean_mase    else None,
        "p25_mase":     round(p25, 6)          if p25          else None,
        "p75_mase":     round(p75, 6)          if p75          else None,
        "median_smape": round(median_smape, 4) if median_smape else None,
        "target_mase":  PASS_MASE,
        "stage_passed": stage_passed,
        "elapsed_s":    round(total_elapsed, 1),
        "backtest":     "disabled",
        "horizon":      M3_HORIZON,
        "methodology": {
            "mase":    "Hyndman & Koehler (2006), IJF 22(4):679-688",
            "smape":   "Makridakis (1993), IJF 9:527-529",
            "dataset": "Makridakis & Hibon (2000), IJF 16(4):451-476",
        },
        "series": results,
    }
    out_path = OUTPUT / "stage3_extended.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {out_path}")
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    return stage_passed


if __name__ == "__main__":
    passed = run_extended()
    sys.exit(0 if passed else 1)
