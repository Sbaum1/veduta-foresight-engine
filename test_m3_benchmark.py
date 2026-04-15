# ==============================================================================
# FILE: test_m3_benchmark.py
# VERSION: 2.0.0
# ROLE: STAGE 4 — OFFICIAL M3 MONTHLY BENCHMARK (1,428 SERIES)
# ENGINE: VEDUTA Foresight Engine v3.0.0
#
# CITATIONS (MANDATORY — must appear in any publication referencing results):
#
#   Dataset:
#     Makridakis, S. & Hibon, M. (2000). 'The M3-Competition: results,
#     conclusions and implications.' International Journal of Forecasting,
#     16(4), 451-476. DOI: 10.1016/S0169-2070(00)00057-1
#
#   Primary metric (MASE):
#     Hyndman, R.J. & Koehler, A.B. (2006). 'Another look at measures of
#     forecast accuracy.' International Journal of Forecasting, 22(4),
#     679-688. DOI: 10.1016/j.ijforecast.2006.03.001
#
#   Secondary metric (sMAPE):
#     Makridakis, S. (1993). 'Accuracy measures: theoretical and practical
#     concerns.' International Journal of Forecasting, 9(4), 527-529.
#
# METHODOLOGY:
#   - Dataset:    1,428 M3 monthly series (Makridakis & Hibon, 2000)
#   - Horizon:    h=18 (official M3 monthly standard)
#   - MASE:       MAE_forecast / mean(|y_t - y_{t-12}|) on training set
#                 (seasonal naive denominator, m=12)
#   - sMAPE:      mean(200 * |actual - forecast| / (|actual| + |forecast|))
#   - Aggregation: Median MASE across all series (mean MAE across horizons)
#   - Backtest:   DISABLED — M3 provides official holdout actuals (series.xx)
#                 Rolling-origin CV is a VEDUTA UI feature, not a benchmark
#                 requirement. Disabling it does not affect forecast quality.
#   - Failures:   Engine exceptions recorded as crashes. No series excluded.
#
# USAGE:
#   cd C:\Dev\VEDUTA\core\foresight_x
#   python test_m3_benchmark.py             Full run
#   python test_m3_benchmark.py --resume    Resume from checkpoint
#
# PREREQUISITES:
#   Stage 1 (timing), Stage 2 (pilot), Stage 3 (extended) must all pass.
# ==============================================================================

from __future__ import annotations

import sys, json, time, argparse, hashlib
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
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "m3_benchmark_checkpoint.json"

PASS_MASE  = 0.72
SAVE_EVERY = 10

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
args = parser.parse_args()


def _no_backtest(df, model_runner, horizon, confidence_level):
    """
    No-op backtest for M3 benchmark.
    The M3 dataset provides official hold-out actuals. Rolling-origin
    cross-validation is not required and would add significant compute
    overhead without improving benchmark accuracy.
    """
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
            f"Primary Ensemble status={ens.get('status')} "
            f"error={ens.get('error','none')}"
        )
    fc_df = ens["forecast_df"].copy()
    fc_df["date"] = pd.to_datetime(fc_df["date"])
    future = fc_df[fc_df["date"] > last_train_date].copy()
    if len(future) < horizon:
        future = fc_df[fc_df["actual"].isna()].copy()
    future = future.head(horizon)
    if len(future) < horizon:
        raise RuntimeError(f"Insufficient forecast rows: {len(future)}")
    fc = future["forecast"].values.astype(float)
    if not np.isfinite(fc).all():
        raise RuntimeError("Non-finite values in Primary Ensemble forecast")
    return fc


def save_checkpoint(results, mase_vals, smape_vals, crashes, elapsed, n_total):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({
            "results":    results,
            "mase_vals":  mase_vals,
            "smape_vals": smape_vals,
            "crashes":    crashes,
            "elapsed":    elapsed,
            "n_total":    n_total,
        }, f, default=str)


def run_benchmark():
    ts = datetime.now()
    print()
    print("=" * 76)
    print("  VEDUTA Foresight Engine v3.0.0 — M3 OFFICIAL BENCHMARK")
    print(f"  Started: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  METHODOLOGY")
    print("  Dataset:   Makridakis & Hibon (2000), IJF 16(4):451-476")
    print("  MASE:      Hyndman & Koehler (2006), IJF 22(4):679-688")
    print("  sMAPE:     Makridakis (1993), IJF 9:527-529")
    print("  Horizon:   h=18 (official M3 monthly standard)")
    print("  Backtest:  Disabled — official holdout actuals used")
    print("=" * 76)

    print(f"\n  Loading M3 dataset...")
    all_series = load_m3_monthly(M3_PATH, verbose=True)
    print(f"  Expected: 1,428  Loaded: {len(all_series)}")
    print()

    from foresight_engine.runner import run_all_models

    results    = []
    mase_vals  = []
    smape_vals = []
    crashes    = 0
    start_idx  = 0

    # Resume from checkpoint
    if args.resume and CHECKPOINT.exists():
        cp = json.load(open(CHECKPOINT, encoding="utf-8"))
        results    = cp["results"]
        mase_vals  = cp["mase_vals"]
        smape_vals = cp.get("smape_vals", [])
        crashes    = cp["crashes"]
        start_idx  = len(results)
        print(f"  RESUMED from series {start_idx + 1} "
              f"({len(mase_vals)} valid so far)")
        print()

    t_start = time.time()

    for i, s in enumerate(all_series):
        if i < start_idx:
            continue

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
            error_msg = str(e)[:200]
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

        # Progress + checkpoint every SAVE_EVERY series
        done = i + 1 - start_idx
        if done % SAVE_EVERY == 0 or (i + 1) == len(all_series):
            elapsed_total = time.time() - t_start
            remain   = len(all_series) - i - 1
            eta      = (elapsed_total / max(done, 1)) * remain
            med_mase = round(float(np.median(mase_vals)), 4) if mase_vals else "—"
            med_smap = round(float(np.median(smape_vals)), 1) if smape_vals else "—"
            sps      = elapsed_total / max(done, 1)
            print(f"  [{i+1:4d}/{len(all_series)}]  "
                  f"MASE:{med_mase}  sMAPE:{med_smap}%  "
                  f"Crashes:{crashes}  "
                  f"{sps:.1f}s/series  ETA:{eta/60:.0f}min")
            save_checkpoint(results, mase_vals, smape_vals, crashes,
                            time.time() - t_start, len(all_series))

    # ── Final results ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    n_valid       = len(mase_vals)
    n_total       = len(all_series)

    arr_mase  = np.array(mase_vals)
    arr_smape = np.array(smape_vals) if smape_vals else np.array([np.nan])

    median_mase  = float(np.median(arr_mase))
    mean_mase    = float(np.mean(arr_mase))
    p25_mase     = float(np.percentile(arr_mase, 25))
    p75_mase     = float(np.percentile(arr_mase, 75))
    median_smape = float(np.median(arr_smape)) if smape_vals else None
    mean_smape   = float(np.mean(arr_smape))   if smape_vals else None

    passed = (crashes == 0 and median_mase <= PASS_MASE)

    print()
    print("─" * 76)
    print(f"  Series tested:    {n_total}")
    print(f"  Valid results:    {n_valid}")
    print(f"  Crashes:          {crashes}")
    print()
    print(f"  ── MASE (Hyndman & Koehler, 2006) ──────────────────────────────")
    print(f"  Median MASE:      {median_mase:.6f}  (threshold: ≤{PASS_MASE})")
    print(f"  Mean MASE:        {mean_mase:.6f}")
    print(f"  P25 / P75:        {p25_mase:.6f} / {p75_mase:.6f}")
    print()
    if median_smape:
        print(f"  ── sMAPE (Makridakis, 1993 — original M3 metric) ────────────────")
        print(f"  Median sMAPE:     {median_smape:.4f}%")
        print(f"  Mean sMAPE:       {mean_smape:.4f}%")
        print()
    print(f"  Elapsed:          {total_elapsed/3600:.2f} hours")
    print()
    print(f"  RESULT: {'✅ PASS' if passed else '❌ FAIL'}")
    print("=" * 76 + "\n")

    # ── SHA-256 certification hash ─────────────────────────────────────────────
    # Hash all results sorted by series_id for reproducibility
    cert_payload = json.dumps(
        sorted(results, key=lambda r: r["series_id"]),
        sort_keys=True, ensure_ascii=True, default=str
    )
    sha256 = hashlib.sha256(cert_payload.encode()).hexdigest()

    now = datetime.now().isoformat()

    official_metrics = {
        "engine":               "VEDUTA Foresight Engine v3.0.0",
        "benchmark":            "M3 Monthly — Makridakis & Hibon (2000)",
        "timestamp":            now,
        "horizon":              M3_HORIZON,
        "n_series_total":       n_total,
        "n_series_valid":       n_valid,
        "n_crashes":            crashes,
        "median_mase":          round(median_mase, 4),
        "mean_mase":            round(mean_mase, 4),
        "p25_mase":             round(p25_mase, 4),
        "p75_mase":             round(p75_mase, 4),
        "median_smape_pct":     round(median_smape, 2) if median_smape else None,
        "mean_smape_pct":       round(mean_smape, 2)   if mean_smape   else None,
        "sha256":               sha256,
        "mase_reference":       "Hyndman & Koehler (2006), IJF 22(4):679-688",
        "smape_reference":      "Makridakis (1993), IJF 9:527-529",
        "dataset_reference":    "Makridakis & Hibon (2000), IJF 16(4):451-476",
        "mase_formula":         "MAE_forecast / mean(|y_t - y_{t-12}|) on training set",
        "mase_denominator":     "Seasonal naive, m=12",
        "mase_aggregation":     "Median across 1,428 series (mean MAE per series)",
        "backtest":             "Disabled — M3 official holdout actuals used",
        "passed":               passed,
    }

    full_results = {
        "test":             "m3_official_benchmark",
        "engine":           "VEDUTA Foresight Engine v3.0.0",
        "timestamp":        now,
        "horizon":          M3_HORIZON,
        "n_total":          n_total,
        "n_valid":          n_valid,
        "n_crashes":        crashes,
        "median_mase":      round(median_mase, 6),
        "mean_mase":        round(mean_mase, 6),
        "p25_mase":         round(p25_mase, 6),
        "p75_mase":         round(p75_mase, 6),
        "median_smape":     round(median_smape, 4) if median_smape else None,
        "mean_smape":       round(mean_smape, 4)   if mean_smape   else None,
        "sha256":           sha256,
        "elapsed_s":        round(total_elapsed, 1),
        "passed":           passed,
        "threshold_mase":   PASS_MASE,
        "backtest":         "disabled",
        "methodology": {
            "mase_reference":    "Hyndman & Koehler (2006). IJF 22(4):679-688.",
            "smape_reference":   "Makridakis (1993). IJF 9:527-529.",
            "dataset_reference": "Makridakis & Hibon (2000). IJF 16(4):451-476.",
            "mase_formula":      "MAE_forecast / mean(|y_t - y_{t-12}|)",
            "smape_formula":     "mean(200*|a-f|/(|a|+|f|))",
            "aggregation":       "Median MASE across 1,428 series",
            "backtest_note":     "Disabled for M3 — official holdout actuals used",
        },
        "series": results,
    }

    summary = {
        "benchmark":        "M3 Monthly Competition",
        "engine":           "VEDUTA Foresight Engine v3.0.0",
        "timestamp":        now,
        "series_tested":    n_total,
        "valid_results":    n_valid,
        "crashes":          crashes,
        "median_mase":      round(median_mase, 6),
        "mean_mase":        round(mean_mase, 6),
        "median_smape":     round(median_smape, 4) if median_smape else None,
        "passed":           passed,
        "sha256":           sha256,
        "rank_note": (
            "Median MASE benchmarked against published methods on M3 Monthly "
            "dataset (h=18, seasonal naive denominator m=12). MASE per "
            "Hyndman & Koehler (2006). sMAPE per Makridakis (1993)."
        ),
    }

    for fname, data in [
        ("m3_results.json",          full_results),
        ("m3_summary.json",          summary),
        ("m3_official_metrics.json", official_metrics),
    ]:
        p = OUTPUT_DIR / fname
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved → {p}")

    print(f"\n  SHA-256: {sha256}")
    print(f"  Run python test_certify.py to generate the signed certificate.\n")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    return passed


if __name__ == "__main__":
    passed = run_benchmark()
    sys.exit(0 if passed else 1)
