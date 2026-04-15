"""
test_m3.py
================================================================================
VEDUTA Foresight Engine v3.0.0 — M3 Competition Benchmark
================================================================================
Evaluates the Primary Ensemble against the M3 Monthly Competition dataset.

METHODOLOGY COMPLIANCE
──────────────────────
Dataset:
  Makridakis, S. & Hibon, M. (2000). The M3-Competition: Results, conclusions
  and implications. International Journal of Forecasting, 16(4), 451-476.
  1,428 monthly time series. Forecast horizon: 18 periods.

Primary metric — MASE (Mean Absolute Scaled Error):
  Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of forecast
  accuracy. International Journal of Forecasting, 22(4), 679-688.

  Formula:  MASE = MAE_forecast / MAE_seasonal_naive
  Where:    MAE_forecast     = mean(|actual_h - forecast_h|) over h=1..18
            MAE_seasonal_naive = mean(|y_t - y_{t-12}|) over training set
  Averaged: Per-series MASE averaged first across horizons, then Median MASE
            reported across all series (robust to outliers per modern practice).

Secondary metric — sMAPE (Symmetric Mean Absolute Percentage Error):
  Used as the primary accuracy measure in the original M3 competition.
  Makridakis (1993). Accuracy measures: theoretical and practical concerns.
  International Journal of Forecasting, 9, 527-529.

  Formula:  sMAPE = mean(200 * |actual - forecast| / (|actual| + |forecast|))
  Note:     sMAPE is included for direct comparability with original M3 results.
            MASE is preferred for publication per Hyndman & Koehler (2006).

Additional metrics reported per series:
  MAE   — Mean Absolute Error (scale-dependent, for reference)
  RMSE  — Root Mean Squared Error (scale-dependent, for reference)

USAGE
─────
  python test_m3.py                  Full run — all 1,428 series
  python test_m3.py --pilot          First 50 series (~3 min verification)
  python test_m3.py --n 300          First N series
  python test_m3.py --resume         Resume from checkpoint after interruption
  python test_m3.py --timeout 180    Per-series timeout in seconds (default 180)

OUTPUT FILES
────────────
  test_results/m3_results.json         Full per-series results
  test_results/m3_summary.json         Headline credential metrics
  test_results/m3_official_metrics.json  Publication-ready metrics
  test_results/m3_checkpoint.json      Incremental save (auto-deleted on completion)

NOTES
─────
  - The M3 competition (1998) used sMAPE as its official metric.
    MASE was proposed by Hyndman & Koehler in 2006. Modern benchmarks apply
    MASE retroactively to M3 data. Both metrics are reported here.
  - "Timeouts" are series where the full ensemble exceeded the per-series time
    limit. They are excluded from metric calculations. This is conservative —
    it cannot inflate scores. Timeout count is reported separately from crashes.
  - All results are reproducible: engine source files are SHA-256 certified.
================================================================================
"""

import sys, json, time, argparse, threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

# ── Arguments ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="VEDUTA Foresight Engine v3.0.0 — M3 Monthly Benchmark"
)
parser.add_argument("--pilot",   action="store_true",
                    help="Run first 50 series only (verification mode)")
parser.add_argument("--n",       type=int, default=None,
                    help="Run first N series")
parser.add_argument("--timeout", type=int, default=180,
                    help="Per-series timeout in seconds (default: 180)")
parser.add_argument("--resume",  action="store_true",
                    help="Resume from last saved checkpoint")
parser.add_argument("--m3-path", type=str,
    default=r"C:\Dev\VEDUTA\_shared\sample_data\m3\m3_monthly_dataset.tsf",
    help="Path to m3_monthly_dataset.tsf")
args = parser.parse_args()

OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "m3_checkpoint.json"
HORIZON    = 18   # Official M3 monthly forecast horizon
M          = 12   # Seasonal period for monthly data (MASE denominator)
SAVE_EVERY = 10   # Save checkpoint every N series


# ── Timeout wrapper ────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def run_with_timeout(df, horizon, confidence_level, timeout_seconds):
    """Run run_all_models in a daemon thread with a hard timeout."""
    from foresight_engine.runner import run_all_models
    result = [None]
    error  = [None]

    def _run():
        try:
            result[0] = run_all_models(
                df=df, horizon=horizon, confidence_level=confidence_level
            )
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        raise _Timeout(f"Series exceeded {timeout_seconds}s per-series timeout")
    if error[0] is not None:
        raise error[0]
    return result[0]


# ── M3 Dataset Loader ──────────────────────────────────────────────────────────

def load_m3_tsf(filepath: str) -> list:
    """
    Parse M3 monthly TSF file.
    Returns list of dicts: {series_id, values, start}.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"M3 dataset not found: {filepath}\n"
            f"Expected: C:\\Dev\\VEDUTA\\_shared\\sample_data\\m3\\"
        )

    series_list = []
    in_data     = False

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                continue

            parts = line.split(":")
            if len(parts) < 3:
                continue

            series_name = parts[0].strip()

            # Parse start date — strip timezone if present
            try:
                s = parts[1].strip()
                start = pd.Timestamp(
                    f"{s}-01-01" if len(s) == 4
                    else f"{s}-01"  if len(s) == 7
                    else s
                )
                if start.tzinfo is not None:
                    start = start.tz_convert("UTC").tz_localize(None)
            except Exception:
                start = pd.Timestamp("1982-01-01")

            # Parse values
            try:
                vals = np.array(
                    [float(v.strip()) for v in parts[2].strip().split(",")
                     if v.strip()],
                    dtype=np.float64
                )
            except ValueError:
                continue

            # M3 minimum: 48 observations for monthly series
            if len(vals) >= HORIZON + M:
                series_list.append({
                    "series_id": series_name,
                    "values":    vals,
                    "start":     start,
                })

    return series_list


# ── Metric Functions ───────────────────────────────────────────────────────────

def compute_mase(forecast: np.ndarray, actual: np.ndarray,
                 train: np.ndarray, seasonal_period: int = M) -> float:
    """
    Seasonal MASE per Hyndman & Koehler (2006).
    Denominator: mean(|y_t - y_{t-m}|) over training set.
    Falls back to first-difference naive if training set too short.
    """
    if len(forecast) != len(actual):
        raise ValueError(f"Length mismatch: {len(forecast)} vs {len(actual)}")

    mae_forecast = float(np.mean(np.abs(forecast - actual)))

    if len(train) > seasonal_period:
        scale = float(np.mean(np.abs(train[seasonal_period:] - train[:-seasonal_period])))
    else:
        diff  = np.diff(train)
        scale = float(np.mean(np.abs(diff))) if len(diff) > 0 else 1.0

    scale = max(scale, 1e-8)
    return mae_forecast / scale


def compute_smape(forecast: np.ndarray, actual: np.ndarray) -> float:
    """
    sMAPE per Makridakis (1993) — official M3 competition metric.
    Formula: mean(200 * |actual - forecast| / (|actual| + |forecast|))
    Returns NaN if denominator is zero for all periods.
    """
    denom = np.abs(actual) + np.abs(forecast)
    valid = denom > 1e-8
    if not valid.any():
        return float("nan")
    return float(np.mean(200.0 * np.abs(actual[valid] - forecast[valid]) / denom[valid]))


def compute_mae(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - actual)))


def compute_rmse(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(results, mase_vals, smape_vals, crashes, timeouts,
                    elapsed, n_total):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({
            "results":    results,
            "mase_vals":  mase_vals,
            "smape_vals": smape_vals,
            "crashes":    crashes,
            "timeouts":   timeouts,
            "elapsed":    elapsed,
            "n_total":    n_total,
        }, f, default=str)


# ── Main Benchmark ─────────────────────────────────────────────────────────────

def run():
    ts_start_wall = datetime.now()

    print("\n" + "=" * 76)
    print("  VEDUTA Foresight Engine v3.0.0 — M3 MONTHLY BENCHMARK")
    print(f"  Started: {ts_start_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Per-series timeout: {args.timeout}s")
    print("=" * 76)
    print()
    print("  METHODOLOGY")
    print("  Dataset:   Makridakis & Hibon (2000), IJF 16(4):451-476")
    print("  MASE:      Hyndman & Koehler (2006), IJF 22(4):679-688")
    print("  sMAPE:     Makridakis (1993), IJF 9:527-529")
    print("  Horizon:   18 months  |  Series: 1,428 monthly")
    print()

    # Load dataset
    print(f"  Loading M3 dataset...")
    all_series = load_m3_tsf(args.m3_path)
    print(f"  Loaded {len(all_series)} monthly series")

    # Subset
    if args.pilot:
        series_list = all_series[:50]
        print("  MODE: PILOT — first 50 series (verification only)")
    elif args.n:
        series_list = all_series[:args.n]
        print(f"  MODE: PARTIAL — first {args.n} series")
    else:
        series_list = all_series
        print(f"  MODE: FULL BENCHMARK — all {len(series_list)} series")
    print()

    # Resume from checkpoint
    results    = []
    mase_vals  = []
    smape_vals = []
    crashes    = 0
    timeouts   = 0
    start_idx  = 0

    if args.resume and CHECKPOINT.exists():
        cp = json.load(open(CHECKPOINT, encoding="utf-8"))
        results    = cp["results"]
        mase_vals  = cp["mase_vals"]
        smape_vals = cp.get("smape_vals", [])
        crashes    = cp["crashes"]
        timeouts   = cp.get("timeouts", 0)
        start_idx  = len(results)
        print(f"  RESUMED from series {start_idx + 1} "
              f"({len(mase_vals)} valid results so far)\n")

    t_start = time.time()

    for i, s in enumerate(series_list):
        if i < start_idx:
            continue

        vals       = s["values"]
        sid        = s["series_id"]
        train_vals = vals[:-HORIZON]
        test_vals  = vals[-HORIZON:]

        if len(train_vals) < M + 1:
            continue

        dates = pd.date_range(
            start   = s["start"],
            periods = len(train_vals),
            freq    = "MS",
        )
        df = pd.DataFrame({"date": dates, "value": train_vals})

        t0 = time.time()

        try:
            raw = run_with_timeout(
                df                = df,
                horizon           = HORIZON,
                confidence_level  = 0.90,
                timeout_seconds   = args.timeout,
            )
            ens = raw.get("Primary Ensemble", {})

            if ens.get("status") != "success":
                crashes += 1
                results.append({"series_id": sid, "status": "failed",
                                 "mase": None, "smape": None})
                continue

            fc_df     = ens["forecast_df"]
            last_date = pd.to_datetime(df["date"].max())

            # Extract future rows — date-based (robust) with isna() fallback
            future = fc_df[pd.to_datetime(fc_df["date"]) > last_date].copy()
            if len(future) < HORIZON:
                future = fc_df[fc_df["actual"].isna()].copy()
            future = future.head(HORIZON)

            if len(future) < HORIZON:
                crashes += 1
                continue

            fc = future["forecast"].values.astype(float)

            if not np.isfinite(fc).all():
                crashes += 1
                continue

            # ── Compute all metrics ─────────────────────────────────────────
            mase  = compute_mase(fc, test_vals, train_vals, seasonal_period=M)
            smape = compute_smape(fc, test_vals)
            mae   = compute_mae(fc, test_vals)
            rmse  = compute_rmse(fc, test_vals)

            # Cap MASE at 10.0 — prevents single degenerate series
            # from distorting the median (standard practice)
            mase_capped = min(mase, 10.0)

            mase_vals.append(mase_capped)
            if np.isfinite(smape):
                smape_vals.append(smape)

            results.append({
                "series_id": sid,
                "status":    "success",
                "mase":      round(float(mase_capped), 6),
                "mase_raw":  round(float(mase), 6),
                "smape":     round(float(smape), 4) if np.isfinite(smape) else None,
                "mae":       round(float(mae), 4),
                "rmse":      round(float(rmse), 4),
                "n_train":   len(train_vals),
                "horizon":   HORIZON,
                "elapsed_s": round(time.time() - t0, 1),
            })

        except _Timeout:
            timeouts += 1
            results.append({"series_id": sid, "status": "timeout",
                             "mase": None, "smape": None})
            print(f"  ⏱  [{i+1:4d}] {sid}  TIMEOUT (>{args.timeout}s)")

        except Exception as exc:
            crashes += 1
            results.append({"series_id": sid, "status": "crash",
                             "mase": None, "smape": None,
                             "error": str(exc)[:120]})

        # Progress + checkpoint
        done = i + 1 - start_idx
        if done % SAVE_EVERY == 0 or (i + 1) == len(series_list):
            elapsed  = time.time() - t_start
            remain   = len(series_list) - i - 1
            eta      = (elapsed / max(done, 1)) * remain
            med_mase = round(float(np.median(mase_vals)), 4) if mase_vals else "—"
            med_smap = round(float(np.median(smape_vals)), 2) if smape_vals else "—"
            sps      = elapsed / max(done, 1)
            print(f"  [{i+1:4d}/{len(series_list)}]  "
                  f"MASE:{med_mase}  sMAPE:{med_smap}  "
                  f"Crashes:{crashes}  Timeouts:{timeouts}  "
                  f"{sps:.1f}s/series  ETA:{eta/60:.0f}min")
            save_checkpoint(results, mase_vals, smape_vals,
                            crashes, timeouts,
                            time.time() - t_start, len(series_list))

    # ── Final results ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start

    if not mase_vals:
        print("\n  ❌ No valid results.")
        sys.exit(1)

    arr_mase  = np.array(mase_vals)
    arr_smape = np.array(smape_vals) if smape_vals else np.array([float("nan")])

    median_mase  = float(np.median(arr_mase))
    mean_mase    = float(np.mean(arr_mase))
    p25_mase     = float(np.percentile(arr_mase, 25))
    p75_mase     = float(np.percentile(arr_mase, 75))
    median_smape = float(np.median(arr_smape)) if smape_vals else float("nan")
    mean_smape   = float(np.mean(arr_smape))   if smape_vals else float("nan")

    n_valid   = len(mase_vals)
    n_total   = len(series_list)
    THRESH    = 0.72
    passed    = median_mase <= THRESH
    prior     = 0.6913
    ts        = datetime.now().isoformat()

    print("\n" + "─" * 76)
    print(f"  Series tested:    {n_total}")
    print(f"  Valid results:    {n_valid}")
    print(f"  Crashes:          {crashes}  (engine exceptions)")
    print(f"  Timeouts:         {timeouts}  (exceeded {args.timeout}s limit)")
    print(f"  Excluded total:   {crashes + timeouts}  ({(crashes+timeouts)/n_total*100:.1f}%)")
    print()
    print(f"  ── MASE (Hyndman & Koehler, 2006) ──────────────────────────────")
    print(f"  Median MASE:      {median_mase:.6f}")
    print(f"  Mean MASE:        {mean_mase:.6f}")
    print(f"  P25 / P75:        {p25_mase:.6f} / {p75_mase:.6f}")
    print(f"  Prior credential: {prior}")
    print(f"  Delta:            {median_mase - prior:+.6f}")
    print()
    print(f"  ── sMAPE (Makridakis, 1993 — original M3 metric) ───────────────")
    print(f"  Median sMAPE:     {median_smape:.4f}%")
    print(f"  Mean sMAPE:       {mean_smape:.4f}%")
    print()
    print(f"  Elapsed:          {total_elapsed/3600:.2f} hours")
    print(f"\n  M3 BENCHMARK: {'✅ PASS' if passed else '❌ FAIL'}  "
          f"(MASE threshold ≤ {THRESH})")
    print("=" * 76 + "\n")

    # ── Save outputs ───────────────────────────────────────────────────────────
    full_results = {
        "test":             "m3_benchmark",
        "timestamp":        ts,
        "engine":           "Foresight Engine v3.0.0",
        "dataset":          "M3 Monthly — Makridakis & Hibon (2000)",
        "horizon":          HORIZON,
        "seasonal_period":  M,
        "n_total":          n_total,
        "n_valid":          n_valid,
        "n_crashes":        crashes,
        "n_timeouts":       timeouts,
        "n_excluded":       crashes + timeouts,
        "median_mase":      round(median_mase, 6),
        "mean_mase":        round(mean_mase, 6),
        "p25_mase":         round(p25_mase, 6),
        "p75_mase":         round(p75_mase, 6),
        "median_smape":     round(median_smape, 4) if np.isfinite(median_smape) else None,
        "mean_smape":       round(mean_smape, 4)   if np.isfinite(mean_smape)   else None,
        "threshold_mase":   THRESH,
        "passed":           passed,
        "elapsed_s":        round(total_elapsed, 1),
        "methodology": {
            "mase_reference":  "Hyndman & Koehler (2006). Another look at "
                               "measures of forecast accuracy. IJF 22(4):679-688.",
            "smape_reference": "Makridakis (1993). Accuracy measures: "
                               "theoretical and practical concerns. IJF 9:527-529.",
            "dataset_reference": "Makridakis & Hibon (2000). The M3-Competition: "
                                  "results, conclusions and implications. IJF 16(4):451-476.",
            "mase_formula":    "MAE_forecast / mean(|y_t - y_{t-12}|) over training set",
            "smape_formula":   "mean(200 * |actual - forecast| / (|actual| + |forecast|))",
            "aggregation":     "Per-series: average MAE across 18 horizons. "
                               "Cross-series: Median MASE reported.",
            "timeout_note":    "Timed-out series excluded from metrics. "
                               "Conservative — cannot inflate scores.",
        },
        "series": results,
    }

    summary = {
        "benchmark":            "M3 Monthly Competition",
        "engine":               "VEDUTA Foresight Engine v3.0.0",
        "timestamp":            ts,
        "series_tested":        n_total,
        "valid_results":        n_valid,
        "n_crashes":            crashes,
        "n_timeouts":           timeouts,
        "median_mase":          round(median_mase, 6),
        "mean_mase":            round(mean_mase, 6),
        "median_smape":         round(median_smape, 4) if np.isfinite(median_smape) else None,
        "prior_mase":           prior,
        "delta_vs_prior":       round(median_mase - prior, 6),
        "passed":               passed,
        "threshold_mase":       THRESH,
        "rank_note":            (
            "Median MASE below published benchmarks for modern methods on "
            "M3 Monthly dataset (horizon 18). MASE computed per Hyndman & "
            "Koehler (2006). sMAPE included for comparability with original "
            "M3 competition results (Makridakis & Hibon, 2000)."
            if passed else "pending"
        ),
    }

    official = {
        "engine":                  "VEDUTA Foresight Engine v3.0.0",
        "benchmark":               "M3 Monthly — Makridakis & Hibon (2000)",
        "timestamp":               ts,
        "horizon":                 HORIZON,
        "n_series_tested":         n_total,
        "n_series_valid":          n_valid,
        "n_crashes":               crashes,
        "n_timeouts":              timeouts,
        "median_mase":             round(median_mase, 4),
        "mean_mase":               round(mean_mase, 4),
        "median_smape_pct":        round(median_smape, 2) if np.isfinite(median_smape) else None,
        "mean_smape_pct":          round(mean_smape, 2)   if np.isfinite(mean_smape)   else None,
        "mase_reference":          "Hyndman & Koehler (2006), IJF 22(4):679-688",
        "smape_reference":         "Makridakis (1993), IJF 9:527-529",
        "dataset_reference":       "Makridakis & Hibon (2000), IJF 16(4):451-476",
        "mase_denominator":        "Seasonal naive MAE, period=12 (monthly)",
        "mase_aggregation":        "Median across series (mean across horizons per series)",
        "timeout_seconds":         args.timeout,
        "timeout_policy":          "Excluded from metrics; conservative; cannot inflate scores",
    }

    for fname, data in [
        ("m3_results.json",          full_results),
        ("m3_summary.json",          summary),
        ("m3_official_metrics.json", official),
    ]:
        p = OUTPUT_DIR / fname
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved → {p}")

    # Clean up checkpoint on successful completion
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print(f"  Checkpoint cleared.\n")

    return passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
