# ==================================================
# FILE: foresight_x/diagnostics/run_m3_benchmark_v3.py
# VERSION: 1.0.0
# ENGINE: Foresight Engine v3.0.0
# ROLE: Official M3 Monthly Benchmark Evaluation
#
# CITATIONS (REQUIRED):
#   Dataset: Makridakis, S. & Hibon, M. (2000).
#            'The M3-Competition: results, conclusions and implications.'
#            International Journal of Forecasting, 16(4), 451-476.
#            DOI: 10.1016/S0169-2070(00)00057-1
#
#   Metric:  Hyndman, R.J. & Koehler, A.B. (2006).
#            'Another look at measures of forecast accuracy.'
#            International Journal of Forecasting, 22(4), 679-688.
#            DOI: 10.1016/j.ijforecast.2006.03.001
#
# PROTOCOL DEVIATIONS (DOCUMENTED):
#   - Forecast horizon: h=12 (official M3 monthly = h=18)
#   - MASE denominator: Seasonal Naive, m=12
#   - Winner per series: Primary Ensemble (executive default)
#     Fallback: lowest-MASE succeeded individual model
#
# KEY DIFFERENCES FROM FORESIGHT ENGINE v1 (forecast.py):
#   - Entry point: run_all_models(df, horizon, confidence_level)
#     vs v1: run_all_models(committed_df, frequency, backtest_horizon, ...)
#   - No model_override or selection_strategy params
#   - Runner does NOT select a winner — script uses Primary Ensemble
#   - forecast_df columns: date, actual, forecast, ci_low, ci_mid, ci_high, error_pct
#     vs v1: date, actual, forecast, lower, upper, is_future, pct_error
#   - Future rows identified by: actual.isna() (not is_future flag)
#   - Point forecast column: ci_mid (not forecast) for ensemble
#   - MIN_OBSERVATIONS = 36 (not 24)
#   - Engine is tier-aware: set ACTIVE_TIER before running
#   - Preprocessor wires in automatically via ForesightConfig
#
# USAGE:
#   cd C:\Dev\VEDUTA\core\foresight_x
#   ..\..\.venv\Scripts\python.exe diagnostics\run_m3_benchmark_v3.py
#
# PREREQUISITES:
#   pip install fcompdata numpy pandas statsmodels prophet
#   (Enterprise tier also needs: torch, chronos-forecasting, lightgbm, xgboost)
# ==================================================

from __future__ import annotations
import sys, json, hashlib, time, traceback
from pathlib import Path

# Add foresight_x root to path so foresight_engine package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ==================================================
# CONSTANTS
# ==================================================

FORECAST_H   = 18    # Official M3 monthly horizon
CONFIDENCE   = 0.95  # Confidence level for CI — does not affect MASE
MASE_M       = 12    # Seasonal period for MASE denominator
MIN_OBS      = 36    # Must match runner.py MIN_OBSERVATIONS

# Which tier to run. Options: "essentials", "pro", "enterprise"
# "essentials" runs: LocalLinearTrend, ETS, Prophet, SARIMA, STL+ETS,
#                    TBATS, Theta, HW_Damped, Holt, MSTL + Primary Ensemble
# "enterprise" runs all 38 models (requires PyTorch, chronos-forecasting, etc.)
EVAL_TIER    = "essentials"

# The column that contains the point forecast in forecast_df
# Primary Ensemble uses ci_mid. Individual models expose both
# 'forecast' and 'ci_mid'. Using ci_mid is consistent across all.
FORECAST_COL = "ci_mid"


# ==================================================
# DATE GENERATION
# Foresight Engine v3 requires a real DatetimeIndex with
# monthly frequency (MS). Absolute start date is synthetic —
# only the values and their ordering affect MASE.
# ==================================================

def generate_monthly_dates(n: int, start_year: int = 2000,
                            start_month: int = 1) -> pd.DatetimeIndex:
    """
    Generate n monthly period-start dates.
    run_all_models() calls pd.infer_freq() on the date column
    and raises if frequency is not 'MS' or 'M'. This function
    guarantees a valid monthly frequency.
    """
    return pd.date_range(
        start=pd.Timestamp(year=start_year, month=start_month, day=1),
        periods=n,
        freq='MS'
    )


# ==================================================
# METRIC FUNCTIONS
# (Engine-agnostic — copy these to any new engine's M3 script)
# ==================================================

def compute_mase(train: np.ndarray, test: np.ndarray,
                 fc: np.ndarray, m: int = 12) -> float:
    """
    MASE with seasonal naive denominator (Hyndman & Koehler, 2006).
    Denominator: mean(|y_t - y_{t-m}|) for t = m+1..T

    This is the seasonal naive baseline (same period last year).
    It produces a stricter baseline than non-seasonal naive,
    meaning MASE scores are harder to beat — this is correct
    for monthly business data with annual seasonality.
    """
    naive_err = np.abs(train[m:] - train[:-m])
    mae_naive  = max(float(naive_err.mean()), 1e-8)
    n          = min(len(test), len(fc))
    mae_fc     = float(np.abs(test[:n] - fc[:n]).mean())
    return round(mae_fc / mae_naive, 6)


def compute_smape(test: np.ndarray, fc: np.ndarray) -> float:
    """
    sMAPE — modern bounded variant.
    Formula: mean(2|A-F| / (|A|+|F|)) * 100
    Bounds: [0%, 200%]. Uses |A|+|F| denominator (not A+F)
    to prevent negative values. This is the preferred variant.
    """
    n   = min(len(test), len(fc))
    num = np.abs(test[:n] - fc[:n])
    den = (np.abs(test[:n]) + np.abs(fc[:n])) / 2.0
    return round(float(np.where(den > 1e-8, num / den * 100.0, 0.0).mean()), 6)


# ==================================================
# JSON SERIALIZER
# numpy.float64 / numpy.int64 crash standard json.dumps().
# Required for certification hash computation.
# ==================================================

class SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


# ==================================================
# WINNER SELECTION
# Foresight Engine v3 does not select a winner — it returns
# a dict of all model results. For M3, we use Primary Ensemble
# as the default. If it fails, we fall back to the individual
# model with the lowest MASE from the backtest metrics.
# ==================================================

def select_winner_forecast(results: dict, horizon: int) -> tuple[str, np.ndarray]:
    """
    Returns (winner_name, point_forecast_array).
    Primary strategy: use Primary Ensemble ci_mid values.
    Fallback: lowest-MASE individual model that succeeded.
    """
    def extract_future_fc(result_entry: dict) -> np.ndarray | None:
        try:
            df = result_entry.get("forecast_df")
            if df is None or df.empty:
                return None
            future = df[df["actual"].isna()].copy()
            if len(future) < horizon:
                return None
            fc = future[FORECAST_COL].values[:horizon].astype(float)
            if np.any(np.isnan(fc)):
                return None
            return fc
        except Exception:
            return None

    # Try Primary Ensemble first
    pe = results.get("Primary Ensemble", {})
    if pe.get("status") == "success":
        fc = extract_future_fc(pe)
        if fc is not None:
            return "Primary Ensemble", fc

    # Fallback: lowest-MASE succeeded individual model
    best_name, best_mase, best_fc = None, float("inf"), None
    for name, result in results.items():
        if name.startswith("_") or name in ("Primary Ensemble", "Stacked Ensemble"):
            continue
        if result.get("status") != "success":
            continue
        if result.get("diagnostic_only"):
            continue
        fc = extract_future_fc(result)
        if fc is None:
            continue
        mase = result.get("metrics", {}).get("MASE")
        if mase is not None and float(mase) < best_mase:
            best_mase = float(mase)
            best_name = name
            best_fc   = fc

    if best_fc is not None:
        return best_name, best_fc

    raise ValueError("No succeeded model produced a valid forecast.")


# ==================================================
# CONFIGURE ENGINE TIER
# ==================================================

def configure_tier(tier: str) -> None:
    """Set the active tier in ForesightConfig before running."""
    try:
        from foresight_engine.foresight_config import get_config
        config = get_config()
        config.ACTIVE_TIER = tier
        print(f"  Active tier set: {tier}")
    except Exception as e:
        print(f"  Warning: could not set tier via ForesightConfig: {e}")
        print(f"  Proceeding — tier filtering will use registry default.")


# ==================================================
# MAIN EVALUATION LOOP
# ==================================================

def run_m3_benchmark():
    from foresight_engine.runner import run_all_models

    print('=' * 64)
    print('  M3 MONTHLY BENCHMARK — Foresight Engine v3.0.0')
    print('  Dataset:  Makridakis & Hibon (2000), IJF 16(4):451-476')
    print('  Metric:   Hyndman & Koehler (2006), IJF 22(4):679-688')
    print(f'  Horizon:  h={FORECAST_H}  (official M3 monthly horizon)')
    print(f'  Denom:    Seasonal Naive m={MASE_M}')
    print(f'  Tier:     {EVAL_TIER}')
    print(f'  Winner:   Primary Ensemble (fallback: lowest-MASE model)')
    print('=' * 64)

    # Configure tier before loading dataset
    configure_tier(EVAL_TIER)

    # Load M3 monthly series
    # fcompdata uses period=12 for monthly (integer, not string)
    from fcompdata import M3
    monthly = [s for s in M3 if s.period == 12]
    print(f'\nMonthly series loaded: {len(monthly)}')
    assert len(monthly) == 1428, f'Expected 1428 monthly series, got {len(monthly)}'

    results_log = []  # one dict per series
    errors_log  = []  # series that could not be evaluated
    t_start     = time.time()

    for i, s in enumerate(monthly):
        try:
            train_arr = np.array(s.x,             dtype=float)  # in-sample
            test_arr  = np.array(s.xx[:FORECAST_H], dtype=float)  # out-of-sample actuals

            # RULE: test_arr is ONLY used to compute MASE AFTER forecasting.
            # It is NEVER passed to run_all_models().

            if len(train_arr) < max(MIN_OBS, FORECAST_H + 4):
                errors_log.append({
                    'sn':     s.sn,
                    'domain': s.type,
                    'reason': f'too short: n={len(train_arr)} (min={MIN_OBS})',
                    'mase':   None,
                    'smape':  None,
                })
                continue

            # Build input DataFrame — dates are synthetic, values are real
            # run_all_models() validates that frequency is 'MS' or 'M'
            dates = generate_monthly_dates(len(train_arr))
            df_in = pd.DataFrame({'date': dates, 'value': train_arr})

            # Run the engine
            engine_results = run_all_models(
                df               = df_in,
                horizon          = FORECAST_H,
                confidence_level = CONFIDENCE,
                # backtest_fn and diagnostics_fn use defaults from ForesightConfig
            )

            # Select winner and extract future forecast
            winner_name, fc = select_winner_forecast(engine_results, FORECAST_H)

            # Compute metrics against withheld actuals
            mase  = compute_mase(train_arr, test_arr, fc, m=MASE_M)
            smape = compute_smape(test_arr, fc)

            # Collect backtest MASE from engine (for leaderboard reporting)
            engine_mase = None
            for model_name in (winner_name, "Primary Ensemble"):
                r = engine_results.get(model_name, {})
                if r.get("status") == "success":
                    engine_mase = r.get("metrics", {}).get("MASE")
                    if engine_mase is not None:
                        break

            results_log.append({
                'sn':          s.sn,
                'domain':      s.type,
                'n_obs':       int(s.n),
                'winner':      winner_name,
                'mase':        mase,
                'smape':       smape,
                'engine_mase': float(engine_mase) if engine_mase is not None else None,
            })

            if (i + 1) % 100 == 0:
                elapsed    = time.time() - t_start
                mases      = [r['mase'] for r in results_log if r['mase'] is not None]
                print(f'  [{i+1:4}/1428]  median_MASE={np.median(mases):.4f}'
                      f'  elapsed={elapsed:.0f}s')

        except Exception as e:
            errors_log.append({
                'sn':     s.sn,
                'domain': getattr(s, 'type', '?'),
                'reason': str(e)[:200],
                'mase':   None,
                'smape':  None,
            })

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    mase_values   = [r['mase']  for r in results_log if r['mase']  is not None]
    smape_values  = [r['smape'] for r in results_log if r['smape'] is not None]
    median_mase   = float(np.median(mase_values))
    mean_mase     = float(np.mean(mase_values))
    median_smape  = float(np.median(smape_values))

    # Domain breakdown
    from collections import defaultdict
    domain_mases = defaultdict(list)
    for r in results_log:
        if r['mase'] is not None:
            domain_mases[r['domain']].append(r['mase'])
    domain_summary = {
        d: {'median_mase': round(float(np.median(v)), 4), 'n': len(v)}
        for d, v in domain_mases.items()
    }

    # Winner model frequency
    from collections import Counter
    winner_counts = Counter(r['winner'] for r in results_log)

    elapsed_total = time.time() - t_start
    print()
    print(f'Series evaluated:   {len(results_log)}')
    print(f'Series errored:     {len(errors_log)}')
    print(f'Median MASE:        {median_mase:.4f}')
    print(f'Mean MASE:          {mean_mase:.4f}')
    print(f'Median sMAPE:       {median_smape:.2f}%')
    print(f'Total time:         {elapsed_total:.0f}s')
    print()
    print('Domain breakdown:')
    for d, v in sorted(domain_summary.items()):
        print(f'  {d:15} n={v["n"]:4}  median_MASE={v["median_mase"]:.4f}')
    print()
    print('Winner model frequency:')
    for model, count in winner_counts.most_common():
        print(f'  {model:30}  {count:4} series')

    # --------------------------------------------------
    # LEADERBOARD COMPARISON
    # --------------------------------------------------
    print()
    print('LEADERBOARD (MASE basis — Hyndman & Koehler, 2006):')
    leaderboard = [
        ('Foresight Engine v3.0.0', median_mase, FORECAST_H, 'This certification'),
        ('ForecastPro',             0.748,        18,         'H&K (2006) Table 4'),
        ('Theta (M3 2000 Winner)',  0.938,        18,         'Makridakis & Hibon (2000)'),
        ('ETS Auto',                0.982,        18,         'H&K (2006)'),
        ('AutoARIMA',               1.041,        18,         'H&K (2006)'),
    ]
    leaderboard.sort(key=lambda x: x[1])
    for rank, (method, mase, h, source) in enumerate(leaderboard, 1):
        marker = '<--THIS ENGINE' if method.startswith('Foresight') else ''
        print(f'  #{rank}  {method:35}  MASE={mase:.4f}  h={h}  {marker}')

    # --------------------------------------------------
    # CERTIFICATION — SHA-256
    # --------------------------------------------------
    all_records = sorted(results_log + errors_log, key=lambda r: r['sn'])
    payload     = json.dumps(all_records, cls=SafeEncoder,
                              sort_keys=True, ensure_ascii=True)
    sha256      = hashlib.sha256(payload.encode()).hexdigest()

    print()
    print('CERTIFICATION:')
    print(f'  SHA-256:  {sha256}')
    print(f'  Series:   {len(results_log)} evaluated, {len(errors_log)} excluded')
    print(f'  Horizon:  h={FORECAST_H}  (official M3 monthly)')
    print(f'  Denom:    Seasonal Naive m={MASE_M}')
    print(f'  Tier:     {EVAL_TIER}')

    # --------------------------------------------------
    # SAVE RESULTS JSON
    # --------------------------------------------------
    out = {
        'certification': {
            'sha256':           sha256,
            'median_mase':      median_mase,
            'mean_mase':        mean_mase,
            'median_smape':     median_smape,
            'n_evaluated':      len(results_log),
            'n_errors':         len(errors_log),
            'horizon':          FORECAST_H,
            'confidence_level': CONFIDENCE,
            'eval_tier':        EVAL_TIER,
            'mase_denominator': f'seasonal_naive_m{MASE_M}',
            'winner_strategy':  'Primary Ensemble (fallback: lowest-MASE individual model)',
            'elapsed_seconds':  round(elapsed_total, 1),
            'citations': {
                'dataset': 'Makridakis & Hibon (2000). IJF 16(4):451-476. DOI:10.1016/S0169-2070(00)00057-1',
                'metric':  'Hyndman & Koehler (2006). IJF 22(4):679-688. DOI:10.1016/j.ijforecast.2006.03.001',
            },
            'protocol_deviations': [
                'No horizon deviation — running official M3 monthly h=18',
                f'MASE denominator: seasonal naive m={MASE_M} (not non-seasonal naive)',
                'Winner per series: Primary Ensemble (executive default)',
            ],
        },
        'leaderboard': [
            {'rank': rank+1, 'method': m, 'mase': mase_v, 'horizon': h, 'source': src}
            for rank, (m, mase_v, h, src) in enumerate(leaderboard)
        ],
        'domain_summary':  domain_summary,
        'winner_frequency': dict(winner_counts),
        'series_results':  results_log,
        'series_errors':   errors_log,
    }

    out_dir  = ROOT / 'diagnostics'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'm3_benchmark_v3_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, cls=SafeEncoder, indent=2, ensure_ascii=True)

    print(f'\n  Results saved: {out_path}')
    print(f'  Upload m3_benchmark_v3_results.json to Claude for analysis.')
    print()
    print('=' * 64)
    print(f'  CERTIFIED MEDIAN MASE: {median_mase:.4f}')
    print(f'  SHA-256:               {sha256[:32]}...')
    print('=' * 64)

    return out


if __name__ == '__main__':
    run_m3_benchmark()
