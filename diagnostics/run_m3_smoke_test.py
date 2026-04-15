# ==================================================
# FILE: foresight_x/diagnostics/run_m3_smoke_test.py
# VERSION: 1.0.0
# ENGINE: Foresight Engine v3.0.0
# ROLE: M3 Smoke Test — 10-series stratified validation
#
# PURPOSE:
#   Confirms the full M3 certification pipeline works correctly
#   before committing to the 28-hour full run. Tests:
#     1. Data loading and parsing (fcompdata)
#     2. Engine execution (run_all_models)
#     3. Winner selection (Primary Ensemble fallback)
#     4. MASE / sMAPE computation against withheld actuals
#     5. JSON serialization (SafeEncoder numpy handling)
#     6. SHA-256 certification hash
#     7. Output file write
#     8. Structural validation of every result record
#
# STRATIFICATION:
#   2 series per observation-length bucket:
#     Short  : n = 48-59
#     Medium : n = 60-79
#     Long   : n = 80-99
#     Extended: n = 100-119
#     Full   : n = 120+
#
# USAGE:
#   cd C:\Dev\VEDUTA\core\foresight_x
#   ..\\.venv\\Scripts\\python.exe diagnostics\\run_m3_smoke_test.py
#
# EXPECTED RUNTIME: ~20-40 minutes (10 series x 2-4 min each)
#
# SUCCESS CRITERIA:
#   - 0 engine failures
#   - All 10 records have valid mase (float, > 0)
#   - Median MASE < 1.5 (well below naive baseline)
#   - SHA-256 hash generated and non-empty
#   - Output JSON written cleanly
#   - All structural validation checks pass
# ==================================================

from __future__ import annotations
import sys, json, hashlib, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ==================================================
# CONSTANTS — must match run_m3_benchmark_v3.py exactly
# ==================================================
FORECAST_H   = 18    # Official M3 monthly horizon
CONFIDENCE   = 0.95
MASE_M       = 12    # Seasonal naive denominator
MIN_OBS      = 36    # Must match runner.py MIN_OBSERVATIONS
EVAL_TIER    = "essentials"
FORECAST_COL = "ci_mid"

# Smoke test: 2 series from each of 5 length buckets = 10 total
# Using fixed indices for reproducibility — same series every run
SMOKE_BUCKETS = {
    "short_48-59":     [0,  1],   # first 2 series with n in [48,59]
    "medium_60-79":    [0,  1],   # first 2 with n in [60,79]
    "long_80-99":      [0,  1],   # first 2 with n in [80,99]
    "extended_100-119":[0,  1],   # first 2 with n in [100,119]
    "full_120plus":    [0,  1],   # first 2 with n >= 120
}


# ==================================================
# SHARED FUNCTIONS (identical to benchmark script)
# ==================================================

def generate_monthly_dates(n: int, start_year: int = 2000,
                            start_month: int = 1) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(year=start_year, month=start_month, day=1),
        periods=n, freq='MS'
    )


def compute_mase(train: np.ndarray, test: np.ndarray,
                 fc: np.ndarray, m: int = 12) -> float:
    naive_err = np.abs(train[m:] - train[:-m])
    mae_naive  = max(float(naive_err.mean()), 1e-8)
    n          = min(len(test), len(fc))
    mae_fc     = float(np.abs(test[:n] - fc[:n]).mean())
    return round(mae_fc / mae_naive, 6)


def compute_smape(test: np.ndarray, fc: np.ndarray) -> float:
    n   = min(len(test), len(fc))
    num = np.abs(test[:n] - fc[:n])
    den = (np.abs(test[:n]) + np.abs(fc[:n])) / 2.0
    return round(float(np.where(den > 1e-8, num / den * 100.0, 0.0).mean()), 6)


class SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def select_winner_forecast(results: dict, horizon: int) -> tuple[str, np.ndarray]:
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

    pe = results.get("Primary Ensemble", {})
    if pe.get("status") == "success":
        fc = extract_future_fc(pe)
        if fc is not None:
            return "Primary Ensemble", fc

    best_name, best_mase, best_fc = None, float("inf"), None
    for name, result in results.items():
        if name.startswith("_") or name in ("Primary Ensemble", "Stacked Ensemble"):
            continue
        if result.get("status") != "success" or result.get("diagnostic_only"):
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


def configure_tier(tier: str) -> None:
    try:
        from foresight_engine.foresight_config import get_config
        config = get_config()
        config.ACTIVE_TIER = tier
    except Exception as e:
        print(f"  Warning: could not set tier: {e}")


# ==================================================
# STRATIFIED SAMPLE SELECTION
# ==================================================

def select_smoke_series(monthly: list) -> list:
    """Pick 2 series from each of 5 length buckets."""
    buckets = {
        "short_48-59":      [s for s in monthly if 48 <= s.n <= 59],
        "medium_60-79":     [s for s in monthly if 60 <= s.n <= 79],
        "long_80-99":       [s for s in monthly if 80 <= s.n <= 99],
        "extended_100-119": [s for s in monthly if 100 <= s.n <= 119],
        "full_120plus":     [s for s in monthly if s.n >= 120],
    }

    selected = []
    print("\n  Stratified sample:")
    for bucket_name, series_list in buckets.items():
        count = min(2, len(series_list))
        picks = series_list[:count]
        for s in picks:
            selected.append((bucket_name, s))
            print(f"    {bucket_name:25}  {s.sn}  n={s.n}")
        if len(series_list) < 2:
            print(f"    WARNING: bucket '{bucket_name}' has only {len(series_list)} series")

    print(f"\n  Total selected: {len(selected)} series")
    return selected


# ==================================================
# STRUCTURAL VALIDATION
# ==================================================

def validate_result_record(record: dict, series_id: str) -> list[str]:
    """Returns list of validation errors. Empty = valid."""
    errors = []

    required_keys = ['sn', 'domain', 'n_obs', 'winner', 'mase', 'smape']
    for k in required_keys:
        if k not in record:
            errors.append(f"{series_id}: missing key '{k}'")

    mase = record.get('mase')
    if mase is None:
        errors.append(f"{series_id}: mase is None (engine failure)")
    elif not isinstance(mase, (int, float)):
        errors.append(f"{series_id}: mase is not numeric: {type(mase)}")
    elif mase <= 0:
        errors.append(f"{series_id}: mase <= 0: {mase}")
    elif mase > 50:
        errors.append(f"{series_id}: mase suspiciously high: {mase} (check series)")

    smape = record.get('smape')
    if smape is None:
        errors.append(f"{series_id}: smape is None")
    elif not (0 <= smape <= 200):
        errors.append(f"{series_id}: smape out of bounds [0,200]: {smape}")

    winner = record.get('winner')
    if not winner or not isinstance(winner, str):
        errors.append(f"{series_id}: winner missing or not a string")

    return errors


# ==================================================
# MAIN SMOKE TEST
# ==================================================

def run_smoke_test():
    from foresight_engine.runner import run_all_models

    print('=' * 64)
    print('  M3 SMOKE TEST — Foresight Engine v3.0.0')
    print(f'  Horizon: h={FORECAST_H} | Tier: {EVAL_TIER} | Series: 10')
    print('=' * 64)

    configure_tier(EVAL_TIER)

    from fcompdata import M3
    monthly = [s for s in M3 if s.period == 12]
    assert len(monthly) == 1428, f"Expected 1428 monthly, got {len(monthly)}"
    print(f"\n  M3 monthly series loaded: {len(monthly)}")

    smoke_series = select_smoke_series(monthly)

    results_log = []
    errors_log  = []
    val_errors  = []
    t_start     = time.time()

    print()
    print(f"  {'Series':8}  {'Bucket':25}  {'n':5}  {'Time':7}  {'Winner':25}  {'MASE':8}  {'sMAPE':8}  Status")
    print(f"  {'-'*8}  {'-'*25}  {'-'*5}  {'-'*7}  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*6}")

    for bucket_name, s in smoke_series:
        t0 = time.time()
        try:
            train_arr = np.array(s.x,                dtype=float)
            test_arr  = np.array(s.xx[:FORECAST_H],  dtype=float)

            if len(train_arr) < max(MIN_OBS, FORECAST_H + 4):
                errors_log.append({
                    'sn': s.sn, 'domain': s.type,
                    'reason': f'too short: n={len(train_arr)}',
                    'mase': None, 'smape': None,
                })
                elapsed = time.time() - t0
                print(f"  {s.sn:8}  {bucket_name:25}  {s.n:5}  {elapsed:5.1f}s  {'—':25}  {'—':8}  {'—':8}  SKIP")
                continue

            dates = generate_monthly_dates(len(train_arr))
            df_in = pd.DataFrame({'date': dates, 'value': train_arr})

            engine_results = run_all_models(
                df               = df_in,
                horizon          = FORECAST_H,
                confidence_level = CONFIDENCE,
            )

            winner_name, fc = select_winner_forecast(engine_results, FORECAST_H)

            mase  = compute_mase(train_arr, test_arr, fc, m=MASE_M)
            smape = compute_smape(test_arr, fc)

            # Cap extreme MASE (consistent with benchmark script)
            mase_raw = mase
            mase     = min(mase, 10.0)

            elapsed = time.time() - t0

            record = {
                'sn':       s.sn,
                'bucket':   bucket_name,
                'domain':   s.type,
                'n_obs':    int(s.n),
                'winner':   winner_name,
                'mase':     mase,
                'mase_raw': mase_raw,
                'smape':    smape,
                'elapsed_s': round(elapsed, 1),
            }
            results_log.append(record)

            # Validate record structure
            rec_errors = validate_result_record(record, s.sn)
            val_errors.extend(rec_errors)

            status = "HIGH" if mase_raw > 10 else ("WARN" if mase > 1.0 else "PASS")
            winner_short = winner_name[:24]
            print(f"  {s.sn:8}  {bucket_name:25}  {s.n:5}  {elapsed:5.1f}s"
                  f"  {winner_short:25}  {mase:8.4f}  {smape:7.2f}%  {status}")

        except Exception as e:
            elapsed = time.time() - t0
            errors_log.append({
                'sn': s.sn, 'domain': getattr(s, 'type', '?'),
                'reason': str(e)[:200], 'mase': None, 'smape': None,
            })
            print(f"  {s.sn:8}  {bucket_name:25}  {s.n:5}  {elapsed:5.1f}s"
                  f"  {'ERROR':25}  {'—':8}  {'—':8}  FAIL")
            print(f"           {str(e)[:80]}")

    # ==================================================
    # RESULTS SUMMARY
    # ==================================================
    elapsed_total = time.time() - t_start
    mase_values   = [r['mase']  for r in results_log if r['mase']  is not None]
    smape_values  = [r['smape'] for r in results_log if r['smape'] is not None]

    print()
    print('=' * 64)
    print('  SMOKE TEST RESULTS')
    print('=' * 64)
    print(f"  Series attempted:  {len(smoke_series)}")
    print(f"  Engine successes:  {len(results_log)}")
    print(f"  Engine failures:   {len(errors_log)}")
    print(f"  Total elapsed:     {elapsed_total:.1f}s = {elapsed_total/60:.1f}min")
    print()

    if mase_values:
        print(f"  Median MASE:       {np.median(mase_values):.4f}")
        print(f"  Mean MASE:         {np.mean(mase_values):.4f}")
        print(f"  MASE range:        {min(mase_values):.4f} — {max(mase_values):.4f}")
        print(f"  Median sMAPE:      {np.median(smape_values):.2f}%")

    # Winner frequency
    from collections import Counter
    winner_counts = Counter(r['winner'] for r in results_log)
    print()
    print("  Winner breakdown:")
    for model, count in winner_counts.most_common():
        print(f"    {model:30}  {count} series")

    # ==================================================
    # CERTIFICATION PIPELINE TEST
    # ==================================================
    print()
    print("  Testing certification pipeline...")

    all_records = sorted(results_log + errors_log, key=lambda r: r['sn'])

    # Test 1: JSON serialization (SafeEncoder)
    try:
        payload = json.dumps(all_records, cls=SafeEncoder, sort_keys=True, ensure_ascii=True)
        print("  ✓ JSON serialization (SafeEncoder)")
    except Exception as e:
        val_errors.append(f"JSON serialization failed: {e}")
        print(f"  ✕ JSON serialization FAILED: {e}")
        payload = "[]"

    # Test 2: SHA-256
    try:
        sha256 = hashlib.sha256(payload.encode()).hexdigest()
        assert len(sha256) == 64
        print(f"  ✓ SHA-256 hash: {sha256[:32]}...")
    except Exception as e:
        val_errors.append(f"SHA-256 failed: {e}")
        print(f"  ✕ SHA-256 FAILED: {e}")
        sha256 = ""

    # Test 3: Output file write
    out = {
        'test':        'smoke_test_10_series',
        'engine':      'Foresight Engine v3.0.0',
        'horizon':     FORECAST_H,
        'tier':        EVAL_TIER,
        'n_series':    len(results_log),
        'n_errors':    len(errors_log),
        'median_mase': float(np.median(mase_values)) if mase_values else None,
        'mean_mase':   float(np.mean(mase_values))   if mase_values else None,
        'sha256':      sha256,
        'elapsed_s':   round(elapsed_total, 1),
        'series':      results_log,
        'errors':      errors_log,
        'validation':  val_errors,
        'methodology': {
            'mase':    'Hyndman & Koehler (2006), IJF 22(4):679-688',
            'smape':   'Makridakis (1993)',
            'dataset': 'Makridakis & Hibon (2000), IJF 16(4):451-476',
        },
    }

    try:
        out_dir  = ROOT / 'diagnostics'
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / 'm3_smoke_test_results.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, cls=SafeEncoder, indent=2, ensure_ascii=True)
        print(f"  ✓ Output file written: {out_path}")
    except Exception as e:
        val_errors.append(f"File write failed: {e}")
        print(f"  ✕ File write FAILED: {e}")

    # ==================================================
    # FINAL PASS/FAIL
    # ==================================================
    print()
    print('=' * 64)

    checks = [
        ("0 engine failures",          len(errors_log) == 0),
        ("All records have valid MASE", all(r['mase'] is not None and r['mase'] > 0 for r in results_log)),
        ("Median MASE < 1.5",          bool(mase_values) and np.median(mase_values) < 1.5),
        ("SHA-256 generated",          len(sha256) == 64),
        ("0 structural errors",        len(val_errors) == 0),
    ]

    all_pass = True
    for check_name, passed in checks:
        mark = "✓" if passed else "✕"
        print(f"  {mark} {check_name}")
        if not passed:
            all_pass = False

    if val_errors:
        print()
        print("  Structural errors found:")
        for e in val_errors:
            print(f"    ✕ {e}")

    print()
    if all_pass:
        print("  ✓ SMOKE TEST PASSED — safe to run full 1,428 series.")
        print(f"    Estimated full run time: {(elapsed_total/len(smoke_series))*1428/3600:.1f} hours")
    else:
        print("  ✕ SMOKE TEST FAILED — fix issues before full run.")

    print('=' * 64)
    return all_pass


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)
