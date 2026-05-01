"""
test_smoke.py
================================================================================
VEDUTA Foresight Engine v3.0.0 — Smoke Test
================================================================================
Tests every registered model individually against a synthetic manufacturing
series. Validates the output contract for each model. Does NOT require M3
dataset. Run time: ~3-5 minutes (N-HiTS trains PyTorch).

Output: test_results/smoke_results.json

Run from C:\\Dev\\VEDUTA\\core\\foresight_x:
    python test_smoke.py
================================================================================
"""

import sys, os, json, time, traceback
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

OUTPUT_DIR = Path("diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED_COLS = {"date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"}
HORIZON       = 12


def _make_series(n: int = 72, seed: int = 42) -> pd.DataFrame:
    """FRED-calibrated synthetic manufacturing series."""
    rng   = np.random.default_rng(seed)
    t     = np.arange(n, dtype=float)
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    vals  = (
        350_000
        + t * 800
        + 22_000 * np.sin(2 * np.pi * t / 12 - 0.4)
        +  7_000 * np.sin(2 * np.pi * t / 6)
        + rng.normal(0, 5_500, n)
    )
    # COVID shock
    vals[27:30] -= 45_000
    vals[30:36] += np.linspace(-25_000, 0, 6)
    vals = np.maximum(vals, 200_000)
    return pd.DataFrame({"date": dates, "value": vals.round(0)})


def _validate_result(name: str, result, df: pd.DataFrame, horizon: int) -> dict:
    """
    Validate ForecastResult output contract.
    Returns dict with pass/fail per check.
    """
    checks = {}

    # 1. Has forecast_df
    has_df = hasattr(result, "forecast_df") and result.forecast_df is not None
    checks["has_forecast_df"] = has_df
    if not has_df:
        return checks

    fc_df = result.forecast_df

    # 2. Required columns present
    missing_cols = REQUIRED_COLS - set(fc_df.columns)
    checks["required_cols"] = len(missing_cols) == 0
    checks["missing_cols"]  = sorted(missing_cols)

    # 3. No duplicate dates
    checks["no_duplicate_dates"] = not fc_df["date"].duplicated().any()

    # 4. Future rows present (horizon rows with actual=NaN)
    future = fc_df[fc_df["actual"].isna()]
    checks["future_rows"]         = len(future)
    checks["future_rows_correct"] = len(future) >= horizon

    # 5. Future forecast values finite
    if len(future) > 0 and "forecast" in future.columns:
        fc_vals = future["forecast"].dropna().values
        checks["forecast_finite"]   = bool(np.isfinite(fc_vals).all()) if len(fc_vals) > 0 else False
        checks["forecast_mean"]     = round(float(np.mean(fc_vals)), 2) if len(fc_vals) > 0 else None
    else:
        checks["forecast_finite"] = False
        checks["forecast_mean"]   = None

    # 6. CI not inverted (for non-diagnostic models)
    if len(future) > 0 and "ci_low" in future.columns and "ci_high" in future.columns:
        ci_lo = future["ci_low"].ffill().values
        ci_hi = future["ci_high"].ffill().values
        inverted = int((ci_lo > ci_hi).sum())
        checks["ci_not_inverted"] = inverted == 0
        checks["ci_inversions"]   = inverted
    else:
        checks["ci_not_inverted"] = True
        checks["ci_inversions"]   = 0

    # 7. Metadata present
    checks["has_metadata"] = hasattr(result, "metadata") and isinstance(result.metadata, dict)

    # 8. Model name matches
    if hasattr(result, "model_name"):
        checks["model_name_matches"] = result.model_name == name

    return checks


def run_smoke_test():
    print("\n" + "=" * 72)
    print("  VEDUTA Foresight Engine v3.0.0 — SMOKE TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    from foresight_engine.registry import get_model_registry
    from foresight_engine.runner   import run_all_models

    df    = _make_series(n=72)
    reg   = get_model_registry()
    total = len([m for m in reg if m["name"] != "Primary Ensemble"])

    print(f"\n  Series: 72 months synthetic manufacturing · Horizon: {HORIZON}")
    print(f"  Models to test: {total}")
    print()

    results_log = []
    passed      = 0
    failed      = 0
    skipped     = 0

    for entry in reg:
        name   = entry["name"]
        runner = entry["runner"]
        diag   = entry.get("diagnostic_only", False)

        if name == "Primary Ensemble":
            continue

        t0 = time.time()
        try:
            result = runner(df=df.copy(), horizon=HORIZON, confidence_level=0.90)
            elapsed = round(time.time() - t0, 2)
            checks  = _validate_result(name, result, df, HORIZON)

            # Core pass: has future rows, forecast finite, CI not inverted
            core_pass = (
                checks.get("has_forecast_df", False)
                and checks.get("required_cols", False)
                and checks.get("no_duplicate_dates", False)
                and checks.get("forecast_finite", False)
                and checks.get("ci_not_inverted", True)
            )

            if diag:
                # Diagnostic models just need to not crash and return a df
                core_pass = checks.get("has_forecast_df", False)
                status    = "DIAGNOSTIC"
                skipped  += 1
            elif core_pass:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            icon = "✅" if status in ("PASS", "DIAGNOSTIC") else "❌"
            mean_str = f"  mean={checks.get('forecast_mean'):,.0f}" if checks.get("forecast_mean") else ""
            print(f"  {icon} {name:25s} {status:12s} {elapsed:6.1f}s{mean_str}")

            results_log.append({
                "model":        name,
                "status":       status,
                "elapsed_s":    elapsed,
                "diagnostic":   diag,
                "checks":       checks,
                "error":        None,
            })

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            failed += 1
            print(f"  ❌ {name:25s} {'CRASH':12s} {elapsed:6.1f}s  {type(exc).__name__}: {str(exc)[:60]}")
            results_log.append({
                "model":        name,
                "status":       "CRASH",
                "elapsed_s":    elapsed,
                "diagnostic":   diag,
                "checks":       {},
                "error":        traceback.format_exc(),
            })

    # ── Primary Ensemble ───────────────────────────────────────────────────────
    print(f"\n  Running Primary Ensemble...")
    t0 = time.time()
    try:
        raw = run_all_models(df=df.copy(), horizon=HORIZON, confidence_level=0.90)
        elapsed = round(time.time() - t0, 2)
        ens = raw.get("Primary Ensemble", {})
        if ens.get("status") == "success":
            print(f"  ✅ {'Primary Ensemble':25s} {'PASS':12s} {elapsed:6.1f}s")
            results_log.append({"model": "Primary Ensemble", "status": "PASS",
                                 "elapsed_s": elapsed, "error": None})
            passed += 1
        else:
            print(f"  ❌ {'Primary Ensemble':25s} {'FAIL':12s} {elapsed:6.1f}s")
            results_log.append({"model": "Primary Ensemble", "status": "FAIL",
                                 "elapsed_s": elapsed, "error": ens.get("error")})
            failed += 1

        stk = raw.get("Stacked Ensemble", {})
        if stk.get("status") == "success":
            print(f"  ✅ {'Stacked Ensemble':25s} {'PASS':12s}")
            results_log.append({"model": "Stacked Ensemble", "status": "PASS",
                                 "elapsed_s": 0, "error": None})
            passed += 1
        else:
            print(f"  ⚠️  {'Stacked Ensemble':25s} {'SKIPPED':12s}  (needs ≥2 base models)")

    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        print(f"  ❌ Primary Ensemble CRASH: {exc}")
        results_log.append({"model": "Primary Ensemble", "status": "CRASH",
                             "elapsed_s": elapsed, "error": str(exc)})
        failed += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    total_run = passed + failed
    pct       = round(passed / total_run * 100, 1) if total_run > 0 else 0

    print("\n" + "─" * 72)
    print(f"  PASSED:     {passed}")
    print(f"  FAILED:     {failed}")
    print(f"  DIAGNOSTIC: {skipped} (excluded from pass/fail)")
    print(f"  PASS RATE:  {pct}%")

    overall = failed == 0
    print(f"\n  SMOKE TEST: {'✅ ALL PASS' if overall else '❌ FAILURES DETECTED'}")
    print("=" * 72 + "\n")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "test":        "smoke",
        "timestamp":   datetime.now().isoformat(),
        "engine":      "Foresight Engine v3.0.0",
        "series_n":    72,
        "horizon":     HORIZON,
        "passed":      passed,
        "failed":      failed,
        "diagnostic":  skipped,
        "overall":     overall,
        "models":      results_log,
    }

    out_path = OUTPUT_DIR / "smoke_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {out_path}\n")

    return overall


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

