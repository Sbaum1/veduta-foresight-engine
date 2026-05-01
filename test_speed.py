"""
test_speed.py
Runs every model once on a small series and prints elapsed time.
Shows which models are slow so we can tune the M3 run.
Run from C:\Dev\VEDUTA\core\foresight_x:
    python test_speed.py
"""
import sys, time
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
from foresight_engine.registry import get_model_registry

dates = pd.date_range("1982-01-01", periods=60, freq="MS")
vals  = (1000 + np.arange(60) * 5
         + 200 * np.sin(2 * np.pi * np.arange(60) / 12))
df    = pd.DataFrame({"date": dates, "value": vals.astype(float)})

print(f"\n{'Model':<30s} {'Time':>8s}  {'Status'}")
print("-" * 55)

total = 0.0
slow  = []

for m in get_model_registry():
    name = m["name"]
    if name == "Primary Ensemble":
        continue
    t0 = time.time()
    try:
        m["runner"](df=df.copy(), horizon=18, confidence_level=0.90)
        elapsed = time.time() - t0
        total  += elapsed
        flag    = "  ⚠ SLOW" if elapsed > 5 else ""
        print(f"  {name:<28s} {elapsed:7.1f}s{flag}")
        if elapsed > 5:
            slow.append((name, round(elapsed, 1)))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {name:<28s} {elapsed:7.1f}s  FAIL: {str(e)[:45]}")

print("-" * 55)
print(f"  {'TOTAL':<28s} {total:7.1f}s")
print(f"\nSlow models (>5s per series):")
if slow:
    for name, t in slow:
        projected = round(t * 1428 / 3600, 1)
        print(f"  {name}: {t}s/series = ~{projected}h for full M3")
else:
    print("  None — all models under 5s")
print()
