"""debug_m3.py - Run one M3 series, show full traceback"""
import sys, traceback
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

M3_PATH = r"C:\Dev\VEDUTA\_shared\sample_data\m3\m3_monthly_dataset.tsf"
HORIZON = 18

def load_one_series(filepath):
    with open(filepath, encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data or not line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            start_str = parts[1].strip()
            try:
                start = pd.Timestamp(f"{start_str}-01-01" if len(start_str)==4
                        else f"{start_str}-01" if len(start_str)==7
                        else start_str)
            except:
                start = pd.Timestamp("1982-01-01")
            vals = np.array([float(v.strip()) for v in parts[2].strip().split(",")
                             if v.strip()], dtype=np.float64)
            if len(vals) >= HORIZON + 24:
                return vals, start
    return None, None

vals, start = load_one_series(M3_PATH)
train_vals = vals[:-HORIZON]
dates = pd.date_range(start=start, periods=len(train_vals), freq="MS")
df    = pd.DataFrame({"date": dates, "value": train_vals})

print(f"Series: n={len(train_vals)}  mean={np.mean(train_vals):,.0f}")
print(f"Date dtype: {df['date'].dtype}")

# Patch runner to catch full traceback
import foresight_engine.runner as runner_mod
original_run = runner_mod.run_all_models

def patched_run(*args, **kwargs):
    try:
        return original_run(*args, **kwargs)
    except Exception as e:
        print(f"\nTOP-LEVEL EXCEPTION: {e}")
        traceback.print_exc()
        raise

runner_mod.run_all_models = patched_run

from foresight_engine.runner import run_all_models
raw = run_all_models(df=df, horizon=HORIZON, confidence_level=0.90)

ens = raw.get("Primary Ensemble", {})
print(f"\nPrimary Ensemble status: {ens.get('status')}")

if ens.get("status") == "failed":
    print(f"Error message: {ens.get('error','')}")
    # Try to get the raw exception by re-running just the ensemble part
    print("\n--- Investigating which line fails ---")
    
    # Find a success result to inspect its forecast_df date dtype
    for name, result in raw.items():
        if result.get("status") == "success" and not result.get("diagnostic_only"):
            fc_df = result.get("forecast_df")
            if fc_df is not None:
                print(f"\n{name} forecast_df date dtype: {fc_df['date'].dtype}")
                print(f"{name} actual dtype: {fc_df['actual'].dtype}")
                print(f"{name} actual sample: {fc_df['actual'].head(3).tolist()}")
                break
