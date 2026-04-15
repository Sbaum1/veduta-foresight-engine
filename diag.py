import sys
sys.path.insert(0, '.')
from sentinel_engine.runner import run_all_models
import pandas as pd, numpy as np

dates = pd.date_range('2016-01-01', periods=96, freq='MS')
vals = 248000 + np.arange(96)*620 + 18500*np.sin(2*3.14159*np.arange(96)/12)
df = pd.DataFrame({'date': dates, 'value': vals})

results = run_all_models(df, horizon=12, confidence_level=0.90)
for name, r in results.items():
    if name.startswith('_'):
        continue
    if r.get('status') == 'failed':
        etype = r.get('error_type', 'unknown')
        emsg  = str(r.get('error_message', 'none'))[:120]
        print(name + ': ' + etype + ' -- ' + emsg)
