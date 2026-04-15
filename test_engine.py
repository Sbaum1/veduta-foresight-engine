import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

# Build a simple 60-month series
dates = pd.date_range('2019-01-01', periods=60, freq='MS')
vals  = (100000 + np.arange(60) * 500
         + 8000 * np.sin(2 * np.pi * np.arange(60) / 12)
         + np.random.default_rng(42).normal(0, 2000, 60))
series = pd.Series(vals, index=dates, name='test')

# Build input
from forecast_engine.contracts import ForecastInput, Frequency
fi = ForecastInput(series_id='TEST', values=series, horizon=12, frequency=Frequency.MONTHLY)

# Run engine
from veduta_foresight_app.charts_utils import get_foresight_engine
engine = get_foresight_engine()
print('Running engine...')
result = engine.run(fi, run_id='TEST_001')

# Report what we got back
print('Result type:', type(result))
print('Result attrs:', [a for a in dir(result) if not a.startswith('_')])

if hasattr(result, 'rankings') and result.rankings:
    print('Rankings count:', len(result.rankings))
    print('Top model:', result.rankings[0].model_id)
    print('Top MASE:', result.rankings[0].mase)
    print('Ranking fields:', [a for a in dir(result.rankings[0]) if not a.startswith('_')])

if hasattr(result, 'point_forecast'):
    print('Forecast length:', len(result.point_forecast))
    print('Forecast mean:', round(float(sum(result.point_forecast)/len(result.point_forecast)), 2))

print('DONE')
