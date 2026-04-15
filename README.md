# VEDUTA Foresight X
## Executive Forecasting Intelligence Platform
### Foresight Engine v3.0.0

**Brand:** Nocturne Black · Venetian Gold · Canal Teal  
**Tagline:** The veduta is clear.

---

## Setup (Windows PowerShell)

```powershell
cd C:\Dev\Repos\Veduta_Foresight_X
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install prophet
pip install arch
pip install "scikit-learn<1.6"
```

## Run

```powershell
cd veduta_foresight_x
streamlit run veduta_foresight_app\app.py
```

## Architecture

```
veduta_foresight_x/
├── foresight_engine/          # 14-model forecasting engine
│   ├── __init__.py            # ForesightEngine orchestrator
│   ├── contracts.py           # Typed data contracts
│   ├── registry.py            # Model registry (14 models)
│   ├── runner.py              # Parallel model execution
│   ├── ensemble.py            # MASE-weighted ensemble
│   ├── stacker.py             # Model stacking layer
│   ├── certifier.py           # Certification framework
│   ├── preprocessor.py        # Structural break detection
│   ├── backtest.py            # Walk-forward backtesting
│   ├── foresight_config.py    # Engine configuration
│   └── models/                # 14 individual model implementations
│       ├── arima.py, sarima.py, sarimax.py
│       ├── ets.py, hw_damped.py, stl_ets.py, mstl.py
│       ├── dhr.py, local_linear_trend.py
│       ├── prophet.py, nnetar.py, lightgbm_model.py
│       ├── theta.py, naive.py, ses.py, croston.py
│       └── tbats.py, var_model.py, garch_model.py, x13.py
└── veduta_foresight_app/      # Streamlit UI
    ├── app.py                 # Main application
    ├── styles.py              # CSS design tokens
    ├── charts_utils.py        # Plotly chart functions
    └── .streamlit/config.toml # Venetian Gold native theme

```

## Notes

- Upload CSV with `Date` column + numeric value columns (monthly data)
- Minimum 24 observations for seasonal models
- Nov 2020 spike in manufacturing data is a known outlier (COVID rebound) — Tier 4 MASE on that dataset is expected
- FRED API integration in Sentinella/Meridiano novellas requires free API key from fred.stlouisfed.org

## Engine: Foresight Engine v3.0.0

14 models: ARIMA, SARIMA, SARIMAX, ETS, HW_Damped, STL+ETS, MSTL, DHR, LocalLinearTrend, Prophet, NNETAR, LightGBM, Theta, Naive

python -c "
from foresight_engine.registry import get_model_registry
registry = get_model_registry()

print('{:<35} {:<14} {:<10} {:<10} {}'.format('Model','MinTier','Ensemble','DiagOnly','Status'))
print('-' * 85)
for cfg in sorted(registry, key=lambda x: x['name']):
    name      = cfg.get('name', '?')
    tier      = cfg.get('min_tier', '?')
    ensemble  = cfg.get('ensemble_member', False)
    diag_only = cfg.get('diagnostic_only', False)
    active    = tier in ('essentials', 'core')
    status    = 'DIAG' if diag_only else ('ACTIVE' if active else 'SKIP')
    print('{:<35} {:<14} {:<10} {:<10} {}'.format(name, tier, str(ensemble), str(diag_only), status))

essentials = [c for c in registry if c.get('min_tier') in ('essentials','core') and not c.get('diagnostic_only')]
print()
print('Essentials-tier active models: {}'.format(len(essentials)))
"
