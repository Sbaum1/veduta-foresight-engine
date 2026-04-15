import sys
sys.path.insert(0, '.')

try:
    from forecast_engine.contracts import ForecastInput, Frequency
    print('OK: forecast_engine.contracts')
except Exception as e:
    print('FAIL: forecast_engine.contracts --', e)

try:
    from veduta_foresight_app.charts_utils import get_foresight_engine
    engine = get_foresight_engine()
    print('OK: get_foresight_engine()')
except Exception as e:
    print('FAIL: get_foresight_engine --', e)

try:
    import foresight_engine
    names = [m['name'] for m in foresight_engine.get_model_registry()]
    print('OK: foresight_engine')
    print('  Models:', names)
except Exception as e:
    print('FAIL: foresight_engine --', e)
