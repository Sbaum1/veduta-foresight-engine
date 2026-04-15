import sys
sys.path.insert(0, '.')

try:
    from foresight_engine.registry import get_model_registry
    names = [m['name'] for m in get_model_registry()]
    print('OK: registry —', len(names), 'models')
    print('  ', names)
except Exception as e:
    print('FAIL: registry --', e)

try:
    from veduta_foresight_app.charts_utils import get_foresight_engine
    engine = get_foresight_engine()
    print('OK: engine adapter')
except Exception as e:
    print('FAIL: engine adapter --', e)
