"""
test_chronos2_api.py
Run this on your Windows machine to discover the exact Chronos-2 API.
"""
import inspect
import sys

print(f"Python: {sys.version}")

try:
    import chronos
    print(f"chronos version: {getattr(chronos, '__version__', 'unknown')}")
    print(f"chronos location: {chronos.__file__}")
    
    from chronos import Chronos2Pipeline
    print("\nChronos2Pipeline.predict signature:")
    sig = inspect.signature(Chronos2Pipeline.predict)
    for pname, param in sig.parameters.items():
        ann = param.annotation if param.annotation != inspect.Parameter.empty else "any"
        default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        print(f"  {pname}: type={ann}  default={default}")
    
    print("\nChronos2Pipeline public methods:")
    for m in dir(Chronos2Pipeline):
        if not m.startswith('_'):
            print(f"  {m}")
    
    # Also check if predict_quantiles exists
    if hasattr(Chronos2Pipeline, 'predict_quantiles'):
        print("\npredict_quantiles signature:")
        sig2 = inspect.signature(Chronos2Pipeline.predict_quantiles)
        for pname, param in sig2.parameters.items():
            print(f"  {pname}: default={param.default}")

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
