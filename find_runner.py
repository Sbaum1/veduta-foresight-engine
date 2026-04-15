import sys
sys.path.insert(0, ".")
import foresight_engine.runner as r
print("runner.py loaded from:", r.__file__)

# Show the actual _validate_forward_boundary source
import inspect
src = inspect.getsource(r._validate_forward_boundary)
print("\n_validate_forward_boundary source:")
print(src)
