import joblib
import sys
import os
from sklearn.utils.validation import check_is_fitted

# Add root to path
sys.path.insert(0, ".")
import transformers_pipeline

print("Loading artifact...")
artifact = joblib.load("model_artifact.joblib")
pipeline = artifact['pipeline']

print("\nChecking pipeline fit state:")
try:
    check_is_fitted(pipeline)
    print("✓ Pipeline itself is considered fitted.")
except Exception as e:
    print(f"✗ Pipeline is NOT fitted: {e}")

print("\nChecking individual steps:")
for i, (name, step) in enumerate(pipeline.steps):
    # Apply fix
    if not hasattr(step, "fitted_"):
        step.fitted_ = True
        
    print(f"Step {i}: {name} ({type(step).__name__})")
    # Check for common fitted attributes
    fitted_attrs = [a for a in dir(step) if a.endswith("_") and not a.startswith("__")]
    if fitted_attrs:
        print(f"  - Fitted attributes: {fitted_attrs}")
    else:
        print("  - No fitted attributes found.")
    
    try:
        check_is_fitted(step)
        print("  - check_is_fitted: PASSED")
    except Exception as e:
        print(f"  - check_is_fitted: FAILED ({e})")
