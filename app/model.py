import joblib
import sys
import os

# Add the root directory (where transformers_pipeline.py lives) to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure transformers_pipeline is importable
try:
    import transformers_pipeline
except ImportError as e:
    # If not found in parent, try same directory (just in case)
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    import transformers_pipeline

# Load the model artifact
# This will now be able to find transformers_pipeline because root_dir is in sys.path
artifact = joblib.load(os.path.join(root_dir, "model_artifact.joblib"))

pipeline = artifact["pipeline"]
model = artifact["model"]
feature_columns = artifact["feature_columns"]

# Manually mark each step in the pipeline as fitted to avoid NotFittedError
# This is necessary because some custom transformers don't set fitted attributes
for name, step in pipeline.steps:
    if not hasattr(step, "fitted_"):
        step.fitted_ = True
