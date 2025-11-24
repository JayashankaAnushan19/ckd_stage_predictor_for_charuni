import joblib

MODEL_PATH = "../models/best_model.joblib"

# Load model
model = joblib.load(MODEL_PATH)

print("\n====================")
print("MODEL LOADED SUCCESSFULLY")
print("====================")

print("\n--- MODEL TYPE ---")
print(type(model))

print("\n--- MODEL DETAILS / PARAMETERS ---")
try:
    print(model.get_params())
except:
    print("This model does not support get_params()")

print("\n--- MODEL ATTRIBUTES ---")
for attr in dir(model):
    if not attr.startswith("_"):
        print(attr)

# If model is tree-based (RandomForest, XGBoost, etc.)
if hasattr(model, "estimators_"):
    print("\n--- TREE MODEL: NUMBER OF ESTIMATORS ---")
    print(len(model.estimators_))

# If model has feature importances
if hasattr(model, "feature_importances_"):
    print("\n--- FEATURE IMPORTANCES ---")
    print(model.feature_importances_)

print("\n--- MODEL STRING OUTPUT ---")
print(model)
