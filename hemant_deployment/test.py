
import joblib
import xgboost as xgb
import os

# Ensure dill is installed
try:
    import dill
except ImportError:
    print("dill module is not installed. Please install it using 'pip install dill'")
    exit()

# Define the path to your model file
model_path = 'xgb_best_model_hemant.pkl'

# Check if the model file exists
if os.path.exists(model_path):
    print(f"Model file found: {model_path}")
    
    # Load the model
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        
        # Perform a simple test (e.g., predict using dummy data)
        test_data = [[1, 0.5]]  # Replace with appropriate feature values
        predictions = model.predict(test_data)
        print("Predictions:", predictions)
    
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found: {model_path}")
