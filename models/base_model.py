import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings

warnings.filterwarnings('ignore')

class BaseModel:
    
    
    def __init__(self, model_path=None, **kwargs):
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def load_model(self, model_path=None):
        
            return False
    
    
    
    def feature_extraction(self, X):
        
        return X
    def feature_reduction(self, X):
        
        return X
    def data_cleaning(self, X):
        
        return X
    
    def normalization(self, X):
       
        return X
    
    def postprocess_output(self, predictions):
       
        return predictions
    
    def validate_input(self, X):
        
        return False
    def prediction(self, X, **kwargs):
        
        if not self.is_loaded:
            raise ValueError("Model not loaded! Use load_model() first.")
        
        print("Starting prediction pipeline...")
        
        # 1. Validate input
        if not self.validate_input(X):
            raise ValueError("Input validation failed!")
        
        # 2. Clean data
        X_cleaned = self.data_cleaning(X)
        
        # 3. Feature extraction
        X_features = self.feature_extraction(X_cleaned)
        
        # 3. Feature reduction
        X_reduced = self.feature_extraction(X_features)

        # 4. Normalization  
        X_normalized = self.normalization(X_reduced)
        
        
        # 6. Model prediction
        print("Making predictions...")
        raw_predictions = self.model.predict(X_normalized, **kwargs)
        
        # 7. Postprocess
        final_predictions = self.postprocess_output(raw_predictions)
        
        
        print("Prediction pipeline completed!")
        return final_predictions

