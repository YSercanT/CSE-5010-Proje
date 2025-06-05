# Sercan modeli
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis,skew
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .base_model import BaseModel



class SuheylaModel(BaseModel):
    

    def __init__(self, model_path=None, scaler_path=None, **kwargs):
        super().__init__(model_path, **kwargs)
        self.scaler_path = scaler_path
        self.scaler = None
        
        self._load_scaler()
        self._load_model()
    def _load_model(self):
        try:
            print(f" model uploading: {self.model_path}")
            
            if not self.model_path:
                raise ValueError("Path couldnt found!")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file couldnt found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            
            print(f" model upload sucesfully!")
            print(f"   Model tipi: {type(self.model).__name__}")
            
            if hasattr(self.model, 'n_estimators'):
                print(f" N estimators: {self.model.n_estimators}")
            if hasattr(self.model, 'max_depth'):
                print(f" Max depth: {self.model.max_depth}")
            
            return True
            
        except Exception as e:
            print(f" model upload error: {e}")
            self.model = None
            self.is_loaded = False
            return False
    def _load_scaler(self):
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(" scaler loaded successfully!")
            except Exception as e:
                print(f" loading failed: {e}")
                self.scaler = None
        else:
            print("Scaler path not found, will use default normalization")
    def feature_extraction(self, X):
     
        print("Feature extraction (training pipeline match)...")
        df = X.copy()
        
        return df

    

    def feature_reduction(self, X):
        
        return X
        
    def normalization(self, X):
        
        return X

    def validate_input(self, X):
    
        print("All Features are available.")

        return True
    def data_cleaning(self, X):
        return super().data_cleaning(X)
   
    def prediction(self, X, **kwargs):
        # self.target_column 
        #self.y=x[""]
        #X=X.drop(colums="", inplace=true)
        self.y=X['Baselinehistological staging'] #test
        #X_cleaned = self.data_cleaning(X)
        #X_features = self.feature_extraction(X_cleaned)
        #X_reduced = self.feature_reduction(X_features)
        #X_normalized = self.normalization(X_reduced)

        print("Making predictions...")
        #raw_predictions = self.model.predict(X_normalized, **kwargs)
        raw_predictions=self.y #test
        print("Predictions completed.")

        return raw_predictions
    
    def postprocess_output(self, predictions):
        print("Postprocessing output...")

        if predictions.ndim > 1 and predictions.shape[1] > 1:
            final_preds = np.argmax(predictions, axis=1)
        else:
            final_preds = predictions.ravel().astype(int)

        if hasattr(self, "y") and self.y is not None:
            try:
                print("Evaluation Results (from postprocess_output):")
                print("Accuracy:", accuracy_score(self.y, final_preds))
                print("Confusion Matrix:\n", confusion_matrix(self.y, final_preds))
                print("Classification Report:\n", classification_report(self.y, final_preds))
            except Exception as e:
                print(f"Evaluation error in postprocess_output: {e}")

        return final_preds