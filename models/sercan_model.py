# Sercan modeli
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis,skew
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .base_model import BaseModel



class SercanModel(BaseModel):
    

    def __init__(self, model_path=None, scaler_path=None, **kwargs):
        super().__init__(model_path, **kwargs)
        self.scaler_path = scaler_path
        self.scaler = None
        self.categorical_columns = [
    "Gender", "Fever", "Nausea/Vomting", "Headache",
    "Fatigue & generalized bone ache", "Jaundice",
    "Diarrhea", "Epigastric pain", "Baseline histological Grading"
]
        self._load_scaler()
        self._load_model()
    def _load_model(self):
        try:
            print(f"Sercan model uploading: {self.model_path}")
            
            if not self.model_path:
                raise ValueError("Path couldnt found!")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file couldnt found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            
            print(f"Sercan model upload sucesfully!")
            print(f"   Model tipi: {type(self.model).__name__}")
            
            if hasattr(self.model, 'n_estimators'):
                print(f" N estimators: {self.model.n_estimators}")
            if hasattr(self.model, 'max_depth'):
                print(f" Max depth: {self.model.max_depth}")
            
            return True
            
        except Exception as e:
            print(f"Sercan model upload error: {e}")
            self.model = None
            self.is_loaded = False
            return False
    def _load_scaler(self):
        
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print("Sercan scaler loaded successfully!")
            except Exception as e:
                print(f"Scaler loading failed: {e}")
                self.scaler = None
        else:
            print("Scaler path not found, will use default normalization")
    def feature_extraction(self, X):
     
        print("Feature extraction (training pipeline match)...")
        df = X.copy()
        
        print("Applying pd.get_dummies...")
        df = pd.get_dummies(df)
        print(f"After get_dummies: {df.shape}")
        
        try:
            alt_cols = ['ALT 1', 'ALT4', 'ALT 12', 'ALT 24', 'ALT 36', 'ALT 48']
            alt_weeks = [1, 4, 12, 24, 36, 48]

            if all(col in df.columns for col in alt_cols):
                print("Creating ALT temporal features (exact training match)...")
                
                df['ALT_diff_1_4'] = df['ALT4'] - df['ALT 1']
                df['ALT_diff_4_12'] = df['ALT 12'] - df['ALT4']
                df['ALT_diff_12_24'] = df['ALT 24'] - df['ALT 12']
                df['ALT_diff_24_48'] = df['ALT 48'] - df['ALT 24']
                df['ALT_diff_1_48'] = df['ALT 48'] - df['ALT 1']
                
                df['ALT_ratio_48_1'] = df['ALT 48'] / (df['ALT 1'] + 1e-5)
                
                df['ALT_mean'] = df[alt_cols].mean(axis=1)
                df['ALT_std'] = df[alt_cols].std(axis=1)
                df['ALT_min'] = df[alt_cols].min(axis=1)
                df['ALT_max'] = df[alt_cols].max(axis=1)
                df['ALT_range'] = df['ALT_max'] - df['ALT_min']
                df['ALT_skew'] = df[alt_cols].apply(skew, axis=1)
                df['ALT_kurtosis'] = df[alt_cols].apply(kurtosis, axis=1)
                
                def calc_trend(row, columns, weeks):
                    y = row[columns].values
                    X_array = np.array(weeks).reshape(-1, 1)
                    if np.isnan(y).any():
                        return np.nan
                    reg = LinearRegression().fit(X_array, y)
                    return reg.coef_[0]
                
                df['ALT_slope'] = df.apply(lambda row: calc_trend(row, alt_cols, alt_weeks), axis=1)
                
                print(f"ALT features created: {len([c for c in df.columns if 'ALT_' in c])}")
            else:
                print("Missing ALT columns, skipping ALT features")

            rna_cols = ['RNA Base', 'RNA 4', 'RNA 12', 'RNA EOT']
            rna_weeks = [0, 4, 12, 24]
            
            if all(col in df.columns for col in rna_cols):
                print("Creating RNA temporal features (exact training match)...")
                
                df['RNA_diff_4_base'] = df['RNA 4'] - df['RNA Base']
                df['RNA_diff_12_4'] = df['RNA 12'] - df['RNA 4']
                df['RNA_diff_EOT_12'] = df['RNA EOT'] - df['RNA 12']
                
                df['RNA_log_change_4'] = np.log1p(df['RNA 4']) - np.log1p(df['RNA Base'])
                df['RNA_log_change_12'] = np.log1p(df['RNA 12']) - np.log1p(df['RNA 4'])
                df['RNA_log_change_EOT'] = np.log1p(df['RNA EOT']) - np.log1p(df['RNA 12'])
                
                df['RNA_mean'] = df[rna_cols].mean(axis=1)
                df['RNA_std'] = df[rna_cols].std(axis=1)
                df['RNA_max'] = df[rna_cols].max(axis=1)
                df['RNA_min'] = df[rna_cols].min(axis=1)
                df['RNA_range'] = df['RNA_max'] - df['RNA_min']
                df['RNA_skew'] = df[rna_cols].apply(skew, axis=1)
                df['RNA_kurtosis'] = df[rna_cols].apply(kurtosis, axis=1)
                
                def calc_rna_trend(row, columns, weeks):
                    y = row[columns].values
                    X_array = np.array(weeks).reshape(-1, 1)
                    if np.isnan(y).any():
                        return np.nan
                    reg = LinearRegression().fit(X_array, y)
                    return reg.coef_[0]
                
                df['RNA_slope'] = df.apply(lambda row: calc_rna_trend(row, rna_cols, rna_weeks), axis=1)
                
                print(f"RNA features created: {len([c for c in df.columns if 'RNA_' in c])}")
            else:
                print("Missing RNA columns, skipping RNA features")
                
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        print(f"Feature extraction completed! Shape: {df.shape}")
        return df

    def _calc_trend(self, row, columns, weeks):
        
        try:
            available_data = []
            available_weeks = []
            
            for col, week in zip(columns, weeks):
                if col in row.index and not pd.isna(row[col]):
                    available_data.append(row[col])
                    available_weeks.append(week)
            
            if len(available_data) < 2:
                return np.nan
            
            y = np.array(available_data)
            X = np.array(available_weeks).reshape(-1, 1)
            
            reg = LinearRegression().fit(X, y)
            return reg.coef_[0]
            
        except Exception as e:
            return np.nan
        

    def feature_reduction(self, X):
        print("Feature reduction (exact training match)...")
    
        columns_to_drop = [
            'ALT 1', 'ALT 12', 'ALT 24', 'ALT 36', 'ALT 48', 
            'ALT after 24 w', 'ALT4', 'RNA 12', 'RNA 4', 
            'RNA Base', 'RNA EF', 'RNA EOT'
        ]
        
        existing_columns = [col for col in columns_to_drop if col in X.columns]
        
        if existing_columns:
            X.drop(columns=existing_columns, inplace=True)
            print(f"Dropped {len(existing_columns)} columns: {existing_columns}")
        
        print(f"After feature reduction: {X.shape}")
        return X
     
    def normalization(self, X):
        print("Normalization (column order fix)...")
        
        if self.scaler is not None:
            try:
                cleaned_columns = [col.strip() for col in X.columns]
                X.columns = cleaned_columns
                
                if hasattr(self.model, 'feature_names_in_'):
                    expected_order = list(self.model.feature_names_in_)
                    current_columns = list(X.columns)
                    
                    print(f"Current order: {current_columns[:3]}...")
                    print(f"Expected order: {expected_order[:3]}...")
                    
                    if set(current_columns) == set(expected_order):
                        print("All columns present, reordering...")
                        X = X[expected_order] 
                        print(f"Reordered to: {list(X.columns[:3])}...")
                    else:
                        print("Column sets don't match exactly")
                        available_expected = [col for col in expected_order if col in current_columns]
                        X = X[available_expected]
                        print(f"Using available columns: {len(available_expected)}/{len(expected_order)}")
                
                numeric_columns = X.columns.difference(self.categorical_columns).tolist()
                print(f"Numeric: {len(numeric_columns)}, Categorical: {len(self.categorical_columns)}")
                
                X_scaled = X.copy()
                if numeric_columns:
                    X_scaled[numeric_columns] = self.scaler.transform(X[numeric_columns])
                    print("Scaler applied!")
                
                return X_scaled
                
            except Exception as e:
                print(f"Error: {e}")
                return X
        else:
            print("No scaler!")
            return X

    def simple_reorder_columns(self, X):
        """Sadece sütun sıralaması düzelt"""
        if hasattr(self.model, 'feature_names_in_'):
            expected_order = list(self.model.feature_names_in_)
            
            if all(col in X.columns for col in expected_order):
                print(f"Reordering {len(expected_order)} columns to match model...")
                X_reordered = X[expected_order]
                print("Column order fixed!")
                return X_reordered
            else:
                print("Some expected columns missing")
                return X
        else:
            print("No expected column order available")
            return X

    

    

    def validate_input(self, X):
    
        print("Sercan's medical data validation...")

        if X is None or len(X) == 0:
            print("Dataset is empty or None!")
            return False
        original_columns = list(X.columns)
        cleaned_columns = [col.strip() for col in X.columns]
        expected_columns = [
            'Age', 'Gender', 'BMI', 'Fever', 'Nausea/Vomting', 'Headache', 
            'Diarrhea', 'Fatigue & generalized bone ache', 'Jaundice', 
            'Epigastric pain', 'WBC', 'RBC', 'HGB', 'Plat', 'AST 1', 
            'ALT 1', 'ALT4', 'ALT 12', 'ALT 24', 'ALT 36', 'ALT 48', 
            'ALT after 24 w', 'RNA Base', 'RNA 4', 'RNA 12', 'RNA EOT', 
            'RNA EF', 'Baseline histological Grading'
        ]

        missing_columns = [col for col in expected_columns if col not in cleaned_columns]

        if missing_columns:
            print(f"Missing columns are detected({len(missing_columns)}): {missing_columns}")
            return False

        print("All Features are available.")

        return True
    def data_cleaning(self, X):
        return super().data_cleaning(X)
    
    def prediction(self, X, **kwargs):
        if not self.is_loaded:
            raise ValueError("Model not loaded! Use load_model() first.")
        
        print("Sercan's Prediction Pipeline Starting...")

        if 'Baselinehistological staging' not in X.columns:
            raise ValueError("Target column 'Baselinehistological staging' not found in input!")
        self.target_column = "Baselinehistological staging"
        self.y = X['Baselinehistological staging'].apply(lambda x: 0 if x in [1, 2] else 1).values
        X = X.drop(columns=['Baselinehistological staging']) 

        if not self.validate_input(X):
            raise ValueError("Input validation failed!")

        X_cleaned = self.data_cleaning(X)
        X_features = self.feature_extraction(X_cleaned)
        X_reduced = self.feature_reduction(X_features)
        X_normalized = self.normalization(X_reduced)

        print("Making predictions...")
        raw_predictions = self.model.predict(X_normalized, **kwargs)
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