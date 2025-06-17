# Sercan modeli
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis,skew
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .base_model import BaseModel
import pickle


class SercanModel(BaseModel):
    

    def __init__(self, model_path=None, scaler_path=None, **kwargs):
        super().__init__(model_path, **kwargs)
        self._load_model(model_path)
        
    def _load_model(self, folder_name="model_files"):
        """Kaydedilen modelleri yükle"""
        print(f"📂 Modeller yükleniyor... ({folder_name}/)")

        try:
            # Model 1
            with open(f"{folder_name}/model_1_main_classifier.pkl", 'rb') as f:
                self.model_blood_donor_vs_non_donor = pickle.load(f)
            print("Model 1 yüklendi")

            # Model 2A
            with open(f"{folder_name}/model_2A_donor_subclass.pkl", 'rb') as f:
                self.model_suspect_vs_true_donor = pickle.load(f)
            print("Model 2A yüklendi")

            # Model 2B
            with open(f"{folder_name}/model_2B_nondonor_subclass.pkl", 'rb') as f:
                self.model_non_donor_sub_classification = pickle.load(f)
            print("Model 2B yüklendi")

            # Label Encoder 2B
            with open(f"{folder_name}/label_encoder_2B.pkl", 'rb') as f:
                self.label_encoder_2B = pickle.load(f)
            print("Label Encoder 2B yüklendi")

            # Mapping
            with open(f"{folder_name}/mapping_2A.pkl", 'rb') as f:
                self.mapping_2A = pickle.load(f)
            print("Mapping 2A yüklendi")

            self.is_trained = True
            print("Tüm modeller başarıyla yüklendi!")
            self.is_loaded=True
            return True
        except FileNotFoundError as e:
            print(f"Dosya bulunamadı: {e}")
            return False
        except Exception as e:
            print(f"Yükleme hatası: {e}")
            return False
    
    def feature_extraction(self, X):
     
        return X

    def feature_reduction(self, X):
        print("Feature reduction (exact training match)...")
        X=X[['ALB', 'CHE', 'ALP', 'Age', 'BIL', 'ALT', 'CHOL', 'PROT','AST','GGT']]
        
        print(f"After feature reduction: {X.shape}")
        return X
    def filter_outliers(self,X):
        outlier_bounds = {
        'Age': {'lower': 16.5000, 'upper': 76.5000},
        'ALB': {'lower': 29.2000, 'upper': 54.8000},
        'ALP': {'lower': 9.9750, 'upper': 122.5750},
        'ALT': {'lower': -8.5750, 'upper': 58.0250},
        'AST': {'lower': 4.6500, 'upper': 49.8500},
        'BIL': {'lower': -3.5500, 'upper': 20.0500},
        'CHE': {'lower': 2.9525, 'upper': 13.5725},
        'CHOL': {'lower': 2.4300, 'upper': 8.2300},
        'CREA': {'lower': 35.5000, 'upper': 119.5000},
        'GGT': {'lower': -21.0500, 'upper': 76.9500},
        'PROT': {'lower': 60.1500, 'upper': 84.5500}
    }
    
        print("Outlier filtreleme başlıyor...")
        X_filtered = X.copy()
        
        # Her feature için winsorization uygula
        for feature, bounds in outlier_bounds.items():
            if feature in X_filtered.columns:
                lower = bounds['lower']
                upper = bounds['upper']
                
                # Outlier sayısını say (isteğe bağlı)
                outliers_below = (X_filtered[feature] < lower).sum()
                outliers_above = (X_filtered[feature] > upper).sum()
                total_outliers = outliers_below + outliers_above
                
                if total_outliers > 0:
                    print(f"  {feature}: {total_outliers} outlier bulundu "
                        f"(alt: {outliers_below}, üst: {outliers_above})")
                
                # Winsorization uygula
                X_filtered[feature] = np.where(
                    X_filtered[feature] < lower, 
                    lower, 
                    X_filtered[feature]
                )
                X_filtered[feature] = np.where(
                    X_filtered[feature] > upper, 
                    upper, 
                    X_filtered[feature]
                )
            else:
                print(f"  ⚠️ {feature} sütunu bulunamadı, atlanıyor...")
        
        print(f"Outlier filtreleme tamamlandı. Shape: {X_filtered.shape}")
        return X_filtered

    def normalization(self, X):
            return X


    def validate_input(self, X):
    
        print("Sercan's medical data validation...")

        if X is None or len(X) == 0:
            print("Dataset is empty or None!")
            return False
        original_columns = list(X.columns)
        cleaned_columns = [col.strip() for col in X.columns]
        expected_columns = ['ALB', 'CHE', 'ALP', 'Age', 'BIL', 'ALT', 'CHOL', 'PROT','AST','GGT']

        missing_columns = [col for col in expected_columns if col not in cleaned_columns]

        if missing_columns:
            print(f"Missing columns are detected({len(missing_columns)}): {missing_columns}")
            return False

        print("All Features are available.")

        return True
    def data_cleaning(self, X):
        return super().data_cleaning(X)
    
    def predict(self, X):
        
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmedi. Lütfen önce `fit` metodunu çağırın.")

        import pandas as pd
        if isinstance(X, pd.DataFrame): X = X.values

        print(f"🔍 Debug: Tahmin başlıyor - {X.shape[0]} örnek")

        # 1. Aşama
        level1_preds = self.model_blood_donor_vs_non_donor.predict(X)
        final_predictions = np.zeros_like(level1_preds, dtype=int)

        print(f"Level 1 tahminleri: {np.bincount(level1_preds)}")

        # 2. Aşama
        donor_mask = (level1_preds == 0)
        non_donor_mask = (level1_preds == 1)

        if np.any(donor_mask):
            preds_2A = self.model_suspect_vs_true_donor.predict(X[donor_mask])
            final_predictions[donor_mask] = np.vectorize(self.mapping_2A.get)(preds_2A)
            print(f"Model 2A tahminleri: {np.bincount(preds_2A)} -> Final: {np.bincount(final_predictions[donor_mask])}")

        if np.any(non_donor_mask):
            preds_2B_encoded = self.model_non_donor_sub_classification.predict(X[non_donor_mask])
            preds_2B_decoded = self.label_encoder_2B.inverse_transform(preds_2B_encoded)
            final_predictions[non_donor_mask] = preds_2B_decoded

            print(f"Model 2B encoded tahminleri: {np.bincount(preds_2B_encoded)}")
            print(f"Model 2B decoded tahminleri: {np.bincount(preds_2B_decoded)}")
            print(f"LabelEncoder classes_: {self.label_encoder_2B.classes_}")
            print(f"Mapping: {dict(zip(range(len(self.label_encoder_2B.classes_)), self.label_encoder_2B.classes_))}")

        print(f"Final tahmin dağılımı: {np.bincount(final_predictions)}")

        return final_predictions
    def prediction(self, X, **kwargs):
        if not self.is_loaded:
            raise ValueError("Model not loaded! Use load_model() first.")
        
        print("Sercan's Prediction Pipeline Starting...")

        if 'Category'  in X.columns:
            
            self.target_column = "Category"
            self.y = X['Category']
            X = X.drop(columns=['Category']) 

        if not self.validate_input(X):
            raise ValueError("Input validation failed!")

        X_cleaned = self.data_cleaning(X)
        X_filtered = self.filter_outliers(X_cleaned)  # BURAYA EKLEYİN
        X_features = self.feature_extraction(X_cleaned)
        
        X_reduced = self.feature_reduction(X_features)
        X_normalized = self.normalization(X_reduced)

        print("Making predictions...")
        raw_predictions = self.predict(X_normalized)
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
                # String etiketleri integer'a dönüştür
                label_mapping = {
                    '0=Blood Donor': 0,
                    '0s=suspect Blood Donor': 4,
                    '1=Hepatitis': 1,
                    '2=Fibrosis': 2,
                    '3=Cirrhosis': 3
                }
                
                # self.y'yi DEĞİŞTİRME! Kopya oluştur
                y_true_encoded = []
                for label in self.y:
                    if label in label_mapping:
                        y_true_encoded.append(label_mapping[label])
                    else:
                        print(f"Bilinmeyen etiket: {label}")
                        y_true_encoded.append(-1)
                
                y_true_encoded = np.array(y_true_encoded)
                
                # Sadece geçerli etiketleri değerlendir
                valid_mask = y_true_encoded != -1
                if np.any(valid_mask):
                    y_true_valid = y_true_encoded[valid_mask]
                    final_preds_valid = final_preds[valid_mask]
                    
                    accuracy = accuracy_score(y_true_valid, final_preds_valid)
                    
                    print("=== POSTPROCESS_OUTPUT DEBUG ===")
                    print(f"self.y tipi: {type(self.y)}")
                    print(f"self.y ilk 3: {self.y[:3]}")
                    print(f"y_true_encoded: {y_true_encoded}")
                    print(f"final_preds: {final_preds}")
                    print(f"Accuracy hesaplandı: {accuracy}")
                    
                    self.last_accuracy = accuracy
                    
                    print("Evaluation Results (from postprocess_output):")
                    print("Accuracy:", accuracy)
                    print("Confusion Matrix:\n", confusion_matrix(y_true_valid, final_preds_valid))
                    print("Classification Report:\n", classification_report(y_true_valid, final_preds_valid))
                else:
                    print("Geçerli etiket bulunamadı!")
                    self.last_accuracy = None
                    
            except Exception as e:
                print(f"Evaluation error in postprocess_output: {e}")
                self.last_accuracy = None
        
        return final_preds