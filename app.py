from flask import Flask, request, jsonify,render_template
import pandas as pd
import os
from flask_cors import CORS      
from models.sercan_model import SercanModel
from models.atakan_model import AtakanModel
from models.suheyla_model import SuheylaModel
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
CORS(app)


UPLOAD_FOLDER = 'uploads'
LAST_FILE_PATH = os.path.join(UPLOAD_FOLDER, 'last_uploaded.csv') 
MODELS_FILE_PATH="model_files"
SCALERS_FILE_PATH="scaler_files"
sercan_model_instance = None
suheyla_model_instance = None
atakan_model_instance = None

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(LAST_FILE_PATH)  
        return jsonify({"status": "ok", "filename": file.filename})
    else:
        return jsonify({"status": "error", "message": "File  couldnt find!"}), 400

@app.route('/last-data', methods=['GET'])
def get_last_data():
    if os.path.exists(LAST_FILE_PATH):
        df = pd.read_csv(LAST_FILE_PATH)
        return df.to_json(orient='records') 
    else:
        return jsonify([])  

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        selected_model = data.get('selected_model')
        
        print(f"{selected_model} model choosed!")
        
        if selected_model == 'sercan':
            results = run_sercan_model()  
        elif selected_model == 'suheyla':
            results = run_suheyla_model()
        elif selected_model == 'atakan':
            results = run_atakan_model()  
        else:
            return {'error': 'Unknown model'}, 400
        
        return {
            'message': f'{selected_model} model successful',
            'results': results
        }, 200
        
    except Exception as e:
        return {'error': str(e)}, 500

def initialize_sercan_model():
    global sercan_model_instance
    try:
        sercan_model_instance = SercanModel(
            model_path=MODELS_FILE_PATH,  
            scaler_path=os.path.join(SCALERS_FILE_PATH, "min_max_scaler_sercan.pkl")   
       
        )
        scaler_path = os.path.join(SCALERS_FILE_PATH, "min_max_scaler_sercan.pkl")
        model_path = os.path.join(MODELS_FILE_PATH, "rf.pkl")
        print(f"Model path: {model_path}")
        print(f"Scaler path: {scaler_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        print(f"Scaler exists: {os.path.exists(scaler_path)}")    
        sercan_model_instance.load_model()
        print("Sercan model instance oluşturuldu!")
        return True
    except Exception as e:
        print(f"Sercan model başlatma hatası: {e}")
        return False
def initialize_atakan_model():
    global atakan_model_instance
    try:
        atakan_model_instance = AtakanModel(
            model_path=os.path.join(MODELS_FILE_PATH, "mlp_model.pkl"), #model yolu 
            scaler_path=os.path.join(SCALERS_FILE_PATH, "atakan_scaler.pkl"), #normalizasyon yolu  
       encoder_path=os.path.join(MODELS_FILE_PATH, "label_encoder.pkl") # Label yolu
        )
       
        atakan_model_instance.load_model()
        print("Atakan model instance oluşturuldu!")
        return True
    except Exception as e:
        print(f"Atakan model başlatma hatası: {e}")
        return False
        
def initialize_suheyla_model():
    global suheyla_model_instance
    try:
        suheyla_model_instance = SuheylaModel(
            model_path=os.path.join(MODELS_FILE_PATH, "HCV600_svm_model.pkl"),  #model yolu
            scaler_path=os.path.join(SCALERS_FILE_PATH, "HCV600_scaler.pkl") # normalizasyon yolu  
       
        )
           
        suheyla_model_instance.load_model()
        return True
    except Exception as e:
        print(f"Süheyla model başlatma hatası: {e}")
        return False

def run_sercan_model():
    global sercan_model_instance
    try:
        if sercan_model_instance is None:
            print("Creating Sercan Model instance...")
            if not initialize_sercan_model():
                raise Exception("Sercan model couldnt start")
        
        if not os.path.exists(LAST_FILE_PATH):
            raise Exception("Please upload file first")
        
        print("Data Loading")
        df = pd.read_csv(LAST_FILE_PATH)
        print(f"Data Shape: {df.shape}")
        
        print("Making Prediction")
        predictions = sercan_model_instance.prediction(df.copy())
        
        print("Postprocessing")
        final_predictions = sercan_model_instance.postprocess_output(predictions)
        
        print("=== RUN_SERCAN_MODEL DEBUG ===")
        print(f"sercan_model_instance.y: {getattr(sercan_model_instance, 'y', 'YOK')}")
        print(f"final_predictions: {final_predictions}")
        print(f"last_accuracy: {getattr(sercan_model_instance, 'last_accuracy', 'YOK')}")
        
        results = format_results(df, final_predictions, sercan_model_instance)
        
        final_accuracy = results.get('accuracy', 'N/A')
        print(f"Sercan modeli build successfully! Accuracy: {final_accuracy}")
        
        return results
        
    except Exception as e:
        print(f"Sercan model error: {e}")
        return create_error_result(str(e))

def run_suheyla_model():
    global suheyla_model_instance
    
    try:
        
        if suheyla_model_instance is None:
            print("Creating Süheyla Model instance...")
            if not initialize_suheyla_model():
                raise Exception("Süheyla model couldnt start")
        
        if not os.path.exists(LAST_FILE_PATH):
            raise Exception("VPlease upload file first")
        
        print("Data Loading")
        df = pd.read_csv(LAST_FILE_PATH)
        print(f"Data Shape: {df.shape}")
        
        print("Making Prediction")
        predictions = suheyla_model_instance.prediction(df.copy())
        
        print("Postprocessing")
        final_predictions = suheyla_model_instance.postprocess_output(predictions)
        
        results = format_results(df, final_predictions, suheyla_model_instance)
        print("df")
        print(df)
        print(f"final : {final_predictions}")
        
        print(f"Süheyla modeli build successfully! Accuracy: {results.get('accuracy', 'N/A')}")
        return results
        
    except Exception as e:
        print(f"Süheyla model error: {e}")
        return create_error_result(str(e))
   
def run_atakan_model():
    global atakan_model_instance
    
    try:
        
        if atakan_model_instance is None:
            print("Creating Atakan Model instance...")
            if not initialize_atakan_model():
                raise Exception("Atakan model couldnt start")
        
        if not os.path.exists(LAST_FILE_PATH):
            raise Exception("VPlease upload file first")
        
        print("Data Loading")
        df = pd.read_csv(LAST_FILE_PATH)
        print(f"Data Shape: {df.shape}")
        
        print("Making Prediction")
        predictions = atakan_model_instance.prediction(df.copy())
        
        print("Postprocessing")
        final_predictions = atakan_model_instance.postprocess_output(predictions)
        
        results = format_results(df, final_predictions, atakan_model_instance)
        
        print(f"Atakan modeli build successfully! Accuracy: {results.get('accuracy', 'N/A')}")
        return results
        
    except Exception as e:
        print(f"Atakan model error: {e}")
        return create_error_result(str(e))
   
def create_error_result(error_message):
    return {
        'accuracy': 0.0,
        'total_samples': 0,
        'class_distribution': {'error': 1},
        'target_column': 'Error',
        'description': f'Sercan Model Error: {error_message}',
        'detailed_results': [],
        'predictions': [],
        'error': error_message
    }

def format_results(df, y_pred, model_instance):
    try:
        print("=== FORMAT_RESULTS BAŞLADI ===")
        
        y_true = getattr(model_instance, "y", None)
        if y_true is None:
            raise Exception("True y couldn't find in model")
        
        print(f"🔍 y_true tipi: {type(y_true)}")
        print(f"🔍 y_true shape: {getattr(y_true, 'shape', 'N/A')}")
        print(f"🔍 y_true ilk 3: {y_true[:3] if len(y_true) > 0 else 'boş'}")
        print(f"🔍 y_pred tipi: {type(y_pred)}")
        print(f"🔍 y_pred ilk 3: {y_pred[:3]}")
        
        # Model instance'da son accuracy var mı kontrol et
        if hasattr(model_instance, 'last_accuracy'):
            print(f"🎯 Model'de kayıtlı accuracy: {model_instance.last_accuracy}")
        
        # Eğer modelin içinde LabelEncoder varsa --> Atakan modeli
        if hasattr(model_instance, "le") and model_instance.le is not None:
            print("📝 LabelEncoder kullanılıyor...")
            y_true_encoded = model_instance.le.transform(y_true)
        else:
            # y_true zaten integer mı kontrol et
            if len(y_true) > 0 and isinstance(y_true[0], (int, np.integer)):
                print("📝 y_true zaten integer formatında!")
                y_true_encoded = np.array(y_true)
            else:
                print("📝 y_true string formatında, mapping uygulanıyor...")
                # NUMPY ARRAY SORUNU ÇÖZÜLMESİ
                mapping = {
                    '0=Blood Donor': 0,
                    '1=Hepatitis': 1,
                    '2=Fibrosis': 2,
                    '3=Cirrhosis': 3,
                    '0s=suspect Blood Donor': 4
                }
                
                # Manuel mapping - numpy array için
                y_true_encoded = []
                for value in y_true:
                    if value in mapping:
                        y_true_encoded.append(mapping[value])
                    else:
                        print(f"⚠️ UYARI: '{value}' mapping'de bulunamadı!")
                        y_true_encoded.append(-1)  # Bilinmeyen değer
                
                y_true_encoded = np.array(y_true_encoded)
            
        print(f"🔢 y_true_encoded: {y_true_encoded}")
        
        target_col_name = getattr(model_instance, "target_column", "Unknown")
        
        # -1 değerleri (bilinmeyen) çıkar
        valid_mask = y_true_encoded != -1
        y_true_clean = y_true_encoded[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        print(f"🧹 Temizlik sonrası - y_true: {len(y_true_clean)}, y_pred: {len(y_pred_clean)}")
        
        try:
            accuracy = accuracy_score(y_true_clean, y_pred_clean)
            print(f"✅ Accuracy hesaplandı: {accuracy}")
        except Exception as acc_error:
            print(f"❌ Accuracy hesaplama hatası: {acc_error}")
            print(f"   y_true_clean: {y_true_clean}")
            print(f"   y_pred_clean: {y_pred_clean}")
            accuracy = 0.0
        
        class_dist = pd.Series(y_pred).value_counts().to_dict()
        
        detailed = df.copy()
        detailed["prediction"] = y_pred
        detailed["true"] = y_true_encoded
        
        actual_vs_predicted = [
            {
                "actual": int(true),
                "predicted": int(pred),
                "correct": int(true) == int(pred)
            }
            for true, pred in zip(y_true_clean, y_pred_clean)
        ]
        
        result = {
            "accuracy": round(accuracy, 4),
            "total_samples": len(y_true_clean),
            "class_distribution": class_dist,
            "target_column": target_col_name,
            "description": f"Model '{model_instance.__class__.__name__}' için sonuçlar",
            "detailed_results": detailed.to_dict(orient="records"),
            "predictions": y_pred.tolist(),
            "modelName": model_instance.__class__.__name__,
            "specialFeature": f"Model '{model_instance.__class__.__name__}' için sonuçlar",
            "actualVsPredicted": actual_vs_predicted
        }
        
        print(f"🎯 FORMAT_RESULTS SONUÇ: accuracy = {result['accuracy']}")
        return result
        
    except Exception as e:
        print(f"❌ FORMAT_RESULTS HATASI: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }
def startup_models():
    print("Models are initalizing...")
    
    try:
        # Klasörleri oluştur
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(MODELS_FILE_PATH, exist_ok=True)
        os.makedirs(SCALERS_FILE_PATH, exist_ok=True)
        
        # Modelleri başlat
        #initialize_sercan_model()
        initialize_suheyla_model()
        initialize_atakan_model()
        
        print("Models initialized  clearly!")
        
    except Exception as e:
        print(f"Startup errpr: {e}")

if __name__ == '__main__':
    startup_models()  # Modelleri yükle
    app.run()
