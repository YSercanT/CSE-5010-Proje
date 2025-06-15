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
            model_path=os.path.join(MODELS_FILE_PATH, "rf.pkl"),  
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
            raise Exception("VPlease upload file first")
        
        print("Data Loading")
        df = pd.read_csv(LAST_FILE_PATH)
        print(f"Data Shape: {df.shape}")
        
        print("Making Prediction")
        predictions = sercan_model_instance.prediction(df.copy())
        
        print("Postprocessing")
        final_predictions = sercan_model_instance.postprocess_output(predictions)
        
        results = format_results(df, final_predictions, sercan_model_instance)
        
        print(f"Sercan modeli build successfully! Accuracy: {results.get('accuracy', 'N/A')}")
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
        y_true = getattr(model_instance, "y", None)
        if y_true is None:
            raise Exception("True y couldn't find in model")

        # Eğer modelin içinde LabelEncoder varsa --> Atakan modeli
        if hasattr(model_instance, "le") and model_instance.le is not None:
            y_true_encoded = model_instance.le.transform(y_true)
        else:
            # Suheyla modeli --> mapping uygula
            mapping = {
                '0=Blood Donor': 0,
                '1=Hepatitis': 1,
                '2=Fibrosis': 2,
                '3=Cirrhosis': 3
            }
            y_true_encoded = y_true.map(mapping)

        target_col_name = getattr(model_instance, "target_column", "Unknown")

        accuracy = accuracy_score(y_true_encoded, y_pred)
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
            for true, pred in zip(y_true_encoded, y_pred)
        ]

        return {
            "accuracy": round(accuracy, 4),
            "total_samples": len(y_true_encoded),
            "class_distribution": class_dist,
            "target_column": target_col_name,
            "description": f"Model '{model_instance.__class__.__name__}' için sonuçlar",
            "detailed_results": detailed.to_dict(orient="records"),
            "predictions": y_pred.tolist(),
            "modelName": model_instance.__class__.__name__,
            "specialFeature": f"Model '{model_instance.__class__.__name__}' için sonuçlar",
            "actualVsPredicted": actual_vs_predicted
        }
    except Exception as e:
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
