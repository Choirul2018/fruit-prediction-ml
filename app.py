from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

def load_models():
    """Load all models and return them in a dictionary."""
    try:
        model_files = {
            'svm': os.path.join(os.getcwd(), 'models', 'svmfruit.pkl'),
            'perceptron': os.path.join(os.getcwd(), 'models', 'perceptronfruit.pkl'),
            'rf': os.path.join(os.getcwd(), 'models', 'rffruit.pkl')
        }

        models = {}
        
        for model_name, file_name in model_files.items():
            file_path = os.path.join(os.getcwd(), file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_name} not found in the directory.")
            
            # Load model dengan pickle
            with open(file_path, 'rb') as f:
                models[model_name] = pickle.load(f)
        
        print("All models loaded successfully!")
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load models saat prediksi
    models = load_models()
    if models is None:
        return jsonify({'success': False, 'error': 'Failed to load models.'})
    
    try:
        # Ambil fitur dari request form
        features = np.array([[
            float(request.form['diameter']),
            float(request.form['weight']),
            float(request.form['red']),
            float(request.form['green']),
            float(request.form['blue'])
        ]])
        
        # Pilih model berdasarkan input
        model_name = request.form['model']
        if model_name not in models:
            return jsonify({'success': False, 'error': f"Model '{model_name}' not found."})
        
        # Prediksi hasil
        prediction = models[model_name].predict(features)
        result = 'Orange' if prediction[0] == 0 else 'Grapefruit'
        
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
