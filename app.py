from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model_path = 'winequality_model.pkl'
scaler_path = 'scaler.pkl'

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Ensure the model is saved as 'winequality_model.pkl'.")

try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Scaler file not found. Ensure the scaler is saved as 'scaler.pkl'.")

@app.route('/')
def home():
    return "Welcome to the Wine Quality Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if len(data['features']) <= 10: 
        return jsonify({'error': 'Invalid number of features'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)