from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('/home/visam/logistic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Input should be a JSON payload
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': probability.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)

