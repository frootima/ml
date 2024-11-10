from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/predict", methods=["POST"])
def predict():
    # Parse input data
    data = request.json
    features = np.array(data["features"])
    # Reshape for prediction if single sample
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    # Predict
    prediction = model.predict(features).tolist()
    probability = model.predict_proba(features).tolist()
    return jsonify({"prediction": prediction, "probability": probability})

if __name__ == "__main__":
    app.run(debug=True)

