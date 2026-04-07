from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

# Credit card dataset ke exact column names
FEATURE_NAMES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                 'V28', 'Amount']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        df = pd.DataFrame([data], columns=FEATURE_NAMES)  # ✅ proper DataFrame
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return jsonify({
            "fraud": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000)