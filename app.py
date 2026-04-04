from flask import Flask,request,jsonify
import joblib
import numpy as np
app=Flask(__name__)
model=joblib.load("fraud_model.pkl")
@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.json['data']
        prediction=model.predict([data])[0]
        probability=model.predict_proba([data])[0][1]
        return jsonify({
            "fraud":int(prediction),
            "probability":float(probability)})
    except Exception as e:
        return jsonify({"error":str(e)})
if __name__=="__main__":    
    app.run(port=5000)