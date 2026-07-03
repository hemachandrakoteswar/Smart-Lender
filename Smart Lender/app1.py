import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load the saved model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'rdf.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scale1.pkl')

model = pickle.load(open(model_path, 'rb'))
scale = pickle.load(open(scaler_path, 'rb'))

# Feature names exactly as used during training
FEATURE_NAMES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'GET':
        return redirect(url_for('predict'))
        
    try:
        # Retrieve form parameters by name
        gender = int(request.form['Gender'])
        married = int(request.form['Married'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        self_employed = int(request.form['Self_Employed'])
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])
        
        # Structure the features in the exact order trained
        feature_vector = [
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        ]
        
        # Convert to DataFrame with feature names to prevent scikit-learn warnings
        features_df = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
        scaled_features = scale.transform(features_df)
        
        # Convert scaled features back to DataFrame with feature names
        scaled_features_df = pd.DataFrame(scaled_features, columns=FEATURE_NAMES)
        
        # Predict outcome
        prediction = model.predict(scaled_features_df)
        
        if prediction[0] == 1:
            result = "Loan will be Approved"
        else:
            result = "Loan will Not be Approved"
            
        return render_template('submit.html', result=result)
        
    except Exception as e:
        return f"An error occurred during prediction processing: {str(e)}", 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
