from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ---- Load Model Artifacts from models/ folder ----
model          = joblib.load('models/cvd_xgboost_model.joblib')
scaler         = joblib.load('models/cvd_scaler.joblib')
encoders       = joblib.load('models/cvd_label_encoders.joblib')
feature_names  = joblib.load('models/cvd_feature_names.joblib')
numerical_cols = joblib.load('models/cvd_numerical_cols.joblib')

# ---- Default values for fields not in the form ----
DEFAULTS = {
    'General_Health' : 'Good',
    'Skin_Cancer'    : 'No',
    'Other_Cancer'   : 'No',
    'Arthritis'      : 'No',
}

# ---- Home Route ----
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# ---- Predict Route ----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data + fill missing fields with defaults
        patient = {
            'General_Health'              : DEFAULTS['General_Health'],
            'Checkup'                     : request.form.get('checkup'),
            'Exercise'                    : request.form.get('exercise'),
            'Skin_Cancer'                 : DEFAULTS['Skin_Cancer'],
            'Other_Cancer'                : DEFAULTS['Other_Cancer'],
            'Depression'                  : request.form.get('depression'),
            'Diabetes'                    : request.form.get('diabetes'),
            'Arthritis'                   : DEFAULTS['Arthritis'],
            'Sex'                         : request.form.get('sex'),
            'Age_Category'                : request.form.get('age_category'),
            'Height_(cm)'                 : float(request.form.get('height')),
            'Weight_(kg)'                 : float(request.form.get('weight')),
            'BMI'                         : float(request.form.get('bmi')),
            'Smoking_History'             : request.form.get('smoking'),
            'Alcohol_Consumption'         : float(request.form.get('alcohol')),
            'Fruit_Consumption'           : float(request.form.get('fruit')),
            'Green_Vegetables_Consumption': float(request.form.get('vegetables')),
            'FriedPotato_Consumption'     : float(request.form.get('fried_potato')),
        }

        # Build feature row in correct order
        row = {feat: patient.get(feat, np.nan) for feat in feature_names}

        # Encode categorical features
        for col, le in encoders.items():
            if col in row:
                val = str(row[col])
                row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

        # Convert to numpy array
        x_arr = np.array(
            [row[f] for f in feature_names],
            dtype=np.float32
        ).reshape(1, -1)

        # Scale numerical features
        num_indices = [feature_names.index(c) for c in numerical_cols
                       if c in feature_names]
        x_arr[:, num_indices] = scaler.transform(x_arr[:, num_indices])

        # Predict
        prob       = float(model.predict_proba(x_arr)[0, 1])
        prediction = int(prob >= 0.5)
        risk_label = 'High Risk' if prediction == 1 else 'Low Risk'
        confidence = f"{prob * 100:.1f}%"

        return render_template(
            'result.html',
            risk_label = risk_label,
            prediction = prediction,
            confidence = confidence,
            prob       = round(prob * 100, 1),
            patient    = patient
        )

    except Exception as e:
        return render_template(
            'result.html',
            error = f"Something went wrong: {str(e)}"
        )

# ---- Run App ----
if __name__ == '__main__':
    app.run(debug=True)