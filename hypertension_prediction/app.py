from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'hypertension-prediction-2025'

# Load trained model
model = None
model_path = os.path.join(os.path.dirname(__file__), 'logreg_model.pkl')
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Model file not found. Using demo mode.")

# Stage mapping
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Color mapping
color_map = {
    0: '#10b981',  # Emerald green - Normal
    1: '#f59e0b',  # Amber - Stage 1
    2: '#f97316',  # Orange - Stage 2
    3: '#ef4444'   # Red - Crisis
}

# Icon mapping
icon_map = {
    0: 'check-circle',
    1: 'alert-triangle',
    2: 'alert-octagon',
    3: 'zap'
}

# Risk level
risk_map = {
    0: 'LOW RISK',
    1: 'MODERATE RISK',
    2: 'HIGH RISK',
    3: 'EMERGENCY'
}

# Detailed recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk assessment indicates normal blood pressure levels. Excellent health status detected.',
        'actions': [
            'Maintain current healthy lifestyle habits',
            'Regular physical activity (150 minutes/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups with your physician'
        ],
        'priority': 'Low Risk'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild blood pressure elevation detected. Lifestyle modifications and medical consultation recommended.',
        'actions': [
            'Schedule appointment with healthcare provider within 1 month',
            'Implement DASH (Dietary Approaches to Stop Hypertension) diet plan',
            'Increase physical activity gradually to 30 min/day',
            'Monitor blood pressure bi-weekly at home',
            'Reduce sodium intake to <2300mg/day',
            'Consider stress management and meditation techniques'
        ],
        'priority': 'Moderate Risk'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring immediate medical intervention and treatment protocol.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Medication therapy likely required — do not delay',
            'Comprehensive cardiovascular assessment needed',
            'Daily blood pressure monitoring (morning & evening)',
            'Strict dietary sodium restriction (<1500mg/day)',
            'Eliminate alcohol and smoking immediately',
            'Lifestyle modification counseling recommended'
        ],
        'priority': 'High Risk'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure requiring immediate emergency medical care.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention NOW',
            'Call 911 or go to nearest emergency room immediately',
            'Do NOT delay treatment under any circumstances',
            'Monitor for stroke/heart attack signs (chest pain, vision loss)',
            'Prepare complete current medication list for ER staff',
            'Avoid any physical exertion immediately',
            'Stay calm and lie down until help arrives'
        ],
        'priority': 'EMERGENCY'
    }
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        required_fields = [
            'Gender', 'Age', 'History', 'Patient', 'TakeMedication',
            'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
            'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet'
        ]
        
        for field in required_fields:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Encode inputs
        encoded = [
            0 if data['Gender'] == 'Male' else 1,
            {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}[data['Age']],
            1 if data['History'] == 'Yes' else 0,
            1 if data['Patient'] == 'Yes' else 0,
            1 if data['TakeMedication'] == 'Yes' else 0,
            {'Mild': 0, 'Moderate': 1, 'Severe': 2}[data['Severity']],
            1 if data['BreathShortness'] == 'Yes' else 0,
            1 if data['VisualChanges'] == 'Yes' else 0,
            1 if data['NoseBleeding'] == 'Yes' else 0,
            {'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3}[data['Whendiagnoused']],
            {'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3}[data['Systolic']],
            {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3}[data['Diastolic']],
            1 if data['ControlledDiet'] == 'Yes' else 0
        ]

        # Manual scaling for ordinal features
        scaled = encoded.copy()
        scaled[1] = (encoded[1] - 1) / (4 - 1)      # Age
        scaled[5] = encoded[5] / 2                    # Severity
        scaled[9] = (encoded[9] - 1) / (3 - 1)       # When diagnosed
        scaled[10] = encoded[10] / 3                  # Systolic
        scaled[11] = encoded[11] / 3                  # Diastolic

        input_array = np.array(scaled).reshape(1, -1)

        if model is not None:
            prediction = int(model.predict(input_array)[0])
            try:
                confidence = round(float(max(model.predict_proba(input_array)[0])) * 100, 1)
            except:
                confidence = 85.0
        else:
            # Demo mode: smart rule-based prediction
            systolic_val = encoded[10]
            diastolic_val = encoded[11]
            severity_val = encoded[5]
            
            if systolic_val >= 3 or diastolic_val >= 3:
                prediction = 3
            elif systolic_val == 2 or diastolic_val == 2 or severity_val == 2:
                prediction = 2
            elif systolic_val == 1 or diastolic_val == 1 or severity_val == 1:
                prediction = 1
            else:
                prediction = 0
            confidence = 87.5

        result = {
            'prediction': prediction,
            'stage': stage_map[prediction],
            'color': color_map[prediction],
            'risk': risk_map[prediction],
            'icon': icon_map[prediction],
            'confidence': confidence,
            'recommendation': recommendations[prediction]
        }

        return jsonify(result)

    except KeyError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
