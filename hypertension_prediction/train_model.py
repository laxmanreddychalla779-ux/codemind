"""
train_model.py — Run this ONCE to train and save the Logistic Regression model.

Usage:
  1. Place your dataset CSV as 'patient_data.csv' in the same folder.
  2. Run: python train_model.py
  3. This generates 'logreg_model.pkl' for the Flask app to load.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_clean(path='patient_data.csv'):
    data = pd.read_csv(path)

    # Rename column 'C' to 'Gender' if needed
    if 'C' in data.columns:
        data.rename(columns={'C': 'Gender'}, inplace=True)

    # Fix inconsistencies
    data['TakeMedication'].replace({'Yes ': 'Yes'}, inplace=True)
    data['NoseBleeding'].replace({'No ': 'No'}, inplace=True)
    data['Systolic'].replace({'121- 130': '121 - 130'}, inplace=True)
    data['Systolic'].replace({'100+': '100 - 110'}, inplace=True)
    data['Stages'].replace({'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)'}, inplace=True)
    data['Stages'].replace({'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'}, inplace=True)
    data['Diastolic'].replace({'130+': '100+'}, inplace=True)

    # Remove duplicates
    data.drop_duplicates(inplace=True)
    print(f"✅ Loaded {len(data)} records after cleaning.")
    return data


def encode(data):
    nominal_features = ['Gender', 'History', 'Patient', 'TakeMedication',
                        'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet']
    ordinal_features = ['Age', 'Severity', 'Whendiagnoused', 'Systolic', 'Diastolic']

    for col in nominal_features:
        if col == 'Gender':
            data[col] = data[col].map({'Male': 0, 'Female': 1})
        else:
            data[col] = data[col].map({'No': 0, 'Yes': 1})

    data['Age'] = data['Age'].map({'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4})
    data['Severity'] = data['Severity'].replace({'Mild': 0, 'Moderate': 1, 'Severe': 2})
    data['Whendiagnoused'] = data['Whendiagnoused'].map({'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3})
    data['Systolic'] = data['Systolic'].map({'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3})
    data['Diastolic'] = data['Diastolic'].map({'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3})
    data['Stages'] = data['Stages'].map({
        'NORMAL': 0,
        'HYPERTENSION (Stage-1)': 1,
        'HYPERTENSION (Stage-2)': 2,
        'HYPERTENSIVE CRISIS': 3
    })

    scaler = MinMaxScaler()
    data[ordinal_features] = scaler.fit_transform(data[ordinal_features])
    return data


def train(data):
    X = data.drop('Stages', axis=1)
    y = data['Stages']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'logreg_model.pkl')
    print("✅ Model saved as logreg_model.pkl")
    return model


if __name__ == '__main__':
    data = load_and_clean()
    data = encode(data)
    train(data)
