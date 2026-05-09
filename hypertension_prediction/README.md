# PredictivePulse — Hypertension Risk Assessment

An AI-powered, full-stack Flask web application for hypertension classification using Logistic Regression (95.2% accuracy).

---

## 📁 Project Structure

```
HYPERTENSION_PREDICTION/
├── static/
│   └── style.css          ← Liquid Glass UI styling
├── templates/
│   └── index.html         ← Full-stack frontend
├── app.py                 ← Flask backend + prediction logic
├── train_model.py         ← One-time model training script
├── logreg_model.pkl       ← Trained model (generated after training)
├── requirements.txt       ← Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (One-Time)
Download the dataset from Kaggle (or the provided Google Drive link), save it as `patient_data.csv` in the project folder, then run:
```bash
python train_model.py
```
This generates `logreg_model.pkl`.

### 3. Run the Application
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🧠 Model Details

| Algorithm           | Accuracy | Generalization |
|---------------------|----------|----------------|
| Logistic Regression | **95.2%**| ✅ Selected    |
| KNN                 | 98.1%    | Good           |
| Ridge Classifier    | 90.0%    | Good           |
| Decision Tree       | 100%     | ❌ Overfitted  |
| Random Forest       | 100%     | ❌ Overfitted  |
| SVM                 | 100%     | ❌ Overfitted  |
| Naive Bayes         | 84.4%    | Good           |

---

## 🏥 Features

- 4-class hypertension classification: Normal, Stage-1, Stage-2, Crisis
- Real-time AI prediction with confidence score
- Color-coded risk assessment (green → red)
- Personalized clinical recommendations per stage
- Liquid Glass dark UI with animated background
- Fully responsive (mobile + desktop)
- Demo mode when model file is absent

---

## 📊 Dataset

- **Source**: Kaggle
- **Records**: 1,825 (1,348 after deduplication)
- **Features**: Gender, Age, Medical History, Symptoms, BP Readings, Lifestyle
- **Target**: Hypertension Stage (0–3)

---

## ⚠️ Disclaimer

This tool is intended for clinical decision support only. It is **not** a substitute for professional medical advice, diagnosis, or treatment.
