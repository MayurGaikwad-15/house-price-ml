# 🏠 House Price Prediction System

### 🚀 Production-Ready ML App with FastAPI + Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-XGBoost-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge"/>
</p>

---

## 🎯 Live Demo

🚧 *Coming Soon (Deploy on Render + Streamlit Cloud)*

---

## 📸 Screenshots

### 🖥️ Streamlit UI

![UI Screenshot](https://via.placeholder.com/900x400.png?text=Streamlit+UI+Preview)

---

### ⚡ FastAPI Docs

![API Screenshot](https://via.placeholder.com/900x400.png?text=FastAPI+Swagger+Docs)

---

## 🎥 Demo GIF

![Demo GIF](https://via.placeholder.com/900x400.gif?text=App+Demo+GIF)

> 💡 Replace placeholders with real screenshots using:
>
> * `Win + Shift + S` (Windows screenshot)
> * Upload to GitHub → copy image link

---

## 🚀 Project Overview

A **full-stack machine learning system** that predicts house prices using structured data.
Designed with **production-level architecture**, not just a notebook model.

---

## 🧠 Features

✅ End-to-End ML Pipeline
✅ Numerical + Categorical Handling
✅ Automatic Model Selection
✅ FastAPI REST API
✅ Streamlit Interactive UI
✅ Schema Validation & Robust Inference

---

## 🏗️ Architecture

```
User (Streamlit UI)
        ↓
FastAPI Backend (/predict)
        ↓
Schema Alignment Layer
        ↓
ML Pipeline (Imputer + Encoder + Scaler)
        ↓
Best Model (XGBoost / RF / Ridge)
        ↓
Prediction Output
```

---

## ⚙️ Tech Stack

| Layer      | Technology                         |
| ---------- | ---------------------------------- |
| Language   | Python                             |
| ML         | Scikit-learn, XGBoost              |
| Backend    | FastAPI                            |
| Frontend   | Streamlit                          |
| Data       | Pandas, NumPy                      |
| Deployment | (Planned) Render / Streamlit Cloud |

---

## 📂 Project Structure

```
house-price-ml/
│
├── api/
│   └── app.py
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── train.py
│   ├── pipeline.py
│   ├── preprocessing.py
│
├── models/
│   └── model.pkl
│
├── data/
│   └── train.csv
│
├── requirements.txt
└── README.md
```

---

## 🧪 Model Training

```bash
python -m src.train
```

✔ Trains multiple models
✔ Selects best based on RMSE
✔ Saves full pipeline

---

## 🚀 Run Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Start Backend

```bash
uvicorn api.app:app --reload
```

📍 http://127.0.0.1:8000

---

### 3️⃣ Start Frontend

```bash
streamlit run app/streamlit_app.py
```

📍 http://localhost:8501

---

## 🔌 API Example

### Request

```json
{
  "MSSubClass": 20,
  "LotArea": 8000,
  "OverallQual": 5,
  "OverallCond": 5,
  "YearBuilt": 2000,
  "TotalBsmtSF": 800,
  "FirstFlrSF": 800,
  "SecondFlrSF": 500,
  "FullBath": 2,
  "BedroomAbvGr": 3,
  "TotRmsAbvGrd": 6,
  "GarageCars": 2,
  "GarageArea": 500,
  "YrSold": 2008
}
```

### Response

```json
{
  "price": 245000.75
}
```

---

## 📊 ML Pipeline Details

* Missing Value Handling → `SimpleImputer`
* Scaling → `StandardScaler`
* Encoding → `OneHotEncoder`
* Pipeline → `ColumnTransformer`
* Model Selection → RMSE-based

---

## 💡 Key Highlights

🔥 Production-grade pipeline (not notebook ML)
🔥 Handles missing + unseen data
🔥 Full-stack ML system
🔥 Clean modular architecture

---

## 🚀 Future Improvements

* 🔍 SHAP Explainability
* 🌍 Cloud Deployment
* 📈 Model Monitoring
* 🎨 Premium UI (Glassmorphism)

---

## 👨‍💻 Author

**Mayur**
Aspiring AI/ML Engineer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
