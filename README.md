## 🎓 Student Performance Prediction – End‑to‑End ML Project

This repository contains an end‑to‑end Machine Learning pipeline and a production‑style Streamlit app that predicts a student’s final exam score (G3) and risk of failing, based on demographic, behavioral, and academic factors.

**Live demo**: [Open the Streamlit app](https://student-performance-ml-jbe5qgg4mq547ngrxpoh8d.streamlit.app)

The project is built to be **reproducible, explainable, and easy to extend**, making it suitable as a portfolio piece or a foundation for academic analytics tools.

---

## 🚀 Key Features

- **End‑to‑end pipeline**
  - Raw data ingestion and cleaning
  - Feature engineering for support, risk, and behavioral signals
  - Model training, evaluation, and persistence to disk

- **Multiple ML models**
  - Baselines: Linear Regression
  - Tree‑based models: Random Forest
  - Gradient boosting: XGBoost (optional, for ensembling)

- **Interactive Streamlit app (`app.py`)**
  - Professional, dark‑themed **Student Performance & Risk Dashboard**
  - **Quick estimate vs Full profile** input modes
  - Final grade (G3) estimation and pass/fail classification
  - Risk gauge, performance meter, and radar profile visualizations
  - Optional SHAP‑based feature attribution for tree models

- **Reproducible environment**
  - All dependencies pinned in `requirements.txt`
  - Clear separation between data, models, notebooks, and source code

---

## 📂 Project Structure

- **`app.py`** – Streamlit web application for real‑time prediction, visualization, and explainability.
- **`src/`**
  - `data_cleaning.py` – Utilities for reading, cleaning, and preparing the raw dataset.
  - `feature_engineering.py` – Construction of derived features and preprocessing pipelines.
  - `model_utils.py` – Model training, evaluation, and persistence helpers.
- **`data/`**
  - `raw/student-mat.csv` – Original student performance dataset (UCI format).
  - `processed/cleaned_student_mat.csv` – Cleaned and feature‑engineered dataset used for modeling.
- **`models/`**
  - `student_performance_model.pkl` – Main trained model (typically Random Forest, possibly wrapped in a pipeline).
  - `test_model.pkl` – Optional secondary/XGBoost model for ensembling.
  - `preprocessor.pkl` – Optional standalone preprocessing pipeline (if not embedded in the main model).
- **`notebooks/`**
  - `01_EDA.ipynb` – Exploratory Data Analysis.
  - `02_Feature_Engineering.ipynb` – Prototyping transformations and derived features.
  - `03_Model_Training.ipynb` – Model comparison, tuning, and evaluation.
- **`requirements.txt`** – Python dependencies for the project.

---

## 📊 Dataset

The project is based on the **UCI Student Performance** dataset (Portuguese language class):

- **Target variable**
  - `G3` – Final grade (0–20)
- **Example feature groups**
  - Demographics: `age`, `sex`, `address`, `famsize`, `Pstatus`
  - Parents & home: `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`, `reason`
  - Study & support: `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`
  - Social & lifestyle: `goout`, `Dalc`, `Walc`, `health`, `absences`
  - Prior performance: `G1`, `G2`

The cleaned version used in the app is stored as `data/processed/cleaned_student_mat.csv`.

---

## 🧠 Modeling Approach

- **Problem types**
  - **Regression**: predict numeric final grade `G3`
  - **Derived classification**: convert predicted grade into **Pass/Fail** based on a configurable threshold (default: 10/20)

- **Models**
  - Linear Regression as a simple, interpretable baseline
  - Random Forest and optionally XGBoost for higher performance and non‑linear interactions
  - Optional ensembling in the app by averaging predictions from Random Forest and XGBoost (when available)

- **Evaluation**
  - Regression metrics: RMSE, MAE, R² (implemented and explored in notebooks / utilities)
  - Qualitative evaluation via SHAP‑style explanations for selected tree‑based models in the app

---

## 🧩 Streamlit App Overview

The Streamlit app (`app.py`) exposes the trained models through a polished dashboard:

- **Input modes**
  - **Quick estimate**: a small set of high‑impact features (age, study time, absences, prior grades, support, social/risk factors). Remaining fields are filled using realistic defaults.
  - **Full profile**: full control over all available features, grouped into logical sections.

- **Key UI components**
  - **Risk gauge** – percentage risk of failing.
  - **Performance meter** – estimated final grade (G3) as a bullet chart with the pass threshold highlighted.
  - **Radar chart** – heuristic scores for:
    - Study Habits
    - Support System
    - Focus vs Distractions
    - Wellbeing
    - Past Performance
  - **Explainability** (when a tree‑based model is available)
    - SHAP‑like bar chart showing the top feature contributions for the current prediction.

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>.git
   cd student-performance-ml
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data and models are in place**

   - Place the raw dataset at `data/raw/student-mat.csv` (if you want to rerun the pipeline or notebooks).
   - Ensure the following files exist to run the app in prediction mode:
     - `models/student_performance_model.pkl`
     - (Optional) `models/test_model.pkl`
     - (Optional) `models/preprocessor.pkl`

   If you do not have the trained models yet, you can train them using the notebooks and utilities under `src/` and then export them with `joblib`.

---

## ▶️ Running the Streamlit App

You can either use the **hosted version** or run the app locally.

- **Hosted (recommended for quick demo)**:  
  [Open the Streamlit app](https://student-performance-ml-jbe5qgg4mq547ngrxpoh8d.streamlit.app)

- **Run locally** (from the project root):

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`) in your browser.

Inside the app you can:

- Choose **Quick estimate** or **Full profile** for inputs.
- Adjust **ensemble usage** and **pass threshold** from the sidebar.
- Inspect risk, predicted grade, and contribution charts for each scenario.

---

## 🔧 Re‑training / Extending the Models

- Use the notebooks in `notebooks/` to:
  - Explore the data (EDA)
  - Prototype new features
  - Compare modeling approaches
- Move stable transformations into:
  - `src/data_cleaning.py`
  - `src/feature_engineering.py`
  - `src/model_utils.py`
- Retrain your models and export them as `.pkl` files under `models/`.
- The app is written to be **resilient**:
  - It can optionally load a separate `preprocessor.pkl`.
  - It can optionally ensemble a second model (e.g. XGBoost) if present.

---

## 💡 Possible Improvements

- Automatic hyperparameter search and experiment tracking (e.g. with Optuna or MLflow).
- More robust model governance (versioning, performance monitoring).
- Support for additional courses or multi‑output models (e.g. Portuguese and Math).
- Role‑based dashboards for teachers, counselors, or administrators.

---

## 📜 License

This project is licensed under the terms specified in `LICENSE`. Feel free to use it as a learning resource or as a starting point for your own educational analytics projects.