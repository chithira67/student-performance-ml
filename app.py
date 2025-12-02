import os
import joblib
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import streamlit as st
import plotly.graph_objects as go


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    """
    Load preprocessor and models.
    - main_model: typically a Random Forest pipeline (may include preprocessing)
    - xgb_model: optional XGBoost model for ensembling (if available)
    - preprocessor: optional separate preprocessor if not inside pipeline
    """
    models: Dict[str, Any] = {"main_model": None, "xgb_model": None, "preprocessor": None}

    # Main model (Random Forest pipeline by default)
    main_model_path = os.path.join("models", "student_performance_model.pkl")
    if os.path.exists(main_model_path):
        models["main_model"] = joblib.load(main_model_path)

    # Optional XGBoost model (stored separately, e.g., test_model.pkl)
    xgb_model_path = os.path.join("models", "test_model.pkl")
    if os.path.exists(xgb_model_path):
        try:
            models["xgb_model"] = joblib.load(xgb_model_path)
        except Exception:
            # Don't break the app if this can't be loaded
            models["xgb_model"] = None

    # Optional standalone preprocessor
    preproc_path = os.path.join("models", "preprocessor.pkl")
    if os.path.exists(preproc_path):
        try:
            models["preprocessor"] = joblib.load(preproc_path)
        except Exception:
            models["preprocessor"] = None

    return models


@st.cache_resource(show_spinner=False)
def load_g3_proxy_model() -> Optional[Dict[str, Any]]:
    """
    Fit a lightweight regression proxy for G3 using the processed dataset.
    This is used to fill the G3 column when the deployed pipeline expects it.
    """
    data_path = os.path.join("data", "processed", "cleaned_student_mat.csv")
    if not os.path.exists(data_path):
        return None

    try:
        df = pd.read_csv(data_path)
    except Exception:
        return None

    if "G3" not in df.columns:
        return None

    feature_cols = [col for col in df.columns if col != "G3"]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df["G3"]

    if X.empty:
        return None

    model = LinearRegression()
    model.fit(X, y)
    return {"model": model, "columns": X.columns.tolist()}


def is_classifier(model: Any) -> bool:
    """Best-effort check if a model behaves like a classifier."""
    if model is None:
        return False
    # Pipeline handling
    base = getattr(model, "named_steps", None)
    if base and "model" in model.named_steps:
        base_est = model.named_steps["model"]
    else:
        base_est = model

    if hasattr(base_est, "predict_proba"):
        return True
    if hasattr(base_est, "_estimator_type") and base_est._estimator_type == "classifier":
        return True
    return False


def get_tree_estimator(model: Any) -> Optional[Any]:
    """Return underlying tree-based estimator for SHAP, if possible."""
    if model is None:
        return None
    base = getattr(model, "named_steps", None)
    if base and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def build_input_schema() -> Dict[str, Dict[str, Any]]:
    """
    Schema describing Streamlit widgets for each feature.
    Based on columns in cleaned_student_mat.csv.
    """
    schema: Dict[str, Dict[str, Any]] = {
        # Demographics
        "age": {"type": "number", "min": 15, "max": 22, "default": 17, "help": "Student age in years"},
        "sex": {"type": "select", "options": ["F", "M"], "default": "F", "help": "Student gender"},
        "address": {
            "type": "select",
            "options": ["U", "R"],
            "default": "U",
            "help": "Home address type: Urban (U) or Rural (R)",
        },
        "famsize": {
            "type": "select",
            "options": ["LE3", "GT3"],
            "default": "GT3",
            "help": "Family size: LE3 (≤3) or GT3 (>3)",
        },
        "Pstatus": {
            "type": "select",
            "options": ["T", "A"],
            "default": "T",
            "help": "Parent cohabitation status: Together (T) or Apart (A)",
        },
        # Parents' education
        "Medu": {
            "type": "number",
            "min": 0,
            "max": 4,
            "default": 2,
            "help": "Mother's education (0–4, higher is more educated)",
        },
        "Fedu": {
            "type": "number",
            "min": 0,
            "max": 4,
            "default": 2,
            "help": "Father's education (0–4, higher is more educated)",
        },
        # Parents' jobs & reason
        "Mjob": {
            "type": "select",
            "options": ["teacher", "health", "services", "at_home", "other"],
            "default": "other",
            "help": "Mother's job",
        },
        "Fjob": {
            "type": "select",
            "options": ["teacher", "health", "services", "at_home", "other"],
            "default": "other",
            "help": "Father's job",
        },
        "reason": {
            "type": "select",
            "options": ["home", "reputation", "course", "other"],
            "default": "course",
            "help": "Reason to choose this school",
        },
        "guardian": {
            "type": "select",
            "options": ["mother", "father", "other"],
            "default": "mother",
            "help": "Main guardian",
        },
        # Study and commute
        "traveltime": {
            "type": "number",
            "min": 1,
            "max": 4,
            "default": 1,
            "help": "Home to school travel time (1–4, higher = longer)",
        },
        "studytime": {
            "type": "number",
            "min": 1,
            "max": 4,
            "default": 2,
            "help": "Weekly study time (1–4, higher = more study)",
        },
        "failures": {
            "type": "number",
            "min": 0,
            "max": 4,
            "default": 0,
            "help": "Number of past class failures (0–4)",
        },
        # Binary yes/no supports and activities
        "schoolsup": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "no",
            "help": "Extra educational support",
        },
        "famsup": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "yes",
            "help": "Family educational support",
        },
        "paid": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "no",
            "help": "Extra paid classes (e.g. math, Portuguese)",
        },
        "activities": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "yes",
            "help": "Extracurricular activities",
        },
        "nursery": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "yes",
            "help": "Attended nursery school",
        },
        "higher": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "yes",
            "help": "Wants to take higher education",
        },
        "internet": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "yes",
            "help": "Internet access at home",
        },
        "romantic": {
            "type": "select",
            "options": ["yes", "no"],
            "default": "no",
            "help": "In a romantic relationship",
        },
        # Social / time use
        "famrel": {
            "type": "number",
            "min": 1,
            "max": 5,
            "default": 4,
            "help": "Quality of family relationships (1–5)",
        },
        "freetime": {
            "type": "number",
            "min": 1,
            "max": 5,
            "default": 3,
            "help": "Free time after school (1–5)",
        },
        "goout": {
            "type": "number",
            "min": 1,
            "max": 5,
            "default": 3,
            "help": "Going out with friends (1–5)",
        },
        "Dalc": {
            "type": "number",
            "min": 1,
            "max": 5,
            "default": 1,
            "help": "Workday alcohol consumption (1–5)",
        },
        "Walc": {
            "type": "number",
            "min": 1,
            "max": 5,
            "default": 1,
            "help": "Weekend alcohol consumption (1–5)",
        },
        "health": {
            "type": "number",
            "min": 1,
            "max": 5,
            "default": 3,
            "help": "Current health status (1–5)",
        },
        "absences": {
            "type": "number",
            "min": 0,
            "max": 93,
            "default": 4,
            "help": "Number of school absences",
        },
        # Prior grades (optional but helpful for some models)
        "G1": {
            "type": "number",
            "min": 0,
            "max": 20,
            "default": 10,
            "help": "First period grade (0–20)",
        },
        "G2": {
            "type": "number",
            "min": 0,
            "max": 20,
            "default": 10,
            "help": "Second period grade (0–20)",
        },
    }
    return schema


def render_sidebar():
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown(
        "Adjust model options, theme preferences, and get documentation links."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Ensemble Settings**")
    st.session_state.setdefault("use_ensemble", True)
    st.session_state["use_ensemble"] = st.sidebar.checkbox(
        "Combine Random Forest + XGBoost (if available)",
        value=st.session_state["use_ensemble"],
        help="If XGBoost model is available, average predictions from both models.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Risk Threshold**")
    st.session_state.setdefault("pass_threshold", 10)
    st.session_state["pass_threshold"] = st.sidebar.slider(
        "Pass threshold on final grade (G3) used to define pass/fail",
        min_value=8,
        max_value=15,
        value=st.session_state["pass_threshold"],
        help="If models output G3 grade, this threshold is used to convert it into pass/fail.",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 Tip: Use tooltips on inputs to better understand each factor's impact."
    )


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global dark theme tweaks */
        .stApp {
            background: radial-gradient(circle at top left, #111827, #020617);
            color: #e5e7eb;
        }
        /* Cards */
        .metric-card {
            padding: 1rem 1.25rem;
            border-radius: 0.9rem;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f9fafb;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        /* Predict button animation */
        .predict-btn > button {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: white;
            font-weight: 600 !important;
            border-radius: 999px !important;
            border: none;
            box-shadow: 0 12px 30px rgba(34, 197, 94, 0.5);
            transition: all 0.25s ease-out;
        }
        .predict-btn > button:hover {
            transform: translateY(-1px) scale(1.02);
            box-shadow: 0 18px 40px rgba(34, 197, 94, 0.7);
        }
        .predict-btn > button:active {
            transform: translateY(1px) scale(0.98);
            box-shadow: 0 6px 18px rgba(34, 197, 94, 0.55);
        }
        /* Section headers */
        .section-header {
            font-size: 1rem;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
            color: #e5e7eb;
        }
        .section-subtitle {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def collect_user_inputs(schema: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    st.markdown(
        '<div class="section-header">🎯 Student Profile</div>'
        '<div class="section-subtitle">Describe the student to estimate performance and risk.</div>',
        unsafe_allow_html=True,
    )

    # Layout inputs into columns by logical groups
    col_demo, col_parents, col_study = st.columns(3)

    inputs: Dict[str, Any] = {}

    # Demographics
    with col_demo:
        st.markdown("**Demographics**")
        for key in ["age", "sex", "address", "famsize", "Pstatus"]:
            cfg = schema[key]
            if cfg["type"] == "number":
                inputs[key] = st.number_input(
                    key,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"],
                    help=cfg["help"],
                )
            else:
                inputs[key] = st.selectbox(
                    key,
                    cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                    help=cfg["help"],
                )

    # Parents
    with col_parents:
        st.markdown("**Parents & Home**")
        for key in ["Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian"]:
            cfg = schema[key]
            if cfg["type"] == "number":
                inputs[key] = st.number_input(
                    key,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"],
                    help=cfg["help"],
                )
            else:
                inputs[key] = st.selectbox(
                    key,
                    cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                    help=cfg["help"],
                )

    # Study / support
    with col_study:
        st.markdown("**Study & Support**")
        for key in ["traveltime", "studytime", "failures"]:
            cfg = schema[key]
            inputs[key] = st.number_input(
                key,
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                help=cfg["help"],
            )

        for key in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]:
            cfg = schema[key]
            inputs[key] = st.selectbox(
                key,
                cfg["options"],
                index=cfg["options"].index(cfg["default"]),
                help=cfg["help"],
            )

    st.markdown("---")
    col_social, col_health, col_grades = st.columns(3)

    with col_social:
        st.markdown("**Social & Time Use**")
        for key in ["famrel", "freetime", "goout"]:
            cfg = schema[key]
            inputs[key] = st.number_input(
                key,
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                help=cfg["help"],
            )

    with col_health:
        st.markdown("**Lifestyle & Health**")
        for key in ["Dalc", "Walc", "health", "absences"]:
            cfg = schema[key]
            inputs[key] = st.number_input(
                key,
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                help=cfg["help"],
            )

    with col_grades:
        st.markdown("**Prior Performance**")
        for key in ["G1", "G2"]:
            cfg = schema[key]
            inputs[key] = st.number_input(
                key,
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                help=cfg["help"],
            )

    # school field (header shows it exists), keep simple in a single select below
    st.markdown("---")
    school = st.selectbox(
        "school",
        ["GP", "MS"],
        index=0,
        help="Student's school (GP: Gabriel Pereira, MS: Mousinho da Silveira).",
    )
    inputs["school"] = school

    # Order columns as in dataset header for maximal compatibility
    ordered_cols = [
        "school",
        "sex",
        "age",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "traveltime",
        "studytime",
        "failures",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        "G1",
        "G2",
    ]

    data = {col: inputs.get(col) for col in ordered_cols}
    df = pd.DataFrame([data])
    return df


def estimate_g3_proxy(
    X: pd.DataFrame,
    proxy_bundle: Optional[Dict[str, Any]],
) -> float:
    """Estimate G3 using a proxy model or fallback heuristics."""
    if proxy_bundle is None:
        if "G2" in X.columns:
            return float(X["G2"].iloc[0])
        if "G1" in X.columns:
            return float(X["G1"].iloc[0])
        return 10.0

    proxy_model = proxy_bundle["model"]
    columns = proxy_bundle["columns"]

    X_enc = pd.get_dummies(X, drop_first=True)
    X_enc = X_enc.reindex(columns=columns, fill_value=0)
    return float(proxy_model.predict(X_enc)[0])


def ensemble_predict(
    X: pd.DataFrame,
    main_model: Any,
    xgb_model: Any,
    preprocessor: Any,
    pass_threshold: float,
    g3_proxy_bundle: Optional[Dict[str, Any]],
) -> Tuple[float, float, str, str]:
    """
    Run predictions from available models and combine them.
    Returns:
        risk_score (0–100),
        est_grade (if regression-like, else converted from classification prob),
        label ("Pass"/"Fail"),
        backing_text (description used for UI)
    """
    preds_reg = []
    preds_prob = []
    backing_pieces = []

    use_preproc = preprocessor is not None and not hasattr(main_model, "named_steps")

    # Some training pipelines (via feature_engineering.py) still included G3 as a feature,
    # so the ColumnTransformer expects a G3 column at prediction time.
    # When serving the model, we *don't* know the final grade yet, so we synthesize it
    # from G2 as a reasonable proxy to avoid column-missing errors.
    X_for_models = X.copy()
    if "G3" not in X_for_models.columns:
        proxy_g3 = estimate_g3_proxy(X, g3_proxy_bundle)
        X_for_models["G3"] = proxy_g3

    def prepare_features(df: pd.DataFrame) -> Any:
        if use_preproc and preprocessor is not None:
            return preprocessor.transform(df)
        return df

    # Main model
    if main_model is not None:
        X_main = prepare_features(X_for_models)
        if is_classifier(main_model):
            if hasattr(main_model, "predict_proba"):
                proba = main_model.predict_proba(X_main)[0, 1]
            else:
                # Fallback: binary decision converted to 0/1
                pred = main_model.predict(X_main)[0]
                proba = float(pred)
            preds_prob.append(proba)
            backing_pieces.append("Random Forest (classification)")
        else:
            y_pred = float(main_model.predict(X_main)[0])
            preds_reg.append(y_pred)
            backing_pieces.append("Random Forest (regression)")

    # XGBoost model
    if xgb_model is not None:
        X_xgb = prepare_features(X_for_models)
        try:
            if is_classifier(xgb_model):
                if hasattr(xgb_model, "predict_proba"):
                    proba = xgb_model.predict_proba(X_xgb)[0, 1]
                else:
                    proba = float(xgb_model.predict(X_xgb)[0])
                preds_prob.append(proba)
                backing_pieces.append("XGBoost (classification)")
            else:
                y_pred = float(xgb_model.predict(X_xgb)[0])
                preds_reg.append(y_pred)
                backing_pieces.append("XGBoost (regression)")
        except Exception:
            # If anything goes wrong, ignore XGBoost contribution
            pass

    # Combine predictions
    backing_text = " + ".join(backing_pieces) if backing_pieces else "No trained model found"

    if preds_reg:
        est_grade = float(np.mean(preds_reg))
        risk_score = max(0.0, min(1.0, (pass_threshold - est_grade) / pass_threshold))
        risk_pct = round(risk_score * 100, 1)
        label = "Pass" if est_grade >= pass_threshold else "Fail"
        return risk_pct, est_grade, label, backing_text

    if preds_prob:
        mean_prob = float(np.mean(preds_prob))
        est_grade = mean_prob * 20.0  # approximate mapping to grade scale
        risk_score = 1.0 - mean_prob
        risk_pct = round(risk_score * 100, 1)
        label = "Pass" if mean_prob >= 0.5 else "Fail"
        return risk_pct, est_grade, label, backing_text

    # If no predictions available
    return 0.0, 0.0, "N/A", backing_text


def plot_risk_gauge(risk_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            number={"suffix": "%"},
            delta={"reference": 50, "increasing": {"color": "#ef4444"}},
            title={"text": "Risk of Failing", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#f97316"},
                "steps": [
                    {"range": [0, 30], "color": "#22c55e"},
                    {"range": [30, 60], "color": "#facc15"},
                    {"range": [60, 100], "color": "#ef4444"},
                ],
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15,23,42,0.0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def plot_performance_meter(est_grade: float, pass_threshold: float) -> go.Figure:
    normalized = max(0.0, min(1.0, est_grade / 20.0))
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=est_grade,
            number={"suffix": "/20"},
            title={"text": "Estimated Final Grade (G3)", "font": {"size": 18}},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 20]},
                "bar": {"color": "#22c55e"},
                "steps": [
                    {"range": [0, pass_threshold], "color": "#1f2937"},
                    {"range": [pass_threshold, 20], "color": "#064e3b"},
                ],
                "threshold": {
                    "line": {"color": "#eab308", "width": 3},
                    "thickness": 0.85,
                    "value": pass_threshold,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15,23,42,0.0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def compute_radar_scores(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build simple heuristic sub-scores from input features for radar chart.
    Returns:
        scores: values for each dimension
        labels: corresponding labels
    """
    row = X.iloc[0]

    # Study discipline (higher is better)
    study_score = np.mean(
        [
            (row.get("studytime", 2) - 1) / 3.0,
            1.0 - min(row.get("absences", 0) / 30.0, 1.0),
            1.0 - min(row.get("failures", 0) / 4.0, 1.0),
        ]
    )

    # Family & support
    fam_support_flags = [
        1.0 if row.get("famsup", "yes") == "yes" else 0.0,
        1.0 if row.get("schoolsup", "no") == "yes" else 0.0,
        1.0 if row.get("higher", "yes") == "yes" else 0.0,
    ]
    fam_score = np.mean(fam_support_flags)

    # Social / distraction (higher = more risk, but invert to get positive score)
    social_raw = np.mean(
        [
            (row.get("goout", 3) - 1) / 4.0,
            (row.get("Dalc", 1) - 1) / 4.0,
            (row.get("Walc", 1) - 1) / 4.0,
        ]
    )
    focus_score = 1.0 - social_raw

    # Health & wellbeing
    health_score = np.mean(
        [
            (row.get("health", 3) - 1) / 4.0,
            1.0 - min(row.get("absences", 0) / 40.0, 1.0),
        ]
    )

    # Prior academic performance
    g1 = row.get("G1", 10)
    g2 = row.get("G2", 10)
    perf_score = np.mean([g1 / 20.0, g2 / 20.0])

    scores = np.array(
        [
            np.clip(study_score, 0.0, 1.0),
            np.clip(fam_score, 0.0, 1.0),
            np.clip(focus_score, 0.0, 1.0),
            np.clip(health_score, 0.0, 1.0),
            np.clip(perf_score, 0.0, 1.0),
        ]
    )
    labels = np.array(
        ["Study Habits", "Support System", "Focus vs Distractions", "Wellbeing", "Past Performance"]
    )
    return scores, labels


def plot_radar(scores: np.ndarray, labels: np.ndarray) -> go.Figure:
    theta = labels.tolist()
    r = (scores * 100.0).tolist()

    fig = go.Figure(
        data=go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name="Profile",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#1f2937",
                linecolor="#4b5563",
            ),
            bgcolor="rgba(15,23,42,0.6)",
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15,23,42,0.0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def render_shap_summary(X: pd.DataFrame, tree_model: Any, preprocessor: Any):
    import shap

    st.markdown(
        '<div class="section-header">🧠 Why did the model predict this?</div>'
        '<div class="section-subtitle">Feature attributions using SHAP values.</div>',
        unsafe_allow_html=True,
    )

    # Prepare one-sample background for fast explanation
    if preprocessor is not None and not hasattr(tree_model, "feature_names_in_"):
        X_proc = preprocessor.transform(X)
    else:
        X_proc = X

    # TreeExplainer for tree-based models
    try:
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_proc)
    except Exception:
        st.info("SHAP explanation is not available for this model configuration.")
        return

    # Binary classification returns list; regression -> ndarray
    if isinstance(shap_values, list):
        # Take positive class (index 1) if available
        sv = shap_values[-1]
    else:
        sv = shap_values

    # Aggregate to a simple bar chart for this single instance
    try:
        if hasattr(tree_model, "feature_names_in_"):
            feat_names = list(tree_model.feature_names_in_)
        else:
            feat_names = [f"f{i}" for i in range(sv.shape[1])]
    except Exception:
        feat_names = [f"f{i}" for i in range(sv.shape[1])]

    contrib = sv[0]
    idx = np.argsort(np.abs(contrib))[::-1][:10]
    contrib_sorted = contrib[idx]
    names_sorted = [feat_names[i] for i in idx]

    bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in contrib_sorted]

    fig = go.Figure(
        go.Bar(
            x=contrib_sorted,
            y=names_sorted,
            orientation="h",
            marker_color=bar_colors,
        )
    )
    fig.update_layout(
        title="Top 10 Feature Contributions",
        xaxis_title="SHAP value (impact on output)",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15,23,42,0.0)",
        plot_bgcolor="rgba(15,23,42,0.0)",
        font=dict(color="#e5e7eb"),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Student Performance Risk Monitor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_sidebar()

    st.markdown(
        """
        ### 🎓 Student Performance & Risk Dashboard

        Predict student performance, classify pass/fail risk, and understand *why* the model made its decision.
        """.strip()
    )

    models = load_models()
    main_model = models["main_model"]
    xgb_model = models["xgb_model"] if st.session_state.get("use_ensemble", True) else None
    preprocessor = models["preprocessor"]
    g3_proxy_bundle = load_g3_proxy_model()

    if main_model is None and xgb_model is None:
        st.error(
            "No trained model found. Please ensure `models/student_performance_model.pkl` "
            "and/or `models/test_model.pkl` exist."
        )
        return

    schema = build_input_schema()
    X = collect_user_inputs(schema)

    st.markdown("---")
    st.markdown(
        '<div class="section-header">🚀 Run Prediction</div>'
        '<div class="section-subtitle">Click the button to estimate final grade and risk level.</div>',
        unsafe_allow_html=True,
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.container()
        with predict_clicked:
            st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
            run = st.button("Predict performance", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            "The model uses a combination of **Random Forest** and optionally **XGBoost** "
            "to provide a robust estimate of the student's outcome."
        )

    if run:
        with st.spinner("Analyzing student profile and running models..."):
            risk_pct, est_grade, label, backing = ensemble_predict(
                X,
                main_model=main_model,
                xgb_model=xgb_model,
                preprocessor=preprocessor,
                pass_threshold=st.session_state.get("pass_threshold", 10),
                g3_proxy_bundle=g3_proxy_bundle,
            )

        st.balloons()

        # Metrics row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Risk Level</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{risk_pct:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-sub">Probability of failing given current profile.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Predicted Outcome</div>', unsafe_allow_html=True)
            outcome_color = "#22c55e" if label == "Pass" else "#f97316"
            st.markdown(
                f'<div class="metric-value" style="color:{outcome_color}">{label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-sub">Classification derived from model predictions.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Model Blend</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value" style="font-size:1.1rem">{backing}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-sub">Models contributing to this prediction.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Charts row: risk gauge + bullet performance + radar
        col_gauge, col_meter, col_radar = st.columns(3)
        with col_gauge:
            st.plotly_chart(plot_risk_gauge(risk_pct), use_container_width=True)

        with col_meter:
            st.plotly_chart(
                plot_performance_meter(
                    est_grade=est_grade,
                    pass_threshold=st.session_state.get("pass_threshold", 10),
                ),
                use_container_width=True,
            )

        with col_radar:
            scores, labels = compute_radar_scores(X)
            st.plotly_chart(plot_radar(scores, labels), use_container_width=True)

        st.markdown("---")

        # SHAP explainability (tree-based only)
        tree_model = get_tree_estimator(main_model or xgb_model)
        if tree_model is not None:
            try:
                render_shap_summary(X, tree_model=tree_model, preprocessor=preprocessor)
            except Exception:
                st.info("SHAP explanations could not be computed for this model.")
        else:
            st.info("No suitable tree-based model found for SHAP explanations.")


if __name__ == "__main__":
    main()


