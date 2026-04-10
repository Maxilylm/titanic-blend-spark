"""
Titanic Survival Prediction Dashboard

Interactive Streamlit dashboard for exploring Titanic data,
viewing model performance, and making survival predictions.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Load environment variables (local .env or Streamlit Cloud secrets)
load_dotenv(ROOT / ".env")
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="\u2693",
    layout="wide",
)

# --- Color constants ---
SURVIVED_COLOR = "#E8634A"
PERISHED_COLOR = "#566573"
TEAL = "#117A65"
NAVY = "#0D1B2A"
OCEAN = "#1B4F72"
GOLD = "#D4AC0D"
SURVIVAL_COLORS = {0: PERISHED_COLOR, 1: SURVIVED_COLOR}


# --- Data loading ---
@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "Titanic-Dataset.csv")
    # Feature engineering for display
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young Adult", "Middle Age", "Senior"],
    )
    df["FareGroup"] = pd.qcut(
        df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )
    return df


@st.cache_data
def load_metrics():
    metrics_path = ROOT / "reports" / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model():
    model_path = ROOT / "models" / "best_model.joblib"
    pipeline_path = ROOT / "models" / "pipeline.joblib"
    if model_path.exists() and pipeline_path.exists():
        return joblib.load(model_path), joblib.load(pipeline_path)
    return None, None


@st.cache_data
def load_comparison():
    path = ROOT / "models" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# Load all data
df = load_data()
metrics = load_metrics()
model, pipeline = load_model()
comparison_df = load_comparison()

# --- Sidebar ---
st.sidebar.title("\u2693 Titanic Dashboard")
st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("Filters")
sex_filter = st.sidebar.multiselect("Sex", df["Sex"].unique().tolist(), default=df["Sex"].unique().tolist())
pclass_filter = st.sidebar.multiselect("Passenger Class", sorted(df["Pclass"].unique().tolist()), default=sorted(df["Pclass"].unique().tolist()))
embarked_map = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
embarked_options = df["Embarked"].dropna().unique().tolist()
embarked_filter = st.sidebar.multiselect(
    "Port of Embarkation",
    embarked_options,
    default=embarked_options,
    format_func=lambda x: embarked_map.get(x, x),
)
age_range = st.sidebar.slider(
    "Age Range", 0, int(df["Age"].max()) + 1, (0, int(df["Age"].max()) + 1),
)

# Apply filters
df_filtered = df[
    (df["Sex"].isin(sex_filter))
    & (df["Pclass"].isin(pclass_filter))
    & (df["Embarked"].isin(embarked_filter))
    & ((df["Age"].between(age_range[0], age_range[1])) | (df["Age"].isna()))
]

st.sidebar.metric("Filtered Passengers", f"{len(df_filtered):,}", f"{len(df_filtered) - 891:+,}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Analysis", "Model Results", "Prediction", "Blend Spark Assistant"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("Titanic Dataset Overview")

    if len(df_filtered) == 0:
        st.warning("No passengers match your filters. Adjust the sidebar filters.")
        st.stop()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    survival_rate = df_filtered["Survived"].mean() * 100
    avg_age = df_filtered["Age"].mean()
    avg_fare = df_filtered["Fare"].mean()

    col1.metric("Total Passengers", f"{len(df_filtered):,}", f"of 891 total")
    col2.metric("Survival Rate", f"{survival_rate:.1f}%", f"vs 38.4% overall")
    col3.metric("Average Age", f"{avg_age:.1f}" if pd.notna(avg_age) else "N/A", f"vs 29.7 overall")
    col4.metric("Average Fare", f"\u00a3{avg_fare:.2f}", f"vs \u00a332.20 overall")

    st.markdown("---")

    # Row 2: Survival distribution + by class
    col_left, col_right = st.columns(2)

    with col_left:
        survived_counts = df_filtered["Survived"].value_counts()
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Perished", "Survived"],
            values=[survived_counts.get(0, 0), survived_counts.get(1, 0)],
            hole=0.5,
            marker_colors=[PERISHED_COLOR, SURVIVED_COLOR],
        )])
        fig_donut.update_layout(title="Survival Distribution", height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        pclass_survival = df_filtered.groupby("Pclass")["Survived"].mean().reset_index()
        pclass_survival["Rate"] = (pclass_survival["Survived"] * 100).round(1)
        fig_pclass = px.bar(
            pclass_survival, x="Pclass", y="Rate",
            color_discrete_sequence=[TEAL],
            labels={"Pclass": "Passenger Class", "Rate": "Survival Rate (%)"},
            title="Survival Rate by Class",
            text="Rate",
        )
        fig_pclass.update_layout(height=350, yaxis_range=[0, 100])
        fig_pclass.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_pclass, use_container_width=True)

    # Row 3: Sex, Age groups, Embarked
    col1, col2, col3 = st.columns(3)

    with col1:
        sex_survival = df_filtered.groupby("Sex")["Survived"].mean().reset_index()
        sex_survival["Rate"] = (sex_survival["Survived"] * 100).round(1)
        fig_sex = px.bar(
            sex_survival, x="Sex", y="Rate",
            color="Sex", color_discrete_map={"female": SURVIVED_COLOR, "male": PERISHED_COLOR},
            title="Survival by Sex", text="Rate",
        )
        fig_sex.update_layout(height=300, showlegend=False, yaxis_range=[0, 100])
        fig_sex.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_sex, use_container_width=True)

    with col2:
        age_survival = df_filtered.dropna(subset=["AgeGroup"]).groupby("AgeGroup", observed=False)["Survived"].mean().reset_index()
        age_survival["Rate"] = (age_survival["Survived"] * 100).round(1)
        fig_age = px.bar(
            age_survival, x="AgeGroup", y="Rate",
            color_discrete_sequence=[OCEAN],
            title="Survival by Age Group", text="Rate",
        )
        fig_age.update_layout(height=300, yaxis_range=[0, 100])
        fig_age.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_age, use_container_width=True)

    with col3:
        emb_survival = df_filtered.dropna(subset=["Embarked"]).groupby("Embarked")["Survived"].mean().reset_index()
        emb_survival["Rate"] = (emb_survival["Survived"] * 100).round(1)
        emb_survival["Port"] = emb_survival["Embarked"].map(embarked_map)
        fig_emb = px.bar(
            emb_survival, x="Port", y="Rate",
            color_discrete_sequence=[TEAL],
            title="Survival by Port", text="Rate",
        )
        fig_emb.update_layout(height=300, yaxis_range=[0, 100])
        fig_emb.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_emb, use_container_width=True)

    # Key findings
    st.info(
        "**Key Findings:** Sex is the dominant predictor (female 74.2% vs male 18.9%). "
        "1st class passengers had 63% survival vs 24.2% for 3rd class. "
        "Children under 12 survived at 58%, confirming 'women and children first'."
    )


# ==================== TAB 2: ANALYSIS ====================
with tab2:
    st.header("Exploratory Data Analysis")

    if len(df_filtered) == 0:
        st.warning("No passengers match your filters.")
        st.stop()

    st.info("177 passengers (19.9%) have unknown age and are excluded from age-filtered views.")

    # Age distribution by survival
    col_left, col_right = st.columns(2)

    with col_left:
        df_age = df_filtered.dropna(subset=["Age"])
        fig_age_dist = px.histogram(
            df_age, x="Age", color="Survived",
            color_discrete_map={0: PERISHED_COLOR, 1: SURVIVED_COLOR},
            barmode="overlay", opacity=0.7,
            title="Age Distribution by Survival",
            labels={"Survived": "Outcome"},
            category_orders={"Survived": [0, 1]},
        )
        fig_age_dist.update_layout(height=400)
        st.plotly_chart(fig_age_dist, use_container_width=True)

    with col_right:
        # Fare distribution (log scale)
        fig_fare = px.histogram(
            df_filtered, x="Fare", color="Survived",
            color_discrete_map={0: PERISHED_COLOR, 1: SURVIVED_COLOR},
            barmode="overlay", opacity=0.7, log_x=True,
            title="Fare Distribution (Log Scale) by Survival",
            labels={"Survived": "Outcome"},
        )
        fig_fare.update_layout(height=400)
        st.plotly_chart(fig_fare, use_container_width=True)

    # Sex x Pclass heatmap
    col_left, col_right = st.columns(2)

    with col_left:
        heatmap_data = df_filtered.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()
        fig_heat = px.imshow(
            (heatmap_data * 100).round(1),
            text_auto=".1f",
            color_continuous_scale=[[0, PERISHED_COLOR], [1, SURVIVED_COLOR]],
            title="Survival Rate (%): Sex x Class",
            labels={"x": "Passenger Class", "y": "Sex", "color": "Rate (%)"},
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        # Family size vs survival
        fam_survival = df_filtered.groupby("FamilySize")["Survived"].agg(["mean", "count"]).reset_index()
        fam_survival["Rate"] = (fam_survival["mean"] * 100).round(1)
        fig_fam = px.bar(
            fam_survival, x="FamilySize", y="Rate",
            color_discrete_sequence=[OCEAN],
            title="Survival Rate by Family Size",
            text="Rate",
        )
        fig_fam.update_layout(height=350, yaxis_range=[0, 100])
        fig_fam.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_fam, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
    corr_matrix = df_filtered[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix.round(2), text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Matrix",
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Fare by class
    col_left, col_right = st.columns(2)
    with col_left:
        fig_fare_class = px.box(
            df_filtered, x="Pclass", y="Fare", color="Pclass",
            color_discrete_sequence=[GOLD, OCEAN, PERISHED_COLOR],
            title="Fare Distribution by Class",
        )
        fig_fare_class.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_fare_class, use_container_width=True)

    with col_right:
        # Survival by alone status
        alone_survival = df_filtered.groupby("IsAlone")["Survived"].mean().reset_index()
        alone_survival["Status"] = alone_survival["IsAlone"].map({0: "With Family", 1: "Alone"})
        alone_survival["Rate"] = (alone_survival["Survived"] * 100).round(1)
        fig_alone = px.bar(
            alone_survival, x="Status", y="Rate",
            color_discrete_sequence=[TEAL],
            title="Survival: Alone vs With Family", text="Rate",
        )
        fig_alone.update_layout(height=350, yaxis_range=[0, 100])
        fig_alone.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_alone, use_container_width=True)


# ==================== TAB 3: MODEL RESULTS ====================
with tab3:
    st.header("Model Performance")

    if metrics is None:
        st.warning("No model evaluation metrics found. Run model training first.")
    else:
        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Precision", f"{metrics['precision']:.1%}")
        col3.metric("Recall", f"{metrics['recall']:.1%}")
        col4.metric("F1 Score", f"{metrics['f1']:.1%}")

        st.markdown("---")

        # Confusion matrix + ROC curve
        col_left, col_right = st.columns(2)

        with col_left:
            cm = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["Predicted Perished", "Predicted Survived"],
                y=["Actual Perished", "Actual Survived"],
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            fpr = metrics["roc_curve"]["fpr"]
            tpr = metrics["roc_curve"]["tpr"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"Model (AUC={metrics['roc_auc']:.3f})",
                line=dict(color=SURVIVED_COLOR, width=2),
                fill="tozeroy", fillcolor="rgba(232,99,74,0.2)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(color="gray", dash="dash"),
            ))
            fig_roc.update_layout(
                title="ROC Curve", height=400,
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance
        st.subheader("Feature Importance")
        importance = metrics.get("feature_importance", {})
        if importance:
            imp_df = pd.DataFrame(
                sorted(importance.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"],
            )
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=[[0, PERISHED_COLOR], [1, TEAL]],
                title="Feature Importance",
            )
            fig_imp.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

        # Model comparison table
        if comparison_df is not None:
            st.subheader("Model Comparison")
            display_cols = ["model", "cv_roc_auc", "cv_f1", "test_roc_auc", "test_f1", "test_accuracy", "fit_time"]
            display_df = comparison_df[display_cols].copy()
            display_df.columns = ["Model", "CV AUC", "CV F1", "Test AUC", "Test F1", "Test Accuracy", "Fit Time (s)"]
            st.dataframe(
                display_df.style.highlight_max(subset=["CV AUC", "CV F1", "Test AUC", "Test F1", "Test Accuracy"], color="#117A65", props="color: white;"),
                use_container_width=True, hide_index=True,
            )

        # Classification report
        with st.expander("Full Classification Report"):
            st.code(metrics.get("classification_report", "Not available"))


# ==================== TAB 4: PREDICTION ====================
with tab4:
    st.header("Predict Survival")

    if model is None or pipeline is None:
        st.warning("No trained model found. Run model training first.")
    else:
        col_form, col_result = st.columns([5, 7])

        with col_form:
            st.subheader("Passenger Details")
            with st.form("prediction_form"):
                pclass = st.selectbox(
                    "Passenger Class", [1, 2, 3], index=2,
                    help="1st class: 63% survival, 2nd: 47%, 3rd: 24%",
                )
                sex = st.radio("Sex", ["male", "female"], horizontal=True,
                               help="Sex is the strongest predictor of survival")
                age = st.number_input("Age", min_value=0.5, max_value=80.0, value=29.7,
                                      help="Children under 12 had 58% survival rate")
                sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=8, value=0)
                parch = st.number_input("Parents/Children aboard", min_value=0, max_value=6, value=0)
                fare = st.number_input("Fare (GBP)", min_value=0.0, max_value=600.0, value=14.45,
                                       help="Median fare; luxury fares exceeded GBP 500")
                embarked = st.selectbox(
                    "Port of Embarkation",
                    ["S", "C", "Q"],
                    format_func=lambda x: embarked_map.get(x, x),
                    help="Cherbourg: 55.4% survival rate",
                )

                submitted = st.form_submit_button("Predict Survival", type="primary")

            family_size = sibsp + parch + 1
            st.metric("Family Size", family_size)

        with col_result:
            if submitted:
                # Build input DataFrame matching raw Titanic format
                # Need a Name with a title for feature engineering
                title_prefix = "Mrs." if sex == "female" and age >= 18 else "Miss." if sex == "female" else "Master." if age < 13 else "Mr."
                input_data = pd.DataFrame([{
                    "PassengerId": 0,
                    "Pclass": pclass,
                    "Name": f"Passenger, {title_prefix} Test",
                    "Sex": sex,
                    "Age": age,
                    "SibSp": sibsp,
                    "Parch": parch,
                    "Ticket": "TEST",
                    "Fare": fare,
                    "Cabin": None,
                    "Embarked": embarked,
                }])

                # Transform through pipeline
                features = pipeline.named_steps["features"]
                X_input = features.transform(input_data)
                scaler = pipeline.named_steps["scaler"]
                X_scaled = scaler.transform(X_input)

                # Predict
                prob = model.predict_proba(X_scaled)[0][1]
                survived = prob >= 0.5

                # Display result
                st.subheader("Prediction Result")
                if survived:
                    st.success(f"**SURVIVED** (Probability: {prob:.1%})")
                else:
                    st.error(f"**DID NOT SURVIVE** (Probability: {prob:.1%})")

                # Probability bar
                st.progress(float(prob))
                st.caption(f"Survival probability: {prob:.1%} | Threshold: 50%")

                # Factor analysis
                st.subheader("Key Factors")
                factors = []
                if sex == "female":
                    factors.append(("+ Female passengers had 74.2% survival rate", True))
                else:
                    factors.append(("- Male passengers had only 18.9% survival rate", False))

                if pclass == 1:
                    factors.append(("+ 1st class passengers had 63.0% survival rate", True))
                elif pclass == 2:
                    factors.append(("~ 2nd class passengers had 47.3% survival rate", True))
                else:
                    factors.append(("- 3rd class passengers had only 24.2% survival rate", False))

                if age < 13:
                    factors.append(("+ Children had 58.0% survival rate", True))
                elif age > 60:
                    factors.append(("- Seniors had lower survival rates", False))

                if family_size == 1:
                    factors.append(("- Solo travelers had 30.4% survival", False))
                elif family_size <= 4:
                    factors.append(("+ Small families had 57.9% survival", True))
                else:
                    factors.append(("- Large families had only 16.1% survival", False))

                for text, positive in factors:
                    if positive:
                        st.markdown(f":green[{text}]")
                    else:
                        st.markdown(f":red[{text}]")

                # Similar passengers
                st.subheader("Similar Historical Passengers")
                similar = df.dropna(subset=["Age"]).copy()
                similar["age_diff"] = (similar["Age"] - age).abs()
                similar = similar[
                    (similar["Sex"] == sex) & (similar["Pclass"] == pclass)
                ].nsmallest(5, "age_diff")

                if len(similar) > 0:
                    display_similar = similar[["Name", "Age", "Survived", "Fare", "Embarked"]].copy()
                    display_similar["Survived"] = display_similar["Survived"].map({0: "No", 1: "Yes"})
                    st.dataframe(display_similar, use_container_width=True, hide_index=True)
                else:
                    st.info("No similar passengers found in the dataset.")
            else:
                st.info("Fill in the passenger details and click **Predict Survival** to see results.")


# ==================== TAB 5: BLEND SPARK ASSISTANT ====================
with tab5:

    # --- Blend brand CSS injection ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap');

    /* Blend Spark tab scoped styles */
    .blend-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .blend-header h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        font-size: 1.75rem;
        color: #053057;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .blend-header .spark-badge {
        background: linear-gradient(135deg, #00EDED 0%, #A2F3F3 100%);
        color: #053057;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        white-space: nowrap;
    }
    .blend-subtitle {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.9rem;
        color: #314550;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .blend-divider {
        height: 2px;
        background: linear-gradient(90deg, #00EDED 0%, #A2F3F3 40%, transparent 100%);
        border: none;
        margin: 16px 0 20px 0;
    }
    .blend-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 6px;
    }
    .blend-section-label {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #00EDED;
        margin-bottom: 10px;
    }
    .blend-powered {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.72rem;
        color: #314550;
        opacity: 0.7;
        text-align: center;
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid rgba(0, 237, 237, 0.15);
    }
    .blend-powered span {
        color: #00EDED;
        font-weight: 500;
    }

    /* Style the Streamlit chat input area */
    [data-testid="stChatInput"] textarea {
        font-family: 'Montserrat', sans-serif !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #053057 !important;
        color: #00EDED !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #00EDED !important;
        color: #053057 !important;
    }

    /* Style suggested question buttons inside this tab */
    div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"] {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.78rem !important;
        border-color: rgba(0, 237, 237, 0.3) !important;
        color: #053057 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"]:hover {
        border-color: #00EDED !important;
        background-color: rgba(0, 237, 237, 0.08) !important;
        color: #053057 !important;
    }

    /* Style the toggle */
    div[data-testid="stVerticalBlockBorderWrapper"] label[data-testid="stWidgetLabel"] p {
        font-family: 'Montserrat', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Blend symbol SVG (inline, recolored to Neon Turquoise) ---
    blend_symbol_svg = """<svg width="38" height="20" viewBox="0 0 205 110" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path fill-rule="evenodd" clip-rule="evenodd" d="M9.72 38.42L10.45 70.74V70.8C10.45 73.58 11.92 76.79 15.79 80.27C19.66 83.75 25.53 87.11 33.21 90.03C48.52 95.85 70 99.56 93.96 99.56C117.93 99.56 139.41 95.86 154.72 90.03C162.39 87.11 168.26 83.76 172.13 80.28C176.01 76.79 177.47 73.58 177.47 70.8C177.47 68.01 176.01 64.8 172.13 61.32C171.75 60.97 171.36 60.63 170.94 60.3C167 62.71 162.45 64.88 157.44 66.78C140.77 73.13 118.07 76.95 93.23 76.95C76.55 76.95 64.36 76.16 52.48 73.48C40.65 70.81 29.35 66.33 14.4 59.25C13.59 58.91 12.13 58.04 11.58 56.08C10.99 53.99 11.98 52.43 12.22 52.07C12.57 51.55 12.93 51.22 13.07 51.1C13.25 50.94 13.41 50.83 13.5 50.77C13.8 50.56 14.07 50.42 14.14 50.39L14.14 50.38C14.35 50.28 14.58 50.18 14.75 50.1C15.14 49.93 15.67 49.71 16.32 49.46C17.62 48.95 19.51 48.23 21.87 47.39C26.6 45.71 33.27 43.48 41.03 41.26C56.44 36.84 76.5 32.32 93.96 32.32C118.8 32.32 141.5 36.14 158.18 42.49C162.83 44.25 167.08 46.25 170.82 48.46C171.02 48.29 171.21 48.12 171.4 47.95C175.27 44.47 176.74 41.26 176.74 38.47C176.74 35.69 175.27 32.47 171.4 28.99C167.53 25.51 161.66 22.15 153.98 19.24C138.67 13.41 117.19 9.71 93.23 9.71C69.26 9.71 47.78 13.41 32.47 19.24C24.8 22.15 18.93 25.51 15.06 28.99C11.21 32.45 9.74 35.65 9.72 38.42ZM178.85 54.29C183.4 49.89 186.46 44.56 186.46 38.47C186.46 31.97 182.98 26.34 177.91 21.78C172.83 17.21 165.76 13.33 157.44 10.16C140.77 3.82 118.07 0 93.23 0C68.39 0 45.69 3.82 29.01 10.16C20.69 13.33 13.63 17.21 8.55 21.78C3.48 26.34 0 31.97 0 38.47V38.53L0.73 70.85C0.75 77.33 4.23 82.94 9.29 87.48C14.36 92.05 21.43 95.93 29.75 99.1C46.42 105.44 69.12 109.27 93.96 109.27C118.8 109.27 141.5 105.45 158.18 99.11C166.5 95.94 173.56 92.05 178.64 87.49C183.71 82.93 187.19 77.29 187.19 70.8C187.19 64.39 183.81 58.82 178.85 54.29ZM161.46 54.45C159.4 53.45 157.15 52.48 154.72 51.56C139.41 45.73 117.93 42.03 93.96 42.03C77.88 42.03 58.87 46.24 43.71 50.59C38.08 52.2 33.03 53.82 28.93 55.21C38.59 59.42 46.53 62.19 54.61 64.01C65.47 66.45 76.82 67.24 93.23 67.24C117.19 67.24 138.67 63.53 153.98 57.71C156.71 56.68 159.2 55.59 161.46 54.45Z" fill="#00EDED"/>
    </svg>"""

    # --- Header with Blend symbol ---
    st.markdown(f"""
    <div class="blend-header">
        {blend_symbol_svg}
        <h1>Blend Spark Assistant</h1>
        <div class="spark-badge">AI-Powered</div>
    </div>
    <p class="blend-subtitle">
        Ask questions about the Titanic dataset, survival patterns, model performance, or methodology.<br>
        Powered by RAG retrieval and live Python analysis.
    </p>
    <div class="blend-divider"></div>
    """, unsafe_allow_html=True)

    # Lazy-load RAG knowledge store
    @st.cache_resource(show_spinner="Loading Blend Spark knowledge base...")
    def init_rag():
        from rag_pipeline import build_store
        return build_store()

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # Settings row
    col_toggle, col_clear = st.columns([4, 1])
    with col_toggle:
        enable_code = st.toggle(
            "Enable Python sandbox",
            value=True,
            help="When enabled, the assistant can write and execute Python code on the dataset for precise, ad-hoc analysis.",
        )
    with col_clear:
        if st.session_state.chat_messages:
            if st.button("Clear Chat", type="secondary", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

    # Suggested questions
    st.markdown('<div class="blend-section-label">Suggested Questions</div>', unsafe_allow_html=True)
    suggested = [
        "What was the overall survival rate and what factors most influenced it?",
        "How does our best model perform? What are its strengths and weaknesses?",
        "What is the survival pattern for women vs men across different classes?",
        "What is the average fare paid by survivors vs non-survivors in each class?",
        "What data quality issues exist in the dataset?",
    ]

    cols = st.columns(3)
    for i, q in enumerate(suggested[:3]):
        with cols[i]:
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_question = q

    cols2 = st.columns(2)
    for i, q in enumerate(suggested[3:]):
        with cols2[i]:
            if st.button(q, key=f"suggested_{i+3}", use_container_width=True):
                st.session_state.pending_question = q

    st.markdown('<div class="blend-divider"></div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask Blend Spark about the Titanic data, model, or methodology...")
    if user_input:
        st.session_state.pending_question = user_input

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("figure_json"):
                import plotly.io as pio
                fig = pio.from_json(msg["figure_json"])
                st.plotly_chart(fig, use_container_width=True)
            if msg.get("code"):
                with st.expander("Code executed"):
                    st.code(msg["code"], language="python")
                    if msg.get("code_output"):
                        st.text(msg["code_output"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    st.write(", ".join(msg["sources"]))

    # Process pending question
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            error_msg = "GROQ_API_KEY not found. Please add it to your `.env` file in the project root."
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Blend Spark is analyzing..."):
                    try:
                        store = init_rag()
                        from rag_pipeline import rag as rag_query
                        result = rag_query(
                            question, store=store, api_key=api_key,
                            enable_code=enable_code,
                        )
                        st.markdown(result["answer"])
                        if result.get("figure"):
                            st.plotly_chart(result["figure"], use_container_width=True)
                        if result.get("code"):
                            with st.expander("Code executed"):
                                st.code(result["code"], language="python")
                                if result.get("code_output"):
                                    st.text(result["code_output"])
                        if result["sources"]:
                            with st.expander("Sources used"):
                                st.write(", ".join(result["sources"]))
                        # Store figure as JSON for chat history replay
                        fig_json = result["figure"].to_json() if result.get("figure") else None
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "code": result.get("code"),
                            "code_output": result.get("code_output"),
                            "figure_json": fig_json,
                        })
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

    # Footer
    st.markdown("""
    <div class="blend-powered">
        <span>Blend</span> Spark Assistant &mdash; We blend in so you stand out.
    </div>
    """, unsafe_allow_html=True)
