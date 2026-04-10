"""
RAG Pipeline for Titanic Dataset Insights Chatbot

Builds a knowledge base from project reports and data analysis,
uses in-memory vector search for retrieval, Groq LLM for generation,
and a sandboxed Python executor for live data analysis.
"""

import json
import os
import re
import io
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Knowledge Base Builder
# ---------------------------------------------------------------------------

def _build_dataset_chunks(df: pd.DataFrame) -> list[dict]:
    """Create knowledge chunks from the raw dataset statistics."""
    chunks = []

    # Overall dataset summary — the most commonly asked question
    survived_count = df["Survived"].sum()
    perished_count = len(df) - survived_count
    survival_rate = df["Survived"].mean() * 100
    chunks.append({
        "text": (
            f"Overall Titanic survival statistics: Out of 891 passengers, "
            f"{survived_count} survived ({survival_rate:.1f}%) and {perished_count} perished ({100-survival_rate:.1f}%). "
            f"The overall survival rate is {survival_rate:.1f}%. "
            f"The dataset has {df['Sex'].value_counts()['male']} male passengers (64.8%) "
            f"and {df['Sex'].value_counts()['female']} female passengers (35.2%). "
            f"Class distribution: 1st class {(df['Pclass']==1).sum()} (24.2%), "
            f"2nd class {(df['Pclass']==2).sum()} (20.7%), 3rd class {(df['Pclass']==3).sum()} (55.1%). "
            f"Age ranges from {df['Age'].min():.1f} to {df['Age'].max():.0f} years "
            f"(mean {df['Age'].mean():.1f}, median {df['Age'].median():.0f}, {df['Age'].isna().sum()} missing). "
            f"Fare ranges from {df['Fare'].min():.2f} to {df['Fare'].max():.2f} GBP "
            f"(mean {df['Fare'].mean():.2f}, median {df['Fare'].median():.2f})."
        ),
        "source": "dataset_overview",
        "topic": "overall statistics",
    })

    # Key survival factors summary
    chunks.append({
        "text": (
            f"The key factors influencing Titanic survival, ranked by importance: "
            f"1) Sex (strongest predictor, Pearson r=+0.543): females survived at 74.2% vs males at 18.9%. "
            f"2) Passenger Class (r=-0.338): 1st class 63.0%, 2nd class 47.3%, 3rd class 24.2%. "
            f"3) Fare (r=+0.257): higher fare correlates with survival via class proxy. "
            f"4) Age: children (<12) survived at 58%, seniors (56+) at only 28%. "
            f"5) Family size: small families (2-4) survived at 57.9%, solo travelers at 30.4%. "
            f"6) Embarkation port: Cherbourg 55.4%, Queenstown 39.0%, Southampton 33.7%. "
            "The 'women and children first' evacuation protocol and class-based lifeboat access "
            "are the primary explanations for these patterns."
        ),
        "source": "key_survival_factors",
        "topic": "survival patterns",
    })

    # Survival by sex
    sex_rates = df.groupby("Sex")["Survived"].mean() * 100
    sex_counts = df.groupby("Sex")["Survived"].agg(["sum", "count"])
    chunks.append({
        "text": (
            f"Survival rate by sex: Female passengers had a {sex_rates['female']:.1f}% survival rate "
            f"({int(sex_counts.loc['female','sum'])} of {int(sex_counts.loc['female','count'])} survived), "
            f"while male passengers had only {sex_rates['male']:.1f}% survival rate "
            f"({int(sex_counts.loc['male','sum'])} of {int(sex_counts.loc['male','count'])} survived). "
            "This is the strongest predictor of survival (Pearson r=+0.543). "
            "The 'women and children first' evacuation protocol explains this dramatic difference. "
            "Females were approximately 4x more likely to survive than males."
        ),
        "source": "survival_by_sex",
        "topic": "survival patterns",
    })

    # Survival by class
    class_rates = df.groupby("Pclass")["Survived"].mean() * 100
    class_counts = df.groupby("Pclass")["Survived"].agg(["sum", "count"])
    chunks.append({
        "text": (
            f"Survival rate by passenger class: "
            f"1st class {class_rates[1]:.1f}% ({int(class_counts.loc[1,'sum'])} of {int(class_counts.loc[1,'count'])} survived), "
            f"2nd class {class_rates[2]:.1f}% ({int(class_counts.loc[2,'sum'])} of {int(class_counts.loc[2,'count'])} survived), "
            f"3rd class {class_rates[3]:.1f}% ({int(class_counts.loc[3,'sum'])} of {int(class_counts.loc[3,'count'])} survived). "
            "There is a strong class gradient — first-class passengers were 2.6x more likely to survive than third-class. "
            "This reflects privileged access to lifeboats and cabin location (upper decks for higher classes). "
            f"Average fares: 1st class {df[df['Pclass']==1]['Fare'].mean():.2f}, "
            f"2nd class {df[df['Pclass']==2]['Fare'].mean():.2f}, "
            f"3rd class {df[df['Pclass']==3]['Fare'].mean():.2f} GBP."
        ),
        "source": "survival_by_class",
        "topic": "survival patterns",
    })

    # Sex x Class interaction
    sex_class = df.groupby(["Sex", "Pclass"])["Survived"].mean() * 100
    chunks.append({
        "text": (
            "Sex and class interaction — the most powerful combined predictor. Survival rates: "
            f"Female 1st class: {sex_class[('female', 1)]:.1f}%, "
            f"Female 2nd class: {sex_class[('female', 2)]:.1f}%, "
            f"Female 3rd class: {sex_class[('female', 3)]:.1f}%, "
            f"Male 1st class: {sex_class[('male', 1)]:.1f}%, "
            f"Male 2nd class: {sex_class[('male', 2)]:.1f}%, "
            f"Male 3rd class: {sex_class[('male', 3)]:.1f}%. "
            "First and second-class women had near-perfect survival (>92%). "
            "Even third-class women (50%) far exceeded first-class men (~37%). "
            "This interaction is the single most informative feature combination for the model."
        ),
        "source": "sex_class_interaction",
        "topic": "survival patterns",
    })

    # Age analysis
    df_age = df.dropna(subset=["Age"])
    age_bins = pd.cut(df_age["Age"], bins=[0, 12, 18, 35, 60, 100],
                      labels=["Child (0-12)", "Teen (13-18)", "Young Adult (19-35)",
                              "Middle Age (36-60)", "Senior (61+)"])
    age_survival = df_age.groupby(age_bins, observed=False)["Survived"].agg(["mean", "sum", "count"])
    chunks.append({
        "text": (
            "Survival by age group: "
            + "; ".join(f"{grp}: {row['mean']*100:.0f}% ({int(row['sum'])} of {int(row['count'])})"
                       for grp, row in age_survival.iterrows())
            + f". Mean age: {df_age['Age'].mean():.1f} years, median: {df_age['Age'].median():.0f} years. "
            f"Youngest passenger: {df['Age'].min():.2f} years (infant). Oldest: {df['Age'].max():.0f} years. "
            f"177 passengers (19.9%) have missing age data, imputed using median by class and sex group. "
            "Children had the highest survival rate, confirming 'women and children first'. "
            "Age is a weaker individual predictor (r=-0.077) but important in combination with sex."
        ),
        "source": "survival_by_age",
        "topic": "survival patterns",
    })

    # Embarkation
    emb_map = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    emb_rates = df.dropna(subset=["Embarked"]).groupby("Embarked")["Survived"].mean() * 100
    emb_counts = df["Embarked"].value_counts()
    chunks.append({
        "text": (
            "Survival by port of embarkation: "
            + "; ".join(f"{emb_map.get(p, p)} ({p}): {emb_rates[p]:.1f}% survival ({emb_counts[p]} passengers, {emb_counts[p]/len(df)*100:.1f}% of total)"
                       for p in ["S", "C", "Q"])
            + ". 2 passengers have missing embarkation data (filled with Southampton). "
            "Cherbourg's higher survival rate is explained by a higher proportion of 1st class passengers boarding there. "
            "Southampton was by far the largest port with 72.3% of all passengers."
        ),
        "source": "survival_by_embarked",
        "topic": "survival patterns",
    })

    # Family size
    df_fam = df.copy()
    df_fam["FamilySize"] = df_fam["SibSp"] + df_fam["Parch"] + 1
    fam_survival = df_fam.groupby("FamilySize")["Survived"].agg(["mean", "count"])
    alone_rate = df_fam[df_fam["FamilySize"] == 1]["Survived"].mean() * 100
    family_rate = df_fam[df_fam["FamilySize"] > 1]["Survived"].mean() * 100
    chunks.append({
        "text": (
            f"Family size analysis: {(df_fam['FamilySize']==1).sum()} passengers ({(df_fam['FamilySize']==1).mean()*100:.1f}%) traveled alone. "
            f"Solo travelers had {alone_rate:.1f}% survival vs {family_rate:.1f}% for those with family. "
            "Survival by family size: "
            + "; ".join(f"size {int(fs)}: {row['mean']*100:.0f}% ({int(row['count'])} passengers)"
                       for fs, row in fam_survival.iterrows())
            + ". Small families (2-4 members) had the best survival (~57.9%). "
            "Large families (5+) had only ~16.1% survival, possibly due to difficulty evacuating together. "
            "SibSp = siblings/spouses aboard; Parch = parents/children aboard."
        ),
        "source": "family_analysis",
        "topic": "survival patterns",
    })

    # Fare analysis
    fare_by_class = df.groupby("Pclass")["Fare"].describe()
    chunks.append({
        "text": (
            f"Fare analysis: Mean fare {df['Fare'].mean():.2f} GBP, median {df['Fare'].median():.2f} GBP. "
            f"Extremely right-skewed (skewness 4.79). Fare quartiles: "
            f"25th={df['Fare'].quantile(0.25):.2f}, 50th={df['Fare'].quantile(0.5):.2f}, "
            f"75th={df['Fare'].quantile(0.75):.2f}. "
            f"15 passengers paid zero fare (possibly crew or honorary tickets). "
            f"Maximum fare: {df['Fare'].max():.2f} GBP (3 passengers, luxury suites). "
            f"Fare by class — 1st: mean {fare_by_class.loc[1,'mean']:.2f} (median {fare_by_class.loc[1,'50%']:.2f}), "
            f"2nd: mean {fare_by_class.loc[2,'mean']:.2f} (median {fare_by_class.loc[2,'50%']:.2f}), "
            f"3rd: mean {fare_by_class.loc[3,'mean']:.2f} (median {fare_by_class.loc[3,'50%']:.2f}). "
            "Higher fares correlate with survival (Pearson r=+0.257) via class proxy. "
            "344 passengers shared ticket numbers, so fares may be per-party."
        ),
        "source": "fare_analysis",
        "topic": "dataset statistics",
    })

    # Cabin and missing data
    has_cabin = df["Cabin"].notna()
    chunks.append({
        "text": (
            f"Data quality and missing values: "
            f"Cabin: {df['Cabin'].isna().sum()} missing (77.1%) — passengers WITH cabin info had "
            f"{df[has_cabin]['Survived'].mean()*100:.1f}% survival vs {df[~has_cabin]['Survived'].mean()*100:.1f}% without. "
            "Having a cabin record is a strong proxy for 1st class. "
            f"Age: {df['Age'].isna().sum()} missing (19.9%) — imputed using median by Pclass+Sex group. "
            f"Embarked: 2 missing (0.2%) — filled with 'S' (Southampton, the mode). "
            "0 duplicate rows. Ticket: 681 unique values across 891 rows (high cardinality). "
            f"Cabin has 147 unique values among non-null rows. "
            "Quality issues: Fare skewness (4.79), 15 zero-fare passengers, "
            "ticket sharing (344 passengers share ticket numbers)."
        ),
        "source": "data_quality",
        "topic": "data quality",
    })

    # Correlations
    chunks.append({
        "text": (
            "Feature correlations with Survived (target): "
            "Sex (female=1): +0.543 (strongest), Pclass: -0.338, Fare: +0.257, "
            "Embarked_C: +0.168, Embarked_S: -0.156, Parch: +0.082, Age: -0.077, SibSp: -0.035. "
            "Notable feature correlations: Pclass vs Fare: -0.550 (moderate collinearity, both encode SES). "
            "No critical multicollinearity issues (no pairs >|0.80| outside one-hot encoding artifacts). "
            "The low correlation of Age with survival (-0.077) is misleading — the relationship is non-linear "
            "(children benefit, seniors disadvantaged) and only becomes powerful when combined with Sex."
        ),
        "source": "correlations",
        "topic": "dataset statistics",
    })

    # Column reference
    chunks.append({
        "text": (
            "Titanic dataset column reference (12 columns): "
            "PassengerId (int64) — unique ID, not a feature; "
            "Survived (int64) — TARGET: 0=died, 1=survived; "
            "Pclass (int64) — ticket class: 1=First, 2=Second, 3=Third; "
            "Name (object) — full name, used to extract Title; "
            "Sex (object) — male/female; "
            "Age (float64) — age in years, 19.9% missing; "
            "SibSp (int64) — number of siblings/spouses aboard; "
            "Parch (int64) — number of parents/children aboard; "
            "Ticket (object) — ticket number, 681 unique; "
            "Fare (float64) — ticket price in GBP; "
            "Cabin (object) — cabin number, 77.1% missing; "
            "Embarked (object) — port: C=Cherbourg, Q=Queenstown, S=Southampton."
        ),
        "source": "column_reference",
        "topic": "dataset schema",
    })

    return chunks


def _build_model_chunks() -> list[dict]:
    """Create knowledge chunks from model evaluation results."""
    chunks = []

    metrics_path = ROOT / "reports" / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        cm = metrics["confusion_matrix"]
        total_test = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        chunks.append({
            "text": (
                f"Best model performance — XGBoost classifier on the test set ({total_test} samples): "
                f"Accuracy: {metrics['accuracy']:.1%} (correctly classified {int(total_test*metrics['accuracy'])} of {total_test}). "
                f"Precision: {metrics['precision']:.1%} (of predicted survivors, {metrics['precision']:.1%} actually survived). "
                f"Recall: {metrics['recall']:.1%} (of actual survivors, {metrics['recall']:.1%} were correctly identified). "
                f"F1 Score: {metrics['f1']:.1%} (harmonic mean of precision and recall). "
                f"ROC-AUC: {metrics['roc_auc']:.3f} (area under ROC curve, 1.0 is perfect). "
                f"Confusion matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}. "
                f"The model is better at predicting non-survival ({cm[0][0]/(cm[0][0]+cm[0][1])*100:.0f}% specificity) "
                f"than survival ({cm[1][1]/(cm[1][0]+cm[1][1])*100:.0f}% sensitivity). "
                "This asymmetry reflects the class imbalance (62% perished, 38% survived)."
            ),
            "source": "model_performance",
            "topic": "model results",
        })

        importance = metrics.get("feature_importance", {})
        if importance:
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            chunks.append({
                "text": (
                    "Feature importance from XGBoost model (gain-based): "
                    + "; ".join(f"{name}: {score:.4f}" for name, score in sorted_features)
                    + ". Top 5: " + ", ".join(f"{name} ({score:.3f})" for name, score in sorted_features[:5])
                    + ". Title_encoded is #1 because it jointly encodes sex, age group, and social status "
                    "(Mr vs Mrs vs Miss vs Master). Pclass (#2) captures class effect. "
                    "Sex_binary (#3) is the raw sex signal. Deck_encoded uses cabin deck letter as location proxy. "
                    "FamilySize captures the non-linear family size effect. "
                    "Features with zero importance (FareBin, IsAlone, FamilyType, AgeBin, IsChild, IsMother) "
                    "are redundant with other features — the tree model doesn't use them when better alternatives exist."
                ),
                "source": "feature_importance",
                "topic": "model results",
            })

    comparison_path = ROOT / "models" / "model_comparison.csv"
    if comparison_path.exists():
        comp_df = pd.read_csv(comparison_path)
        best = comp_df.loc[comp_df["test_roc_auc"].idxmax()]
        chunks.append({
            "text": (
                f"Model comparison — 6 classifiers trained with stratified 5-fold cross-validation: "
                + ". ".join(
                    f"{row['model']}: CV AUC={row['cv_roc_auc']:.3f}, Test AUC={row['test_roc_auc']:.3f}, "
                    f"Test Accuracy={row['test_accuracy']:.1%}, Test F1={row['test_f1']:.3f}, "
                    f"Fit Time={row['fit_time']:.3f}s"
                    for _, row in comp_df.iterrows()
                )
                + f". {best['model']} achieved the highest test AUC. "
                "Random Forest and Logistic Regression also performed competitively. "
                "All models use the same 18 engineered features with StandardScaler preprocessing."
            ),
            "source": "model_comparison",
            "topic": "model results",
        })

    return chunks


def _build_methodology_chunks() -> list[dict]:
    """Create knowledge chunks about the methodology."""
    return [
        {
            "text": (
                "Feature engineering methodology: 18 features were created from the raw 12 columns. "
                "Key transformations: Title extraction from Name via regex (Mr, Mrs, Miss, Master, Rare groups), "
                "binary sex encoding (female=1), Pclass as integer, HasCabin binary flag, "
                "Deck letter extraction from Cabin (ordinal by survival rate), "
                "FamilySize = SibSp + Parch + 1, IsAlone flag, FamilyType (Solo/Small/Large), "
                "log1p(Fare) to normalize skew, FareBin (quartile-based), "
                "Age imputation by Pclass+Sex group median, AgeBin (5 groups), IsChild (age<13), "
                "IsMother (female + has children + age>18 + not Miss), "
                "Sex x Pclass one-hot interaction (6 features), Embarked one-hot (3 features). "
                "Dropped columns: PassengerId, Name, Ticket, raw Cabin, raw SibSp, raw Parch."
            ),
            "source": "feature_engineering",
            "topic": "methodology",
        },
        {
            "text": (
                "Data leakage prevention: The train/test split (80/20, stratified by Survived) happens "
                "BEFORE any feature engineering. Age imputation medians and Fare quartile bin boundaries "
                "are computed on the training set only and applied to the test set. "
                "This is implemented via a custom TitanicFeatureEngineer scikit-learn transformer in src/processing.py. "
                "The full pipeline (features + scaler) is saved as models/pipeline.joblib. "
                "All features use only information available at boarding time — no target leakage."
            ),
            "source": "leakage_prevention",
            "topic": "methodology",
        },
        {
            "text": (
                "Model training pipeline: 6 classifiers trained — Logistic Regression, Random Forest, "
                "Gradient Boosting, XGBoost, LightGBM, and SVM. Stratified 5-fold cross-validation. "
                "Primary metric: ROC-AUC; secondary: F1-score. "
                "Class imbalance (62/38) handled with class_weight='balanced' where supported. "
                "StandardScaler applied to all features. Best model (XGBoost) saved as models/best_model.joblib. "
                "Implementation in src/model.py. API endpoint at api/app.py (/predict). "
                "Dashboard at dashboard/app.py (Streamlit, port 8501). "
                "Docker deployment via docker-compose.yml (API on 8000, dashboard on 8501)."
            ),
            "source": "training_pipeline",
            "topic": "methodology",
        },
        {
            "text": (
                "Historical context of the Titanic disaster: RMS Titanic sank April 15, 1912 "
                "after hitting an iceberg on its maiden voyage from Southampton to New York. "
                "Of ~2,224 passengers and crew, >1,500 died — one of history's deadliest maritime disasters. "
                "The ship had lifeboats for only ~1,178 people (about half capacity). "
                "The 'women and children first' evacuation protocol is clearly reflected in the data. "
                "Class-based access to lifeboats (1st class cabins were closer to the boat deck) "
                "explains the strong class survival gradient. "
                "This dataset contains 891 of the 1,309 passengers (crew are excluded)."
            ),
            "source": "historical_context",
            "topic": "domain knowledge",
        },
        {
            "text": (
                "Project architecture and files: "
                "Data: Titanic-Dataset.csv (891 rows, 12 columns). "
                "Source code: src/processing.py (TitanicFeatureEngineer transformer), "
                "src/model.py (training pipeline), src/ml_utils.py (reusable ML utilities), "
                "src/rag_pipeline.py (this RAG chatbot). "
                "Models: models/best_model.joblib (XGBoost), models/pipeline.joblib (preprocessing), "
                "models/feature_names.json, models/model_comparison.csv. "
                "Reports: reports/eda_report.md, reports/evaluation_metrics.json, "
                "reports/feature_engineering_plan.md, plus 14 PNG visualizations. "
                "API: api/app.py (FastAPI with /predict endpoint). "
                "Dashboard: dashboard/app.py (Streamlit with 5 tabs). "
                "Deployment: Dockerfile, docker-compose.yml."
            ),
            "source": "project_architecture",
            "topic": "project structure",
        },
    ]


def build_knowledge_base() -> list[dict]:
    """Build the complete knowledge base from all project artifacts."""
    df = pd.read_csv(ROOT / "Titanic-Dataset.csv")

    chunks = []
    chunks.extend(_build_dataset_chunks(df))
    chunks.extend(_build_model_chunks())
    chunks.extend(_build_methodology_chunks())

    for i, chunk in enumerate(chunks):
        chunk["id"] = f"chunk_{i:03d}"

    return chunks


# ---------------------------------------------------------------------------
# In-Memory Vector Store
# ---------------------------------------------------------------------------

class KnowledgeStore:
    """In-memory vector store using numpy cosine similarity."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray, model):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model

    def query(self, question: str, k: int = 5) -> list[dict]:
        q_emb = self.model.encode([question], normalize_embeddings=True)
        scores = (self.embeddings @ q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            {
                "id": self.chunks[i]["id"],
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "topic": self.chunks[i]["topic"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]


def build_store() -> KnowledgeStore:
    """Build the in-memory knowledge store with embeddings."""
    from sentence_transformers import SentenceTransformer

    chunks = build_knowledge_base()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return KnowledgeStore(chunks, embeddings, model)


def retrieve(question: str, store: KnowledgeStore, k: int = 5) -> list[dict]:
    return store.query(question, k=k)


# ---------------------------------------------------------------------------
# Sandboxed Python Code Execution
# ---------------------------------------------------------------------------

def execute_code(code: str, df: pd.DataFrame) -> dict:
    """Execute Python code in a sandboxed environment with access to the dataset.

    Returns dict with 'success', 'output', and optionally 'error'.
    """
    allowed_globals = {
        "__builtins__": {
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "int": int, "float": float, "str": str, "bool": bool, "list": list,
            "dict": dict, "tuple": tuple, "set": set, "type": type,
            "isinstance": isinstance, "print": print, "format": format,
            "True": True, "False": False, "None": None,
        },
        "pd": pd,
        "np": np,
        "df": df.copy(),
    }

    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            exec(code, allowed_globals)
        output = stdout_capture.getvalue()
        return {"success": True, "output": output.strip() if output.strip() else "Code executed successfully (no print output)."}
    except Exception as e:
        return {"success": False, "output": stdout_capture.getvalue(), "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Generation (Groq LLM)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI data analyst assistant for the Titanic survival prediction project.
You answer stakeholder questions using the provided context from our data analysis and model results.

Guidelines:
- Be precise with numbers and statistics — cite them from the context.
- If the context contains the answer, use it directly. Do not say "the context doesn't contain" when it does.
- Provide actionable insights when relevant.
- Keep answers concise but thorough (2-4 paragraphs for complex questions).
- When discussing model performance, explain what the metrics mean in plain language.
- When asked about survival patterns, explain the underlying reasons (e.g., evacuation protocols, class privilege).
"""

SYSTEM_PROMPT_WITH_CODE = """You are an AI data analyst assistant for the Titanic survival prediction project.
You have two capabilities:
1. A knowledge base with pre-computed insights about the dataset and model
2. A Python sandbox where you can run code on the actual Titanic DataFrame (variable `df`)

Available in the sandbox: `df` (pandas DataFrame with all 891 rows), `pd` (pandas), `np` (numpy).

When to use code: If the question requires a specific computation, filtering, or aggregation that isn't directly in the context, write Python code to compute it. Use print() to output results.

IMPORTANT: When you want to run code, output EXACTLY this format (no extra text before the code block):
```python
# your code here
print(result)
```

After seeing the code output, provide your final answer incorporating both the context AND code results.

Guidelines:
- Be precise with numbers — cite from context or compute with code.
- Keep answers concise but thorough.
- When a question can be answered from context alone, just answer it directly without code.
- When you need code, write clean pandas operations and always print() the results.
"""


def _extract_code_block(text: str) -> Optional[str]:
    """Extract Python code block from LLM response."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def generate(question: str, context_chunks: list[dict],
             api_key: Optional[str] = None,
             model: str = "llama-3.3-70b-versatile",
             enable_code: bool = False,
             df: Optional[pd.DataFrame] = None) -> dict:
    """Generate an answer using Groq LLM with retrieved context.

    Returns dict with 'answer', and optionally 'code' and 'code_output'.
    """
    from groq import Groq

    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return {"answer": "Error: GROQ_API_KEY not found. Please set it in your .env file."}

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = SYSTEM_PROMPT_WITH_CODE if enable_code else SYSTEM_PROMPT

    user_message = f"""Context from our Titanic data analysis:
---
{context_text}
---

Stakeholder question: {question}

Answer based on the context above."""
    if enable_code:
        user_message += " If you need a specific computation not in the context, write a Python code block."

    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, max_tokens=1024,
    )
    first_response = response.choices[0].message.content

    # Check if LLM wants to run code
    code = _extract_code_block(first_response) if enable_code else None

    if code and df is not None:
        code_result = execute_code(code, df)
        code_output = code_result["output"]
        if not code_result["success"] and code_result.get("error"):
            code_output += f"\nError: {code_result['error']}"

        # Second LLM call with code results
        messages.append({"role": "assistant", "content": first_response})
        messages.append({
            "role": "user",
            "content": f"Code execution result:\n```\n{code_output}\n```\n\nNow provide your final answer to the stakeholder incorporating these results. Do not include code blocks in your final answer.",
        })

        response2 = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=1024,
        )
        return {
            "answer": response2.choices[0].message.content,
            "code": code,
            "code_output": code_output,
        }

    return {"answer": first_response}


# ---------------------------------------------------------------------------
# End-to-End RAG
# ---------------------------------------------------------------------------

def rag(question: str, store: Optional[KnowledgeStore] = None,
        api_key: Optional[str] = None, k: int = 5,
        enable_code: bool = False) -> dict:
    """End-to-end RAG: retrieve context, optionally run code, generate answer.

    Returns dict with 'answer', 'sources', 'context_chunks', and optionally 'code', 'code_output'.
    """
    if store is None:
        store = build_store()

    context_chunks = retrieve(question, store, k=k)

    df = None
    if enable_code:
        df = pd.read_csv(ROOT / "Titanic-Dataset.csv")

    result = generate(question, context_chunks, api_key=api_key,
                      enable_code=enable_code, df=df)

    result["sources"] = list({c["source"] for c in context_chunks})
    result["context_chunks"] = context_chunks

    return result


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    print("Building knowledge base...")
    store = build_store()
    print(f"Knowledge base: {len(store.chunks)} chunks, embeddings {store.embeddings.shape}")

    test_questions = [
        ("What was the overall survival rate?", False),
        ("What is the average age of 1st class female survivors?", True),
        ("How does our model perform?", False),
    ]

    for q, use_code in test_questions:
        print(f"\nQ: {q} {'[with code]' if use_code else ''}")
        result = rag(q, store=store, enable_code=use_code)
        print(f"A: {result['answer'][:300]}...")
        if result.get("code"):
            print(f"Code: {result['code']}")
            print(f"Output: {result['code_output']}")
        print(f"Sources: {result['sources']}")
