# Feature Engineering Plan — Titanic Survival Prediction

**Generated**: 2026-03-03
**Agent**: feature-engineering-analyst
**Target**: Survived (binary classification)
**Source**: EDA findings from eda-analyst_report.json

---

## 1. Data Assessment

| Property | Value |
|----------|-------|
| Rows | 891 |
| Target | Survived (binary: 0=died, 1=survived) |
| Class balance | 61.6% died / 38.4% survived |
| Granularity | One row per passenger |
| Domain | Historical maritime disaster |

**Key EDA context leveraged:**
- Sex is dominant predictor (Pearson r=0.543); female survival 74.2% vs male 18.9%
- Pclass strongly correlated (r=-0.339); 1st class 63.0% vs 3rd class 24.2%
- Age: 177 missing (19.9%) — stratified imputation required
- Cabin: 687 missing (77.1%) — replace with HasCabin binary + Deck letter
- Fare: heavy right skew (skewness=4.79) — log1p transform
- FamilySize pattern: medium families 2-4 survive most, solo and large families least

---

## 2. Feature Engineering Plan

### Feature Table

| Feature Name | Source Columns | Transformation | Expected Importance | Leakage Risk |
|---|---|---|---|---|
| Sex_binary | Sex | Binary encode (female=1) | Very High | None |
| Title_encoded | Name | Regex extract + group + ordinal | Very High | None |
| Sex_Pclass_interaction | Sex, Pclass | String concat + one-hot | Very High | None |
| Pclass | Pclass | Keep as-is (integer) | High | None |
| HasCabin | Cabin | Binary (notna=1) | High | None |
| Fare_log1p | Fare | log1p transform | High | None |
| FamilyType_encoded | SibSp, Parch | Bin into Solo/Small/Large + ordinal | High | None |
| Age_imputed | Age, Pclass, Sex | Grouped median imputation (train-only) | Medium | Low |
| IsChild | Age | Binary (Age_imputed < 13) | Medium | None |
| AgeBin_encoded | Age | Cut into 5 age groups + ordinal | Medium | None |
| IsMother | Sex, Parch, Age, Name | Multi-condition interaction binary | Medium | None |
| Deck_encoded | Cabin | Regex extract first char + ordinal | Medium | None |
| FamilySize | SibSp, Parch | Arithmetic: SibSp + Parch + 1 | Medium | None |
| IsAlone | SibSp, Parch | Binary (SibSp+Parch == 0) | Medium | None |
| FareBin_encoded | Fare | Quartile bins + ordinal (train-only) | Medium | Low |
| Embarked_C | Embarked | One-hot (impute 2 missing with 'S') | Low | None |
| Embarked_Q | Embarked | One-hot | Low | None |
| Embarked_S | Embarked | One-hot | Low | None |
| Title_IsFemale | Name | Binary from female title list | Low | None |

### Columns to Drop

| Column | Reason |
|--------|--------|
| PassengerId | Row identifier — no predictive signal; memorises training rows |
| Name | Identifier; Title extracted and used instead |
| Ticket | High-cardinality alphanumeric; signal proxied by Pclass and Fare |
| Cabin | 77.1% missing; replaced by HasCabin and Deck_encoded |
| SibSp | Subsumed into FamilySize, IsAlone, FamilyType_encoded |
| Parch | Subsumed into family features; IsMother preserves parenthood signal |

---

## 3. Imputation Strategy

| Column | Missing | Strategy | Notes |
|--------|---------|----------|-------|
| Age | 177 (19.9%) | Grouped median by Pclass+Sex | Fit medians on training data only; apply same values to test |
| Embarked | 2 (0.2%) | Mode = 'S' (Southampton) | Negligible impact |
| Cabin | 687 (77.1%) | Do not impute — replace column | HasCabin=0 and Deck='Unknown' encode missingness |

**Age imputation group medians (from training data):**

| Pclass | Sex | Median Age |
|--------|-----|-----------|
| 1 | female | 35.0 |
| 1 | male | 40.0 |
| 2 | female | 28.0 |
| 2 | male | 30.0 |
| 3 | female | 21.5 |
| 3 | male | 25.0 |

---

## 4. Encoding Strategy

| Feature | Method | Rationale |
|---------|--------|-----------|
| Sex | Binary (female=1, male=0) | Two values, lossless |
| Title | Ordinal by survival rate (Mr=0, Master/Rare_Male=1, Miss=2, Mrs/Rare_Female=3) | Survival order: Mr < Master < Miss < Mrs |
| Embarked | One-hot (3 columns) | Nominal; low cardinality |
| Deck | Ordinal by survival (Unknown=0 through B=7) | 8 levels; ordinal preserves survival ordering |
| FamilyType | Ordinal (Solo=0, Small=1, Large=2) | 3 levels; size order meaningful |
| FareBin | Ordinal (Low=0, Mid=1, High=2, VHigh=3) | Monotonic with survival |
| AgeBin | Ordinal (Child=0 through Senior=4) | Age order meaningful |
| Pclass | Keep as integer 1/2/3 | Tree models handle directly; use one-hot for logistic regression |
| Sex_Pclass | One-hot (6 categories) | Nominal interaction; no spurious ordinal assumption |

---

## 5. Implementation Code

```python
"""
feature_engineering.py
Leakage-safe feature engineering pipeline for Titanic survival prediction.

Usage:
    # Training:
    X_train_fe, encoders = create_features(X_train, fit=True)

    # Inference / test:
    X_test_fe, _ = create_features(X_test, fit=False, encoders=encoders)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Title extraction helpers
# ---------------------------------------------------------------------------

TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare_Male",
    "Rev": "Rare_Male",
    "Col": "Rare_Male",
    "Major": "Rare_Male",
    "Capt": "Rare_Male",
    "Jonkheer": "Rare_Male",
    "Don": "Rare_Male",
    "Sir": "Rare_Male",
    "Lady": "Rare_Female",
    "Countess": "Rare_Female",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Mlle": "Miss",
}

TITLE_ORDINAL = {
    "Mr": 0,
    "Rare_Male": 1,
    "Master": 1,
    "Miss": 2,
    "Mrs": 3,
    "Rare_Female": 3,
}

FEMALE_TITLES = {"Mrs", "Miss", "Ms", "Mme", "Mlle", "Lady", "Countess", "Rare_Female"}

DECK_ORDINAL = {
    "Unknown": 0,
    "T": 0,
    "G": 1,
    "F": 2,
    "A": 3,
    "C": 4,
    "E": 5,
    "D": 6,
    "B": 7,
}

FAMILY_TYPE_ORDINAL = {"Solo": 0, "Small": 1, "Large": 2}

AGE_BIN_ORDINAL = {"Child": 0, "Teen": 1, "YoungAdult": 2, "MiddleAge": 3, "Senior": 4}

FARE_BIN_ORDINAL = {"Low": 0, "Mid": 1, "High": 2, "VHigh": 3}


def _extract_title(name_series: pd.Series) -> pd.Series:
    """Extract title from Name column and map to consolidated groups."""
    raw = name_series.str.extract(r" ([A-Za-z]+)\.", expand=False)
    return raw.map(TITLE_MAPPING).fillna("Rare_Male")


def _extract_deck(cabin_series: pd.Series) -> pd.Series:
    """Extract deck letter from Cabin; fill missing with 'Unknown'."""
    return cabin_series.str.extract(r"^([A-Z])", expand=False).fillna("Unknown")


def _compute_age_medians(df: pd.DataFrame) -> dict:
    """Compute age imputation medians grouped by Pclass and Sex. Fit on training data only."""
    medians = {}
    for (pclass, sex), group in df.groupby(["Pclass", "Sex"]):
        medians[(pclass, sex)] = group["Age"].median()
    return medians


def _impute_age(df: pd.DataFrame, medians: dict) -> pd.Series:
    """Apply pre-computed age medians to fill missing values."""
    age = df["Age"].copy()
    mask = age.isnull()
    global_median = np.nanmedian(list(medians.values()))
    for idx in df[mask].index:
        key = (df.at[idx, "Pclass"], df.at[idx, "Sex"])
        age.at[idx] = medians.get(key, global_median)
    return age


def _compute_fare_bins(fare_series: pd.Series) -> tuple:
    """Compute quartile bin boundaries from training Fare. Returns (labels, bins)."""
    _, bins = pd.qcut(fare_series, q=4, retbins=True, duplicates="drop")
    bins[0] = -np.inf  # ensure all values are captured
    bins[-1] = np.inf
    return bins


def _apply_fare_bins(fare_series: pd.Series, bins) -> pd.Series:
    """Apply pre-computed fare bin boundaries."""
    labels = ["Low", "Mid", "High", "VHigh"][: len(bins) - 1]
    return pd.cut(fare_series, bins=bins, labels=labels).astype(str)


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def create_features(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None,
) -> tuple:
    """
    Create ML features from raw Titanic data.

    Args:
        df:       Input DataFrame (raw columns from CSV)
        fit:      True = compute and store encoders (training).
                  False = apply pre-fitted encoders (test/inference).
        encoders: Required when fit=False. Dict returned from a prior fit=True call.

    Returns:
        Tuple of (feature_df: pd.DataFrame, encoders: dict)
    """
    if not fit and encoders is None:
        raise ValueError("encoders dict is required when fit=False")

    df = df.copy()
    out = pd.DataFrame(index=df.index)
    enc = {} if fit else encoders.copy()

    # ------------------------------------------------------------------
    # Step 1: Impute missing values
    # ------------------------------------------------------------------

    # Embarked: mode impute (2 missing)
    df["Embarked"] = df["Embarked"].fillna("S")

    # Age: grouped median imputation
    if fit:
        enc["age_medians"] = _compute_age_medians(df)
    df["Age_imputed"] = _impute_age(df, enc["age_medians"])

    # ------------------------------------------------------------------
    # Step 2: Title extraction
    # ------------------------------------------------------------------
    title_raw = _extract_title(df["Name"])
    out["Title_encoded"] = title_raw.map(TITLE_ORDINAL).fillna(0).astype(int)
    out["Title_IsFemale"] = title_raw.isin(FEMALE_TITLES).astype(int)

    # ------------------------------------------------------------------
    # Step 3: Sex encoding
    # ------------------------------------------------------------------
    out["Sex_binary"] = (df["Sex"] == "female").astype(int)

    # ------------------------------------------------------------------
    # Step 4: Pclass (keep as integer)
    # ------------------------------------------------------------------
    out["Pclass"] = df["Pclass"].astype(int)

    # ------------------------------------------------------------------
    # Step 5: Cabin features
    # ------------------------------------------------------------------
    out["HasCabin"] = df["Cabin"].notna().astype(int)
    out["Deck_encoded"] = _extract_deck(df["Cabin"]).map(DECK_ORDINAL).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Step 6: Family features
    # ------------------------------------------------------------------
    family_size = df["SibSp"] + df["Parch"] + 1
    out["FamilySize"] = family_size
    out["IsAlone"] = (family_size == 1).astype(int)

    def _family_type(fs):
        if fs <= 1:
            return "Solo"
        elif fs <= 4:
            return "Small"
        else:
            return "Large"

    out["FamilyType_encoded"] = family_size.map(_family_type).map(FAMILY_TYPE_ORDINAL).astype(int)

    # ------------------------------------------------------------------
    # Step 7: Fare features
    # ------------------------------------------------------------------
    out["Fare_log1p"] = np.log1p(df["Fare"])

    if fit:
        enc["fare_bins"] = _compute_fare_bins(df["Fare"])
    fare_bin_labels = _apply_fare_bins(df["Fare"], enc["fare_bins"])
    out["FareBin_encoded"] = fare_bin_labels.map(FARE_BIN_ORDINAL).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Step 8: Age features
    # ------------------------------------------------------------------
    out["Age_imputed"] = df["Age_imputed"]
    out["IsChild"] = (df["Age_imputed"] < 13).astype(int)

    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ["Child", "Teen", "YoungAdult", "MiddleAge", "Senior"]
    age_bin_col = pd.cut(
        df["Age_imputed"], bins=age_bins, labels=age_labels, right=True
    ).astype(str)
    out["AgeBin_encoded"] = age_bin_col.map(AGE_BIN_ORDINAL).fillna(2).astype(int)

    # ------------------------------------------------------------------
    # Step 9: Interaction features
    # ------------------------------------------------------------------

    # Sex x Pclass: one-hot encode 6 combinations
    sex_pclass = df["Sex"] + "_" + df["Pclass"].astype(str)
    sp_dummies = pd.get_dummies(sex_pclass, prefix="SexPclass")
    expected_sp_cols = [
        "SexPclass_female_1", "SexPclass_female_2", "SexPclass_female_3",
        "SexPclass_male_1", "SexPclass_male_2", "SexPclass_male_3",
    ]
    for col in expected_sp_cols:
        out[col] = sp_dummies.get(col, pd.Series(0, index=df.index)).astype(int)

    # IsMother
    out["IsMother"] = (
        (df["Sex"] == "female")
        & (df["Parch"] > 0)
        & (df["Age_imputed"] > 18)
        & (title_raw != "Miss")
    ).astype(int)

    # ------------------------------------------------------------------
    # Step 10: Embarked one-hot
    # ------------------------------------------------------------------
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    for level in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
        out[level] = embarked_dummies.get(level, pd.Series(0, index=df.index)).astype(int)

    return out, enc


# ---------------------------------------------------------------------------
# Convenience: full pipeline from raw CSV to train/test feature matrices
# ---------------------------------------------------------------------------

def build_train_test_features(
    csv_path: str,
    target_col: str = "Survived",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Load raw CSV, split into train/test (stratified), apply leakage-safe
    feature engineering, and return ready-to-use arrays.

    Returns dict with keys:
        X_train, X_test, y_train, y_test, feature_names, encoders
    """
    df = pd.read_csv(csv_path)

    # Drop columns that are never used as features
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", target_col]
    # Keep raw SibSp and Parch for now — create_features uses them before dropping
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratified split BEFORE any feature engineering
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit on training data only
    X_train, encoders = create_features(X_train_raw, fit=True)

    # Apply fitted encoders to test
    X_test, _ = create_features(X_test_raw, fit=False, encoders=encoders)

    # Align columns (in case of unseen categories in test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X_train.columns),
        "encoders": encoders,
    }


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = build_train_test_features(
        csv_path="/Users/maximolorenzoylosada/Documents/2026-03-03 DEMO/Titanic-Dataset.csv"
    )
    print("Training features shape:", result["X_train"].shape)
    print("Test features shape:    ", result["X_test"].shape)
    print("Feature names:")
    for i, name in enumerate(result["feature_names"], 1):
        print(f"  {i:2d}. {name}")
```

---

## 6. Leakage Audit Results

### Cleared (no leakage risk)
All 17 of 19 features passed the leakage audit:

- Sex_binary, Title_encoded, Title_IsFemale: derived purely from passenger attributes available at boarding time
- HasCabin, Deck_encoded: available at boarding
- FamilySize, IsAlone, FamilyType_encoded, IsMother: derived from boarding manifest
- Fare_log1p: ticket price known at boarding
- IsChild: derived from Age (imputed)
- AgeBin_encoded, Age_imputed: imputation fit on training fold only
- IsChild, IsMother: boolean combinations of known attributes
- Embarked one-hot: available at boarding
- Sex_Pclass interaction: combination of known attributes

### Low Risk — Requires Train-Only Fitting

| Feature | Risk | Mitigation |
|---------|------|------------|
| Age_imputed | Low | Group medians must be computed on training data only and re-computed for each CV fold |
| FareBin_encoded | Low | Quartile bin boundaries must be fit on training data only (use `pd.qcut` on train, `pd.cut` on test) |

### No Features Dropped for Leakage

No features were flagged as actual leakage. All transformations use information available at boarding time.

---

## 7. Feature Selection Recommendations

### Priority Ranking (expected importance order)

| Rank | Feature | Basis |
|------|---------|-------|
| 1 | Sex_binary | Pearson r=0.543; largest single predictor |
| 2 | Title_encoded | Encodes Sex + class + marital status jointly |
| 3 | Sex_Pclass interaction | female_1=96.8% vs male_3=13.5% survival |
| 4 | Pclass | Pearson r=-0.339; strong class effect |
| 5 | HasCabin | 66.7% vs 30.0% survival; class proxy |
| 6 | Fare_log1p | Pearson r=0.257; within-class wealth signal |
| 7 | FamilyType_encoded | Non-linear: Small=57.9%, Solo=30.4%, Large=16.1% |
| 8 | Age_imputed | Continuous age after imputation |
| 9 | IsChild | 57.9% child survival; policy-driven |
| 10 | AgeBin_encoded | Captures age-group survival pattern |
| 11 | IsAlone | 60.3% passengers solo; 30.4% survival |
| 12 | IsMother | Targeted mother subgroup |
| 13 | Deck_encoded | Deck B/D/E ~75% survival |
| 14 | FareBin_encoded | Non-parametric fare complement |
| 15 | Embarked features | Moderate effect; C vs S ~22pp difference |
| 16 | FamilySize | Continuous; redundant with FamilyType for linear models |
| 17 | Title_IsFemale | Redundancy check on Sex_binary |

### Minimal Recommended Feature Set (12 features)

For a lean, interpretable model:

```
Sex_binary, Title_encoded, Pclass, Sex_Pclass_interaction (6 cols),
HasCabin, Fare_log1p, FamilyType_encoded, Age_imputed, IsChild, Embarked_C
```

### Full Feature Set (26 columns after one-hot expansion)

Recommended for gradient boosting (XGBoost, LightGBM, RandomForest):

```
Sex_binary, Title_encoded, Title_IsFemale, Pclass,
HasCabin, Deck_encoded,
FamilySize, IsAlone, FamilyType_encoded,
Fare_log1p, FareBin_encoded,
Age_imputed, AgeBin_encoded, IsChild, IsMother,
SexPclass_female_1/2/3, SexPclass_male_1/2/3,
Embarked_C, Embarked_Q, Embarked_S
```

---

## 8. Model Recommendations

Based on the feature engineering design:

- **Logistic Regression**: Use minimal feature set (12 features); apply StandardScaler to Age_imputed and Fare_log1p; one-hot Pclass
- **Random Forest / XGBoost**: Use full feature set (26 columns); tree models handle ordinal encodings and one-hot natively without scaling
- **Class imbalance**: Use `class_weight='balanced'` or `scale_pos_weight` in XGBoost; stratified cross-validation
- **Validation**: Stratified 5-fold CV; primary metric ROC-AUC; secondary F1-score
- **Feature importance**: Run permutation importance after first model fit to validate ranking

---

*Report generated by feature-engineering-analyst agent. Depends on: eda-analyst. Enables: developer, mlops-engineer.*
