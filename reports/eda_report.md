# Titanic Dataset - Exploratory Data Analysis Report

**Generated:** 2026-03-03
**Dataset:** Titanic-Dataset.csv
**Target Variable:** Survived (Binary Classification)
**Data Fingerprint:** `86c46fd2a53e9360...3394f4da`

---

## 1. Dataset Overview

| Property        | Value                                    |
|-----------------|------------------------------------------|
| Rows            | 891                                      |
| Columns         | 12                                       |
| Memory Usage    | ~285.61 KB                               |
| Task Type       | Binary Classification                    |
| Target Variable | Survived (0 = No, 1 = Yes)               |
| Source          | Titanic-Dataset.csv                      |

### Column Reference

| Column      | Type    | Description                                       |
|-------------|---------|---------------------------------------------------|
| PassengerId | int64   | Unique passenger identifier (ID, not a feature)   |
| Survived    | int64   | TARGET: 0 = Did not survive, 1 = Survived         |
| Pclass      | int64   | Ticket class: 1=First, 2=Second, 3=Third          |
| Name        | object  | Full passenger name (high cardinality)            |
| Sex         | object  | Passenger sex: male / female                      |
| Age         | float64 | Age in years (19.9% missing)                      |
| SibSp       | int64   | Number of siblings/spouses aboard                 |
| Parch       | int64   | Number of parents/children aboard                 |
| Ticket      | object  | Ticket number (681 unique values)                 |
| Fare        | float64 | Ticket price in USD (heavily right-skewed)        |
| Cabin       | object  | Cabin number (77.1% missing)                      |
| Embarked    | object  | Port: C=Cherbourg, Q=Queenstown, S=Southampton    |

---

## 2. Data Quality Report

### Missing Values

| Column   | Missing Count | Missing %  | Severity | Recommended Action                           |
|----------|---------------|------------|----------|----------------------------------------------|
| Cabin    | 687           | 77.1%      | HIGH     | Create HasCabin binary flag; extract Deck letter |
| Age      | 177           | 19.9%      | MEDIUM   | Impute with median by Pclass/Sex group        |
| Embarked | 2             | 0.2%       | LOW      | Impute with mode (Southampton)               |

### Duplicate Records

**0 duplicate rows found.** Dataset is clean of duplicates.

### Quality Issues Detected by ml_utils

| Issue              | Column   | Severity | Detail                                      |
|--------------------|----------|----------|---------------------------------------------|
| Majority missing   | Cabin    | HIGH     | 77.1% of values are null                   |
| Highly skewed      | Fare     | MEDIUM   | Skewness = 4.79 (heavy right tail)         |
| High cardinality   | Ticket   | MEDIUM   | 681 unique values across 891 rows          |
| High cardinality   | Cabin    | MEDIUM   | 147 unique values (among non-null rows)    |

### Additional Red Flags

| Observation            | Detail                                                              |
|------------------------|---------------------------------------------------------------------|
| Zero fares             | 15 passengers paid $0 — possible crew or honorary tickets           |
| Max fare outlier       | $512.33 (3 passengers) — likely suite-level first-class tickets     |
| Ticket sharing         | 344 passengers share a ticket number — fare may be per-party        |
| PassengerId as feature | Sequential ID — must be excluded from all models                    |

---

## 3. Statistical Summary

### Numerical Features

| Statistic | Age    | Fare    | SibSp  | Parch  |
|-----------|--------|---------|--------|--------|
| Count     | 714    | 891     | 891    | 891    |
| Mean      | 29.70  | 32.20   | 0.52   | 0.38   |
| Median    | 28.00  | 14.45   | 0.00   | 0.00   |
| Std Dev   | 14.53  | 49.69   | 1.10   | 0.81   |
| Min       | 0.42   | 0.00    | 0.00   | 0.00   |
| Max       | 80.00  | 512.33  | 8.00   | 6.00   |
| Skewness  | +0.39  | +4.79   | +3.70  | +2.75  |
| Kurtosis  | +0.18  | +33.40  | +17.88 | +9.78  |

**Key observations:**
- Age is approximately normally distributed (mean 29.7 years; youngest passenger ~5 months old)
- Fare is extremely right-skewed — log1p transformation strongly recommended
- SibSp and Parch are zero-inflated; majority of passengers traveled alone

### Categorical Features

**Survived (Target):** 549 (61.6%) did not survive, 342 (38.4%) survived — moderate imbalance

**Sex:** 577 male (64.8%), 314 female (35.2%)

**Pclass:** 491 third-class (55.1%), 216 first-class (24.2%), 184 second-class (20.7%)

**Embarked:** 644 Southampton (72.3%), 168 Cherbourg (18.9%), 77 Queenstown (8.6%), 2 missing

---

## 4. Outlier Detection (IQR Method)

| Feature | IQR Bounds       | Outliers | Max Value | Assessment                             |
|---------|------------------|----------|-----------|----------------------------------------|
| Age     | [-6.69, 64.81]   | 11       | 80.00     | Legitimate (elderly passengers)        |
| Fare    | [-26.72, 65.63]  | 116      | 512.33    | Legitimate — log transform recommended |
| SibSp   | [-1.50, 2.50]    | 46       | 8.00      | Legitimate large families              |
| Parch   | [0.00, 0.00]     | 213      | 6.00      | IQR=0 artifact — not true outliers     |

All apparent outliers are legitimate data points. Do not remove them.

---

## 5. Correlation Analysis

### Feature Correlations with Survived

| Feature      | Correlation | Direction | Interpretation                                           |
|--------------|-------------|-----------|----------------------------------------------------------|
| Sex (F=1)    | +0.543      | Positive  | STRONGEST: females far more likely to survive            |
| Pclass       | -0.338      | Negative  | Higher class number = lower survival                     |
| Fare         | +0.257      | Positive  | Higher fare = higher class = higher survival             |
| Embarked_C   | +0.168      | Positive  | Cherbourg passengers slightly more likely to survive     |
| Embarked_S   | -0.156      | Negative  | Southampton passengers slightly less likely to survive   |
| Parch        | +0.082      | Positive  | Small positive effect of traveling with family           |
| Age          | -0.077      | Negative  | Older passengers slightly less likely to survive         |
| SibSp        | -0.035      | Near-zero | Non-linear relationship — analyze by group               |

### Highly Correlated Feature Pairs

| Pair                      | Correlation | Concern                                        |
|---------------------------|-------------|------------------------------------------------|
| Pclass vs Fare            | -0.550      | Moderate collinearity — both encode SES status |
| Embarked_C vs Embarked_S  | -0.778      | Expected artifact of one-hot encoding          |

No critical multicollinearity issues found (no pairs exceed |0.80| outside encoding artifacts).

---

## 6. Target Variable Analysis

### Overall Survival Rate: 38.4%

### By Sex

| Sex    | Survival Rate | vs. Average |
|--------|---------------|-------------|
| Female | 74.2%         | +35.8 pp    |
| Male   | 18.9%         | -19.5 pp    |

"Women and children first" protocol is clearly reflected. Females were ~4x more likely to survive.

### By Passenger Class

| Class | Survival Rate | vs. Average |
|-------|---------------|-------------|
| 1st   | 63.0%         | +24.6 pp    |
| 2nd   | 47.3%         | +8.9 pp     |
| 3rd   | 24.2%         | -14.2 pp    |

Strong class gradient — third-class passengers had nearly 3x lower odds than first-class.

### Sex x Class Interaction (Critical)

| Sex    | 1st Class | 2nd Class | 3rd Class |
|--------|-----------|-----------|-----------|
| Female | ~96.5%    | ~92.1%    | ~50.0%    |
| Male   | ~36.9%    | ~15.7%    | ~13.5%    |

First and second-class women had near-perfect survival. Even third-class women survived at 50% — far exceeding first-class men at 37%.

### By Port of Embarkation

| Port        | Survival Rate |
|-------------|---------------|
| Cherbourg   | 55.4%         |
| Queenstown  | 39.0%         |
| Southampton | 33.7%         |

Cherbourg advantage is likely mediated by higher proportion of 1st-class passengers boarding there.

### By Age Group

| Age Group           | Survival Rate | n     |
|---------------------|---------------|-------|
| Child (0-12)        | ~58%          | ~62   |
| Teen (13-18)        | ~38%          | ~44   |
| Young Adult (19-35) | ~37%          | ~299  |
| Middle Age (36-55)  | ~41%          | ~230  |
| Senior (56+)        | ~28%          | ~79   |

Children had meaningfully higher survival. Age is a weaker predictor than Sex or Class but still meaningful.

### Cabin Information as Proxy

| Cabin Status    | Survival Rate |
|-----------------|---------------|
| Has Cabin Info  | 66.7%         |
| No Cabin Info   | 30.0%         |

The binary HasCabin feature is nearly as predictive as Pclass — likely because cabin records exist primarily for 1st-class passengers.

---

## 7. Key Findings Summary

### Critical Findings

1. **Sex is the dominant predictor** — females survived at 74.2% vs 18.9% for males (|r| = 0.543)
2. **Class creates a clear survival gradient** — 63% (1st) vs 47% (2nd) vs 24% (3rd)
3. **Sex x Class interaction is powerful** — this combination captures most of the variance
4. **Cabin missingness is informative** — its absence is a proxy for lower-class passengers
5. **Fare is heavily right-skewed** — log transformation essential for linear models
6. **Age is non-linear** — children benefited; seniors disadvantaged; middle-aged mixed

### Feature Engineering Opportunities

| New Feature     | Construction                              | Rationale                              |
|-----------------|-------------------------------------------|----------------------------------------|
| `FamilySize`    | SibSp + Parch + 1                        | Total party size                       |
| `IsAlone`       | 1 if FamilySize == 1 else 0              | Solo travelers have distinct pattern   |
| `Title`         | Extracted from Name (Mr., Mrs., Miss.)   | Encodes sex + age + social status      |
| `HasCabin`      | 0/1 based on Cabin nullity               | Strong class/wealth proxy              |
| `Deck`          | First letter of Cabin                    | Physical location on ship              |
| `Fare_log`      | log1p(Fare)                              | Normalize extreme skew                 |
| `AgeGroup`      | Binned Age categories                    | Capture non-linear age effects         |
| `FarePerPerson` | Fare / party ticket size                 | Correct for shared ticket fare pooling |

### Modeling Red Flags

| Flag                      | Recommended Action                                          |
|---------------------------|-------------------------------------------------------------|
| Class imbalance (62/38)   | Use stratified splits; consider class_weight='balanced'     |
| Pclass-Fare collinearity  | Test models with each independently + combined              |
| PassengerId               | Drop before any modeling                                    |
| Name/Ticket raw           | Feature-engineer (Title, ticket prefix) before use          |
| Age imputation leakage    | Fit imputer on train set only; apply same to test           |
| Fare = 0 (15 passengers)  | Investigate; consider capping or flagging separately        |

---

## 8. Reports Saved

### Visualizations (`reports/`)

| File                               | Content                                        |
|------------------------------------|------------------------------------------------|
| `01_numerical_distributions.png`   | Histograms with mean/median: Age, Fare, SibSp, Parch, Pclass |
| `02_categorical_distributions.png` | Bar charts with counts: Survived, Sex, Pclass, Embarked |
| `03_boxplots_outliers.png`         | Box plots with outlier markers and stats        |
| `04_correlation_heatmap.png`       | Pearson correlation matrix (encoded categoricals) |
| `05_target_variable_analysis.png`  | 12-panel survival rate analysis across all features |
| `06_missing_data.png`              | Missing value percentages + pattern heatmap     |

### Structured Reports

| File                        | Format | Purpose                                      |
|-----------------------------|--------|----------------------------------------------|
| `.claude/eda_report.json`   | JSON   | Machine-readable report for downstream agents |
| `reports/eda_report.json`   | JSON   | Backup copy of structured report             |
| `reports/eda_report.md`     | MD     | This human-readable summary report           |

---

## 9. Agent Report & Data Versioning

- **Data Fingerprint (SHA-256):** `86c46fd2a53e9360...3394f4da`
- **Source Path:** `/Users/maximolorenzoylosada/Documents/2026-03-03 DEMO/Titanic-Dataset.csv`
- **Rows:** 891 | **Columns:** 12
- **Detected Task Type:** Binary Classification
- **Enables Next Agents:** `feature-engineering-analyst`, `ml-theory-advisor`

---

*Report generated by EDA Analyst Agent | Claude Sonnet 4.6 | 2026-03-03*
