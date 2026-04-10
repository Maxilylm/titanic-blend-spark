"""
Comprehensive EDA for Titanic Dataset
Classification Task: Predict 'Survived' (binary 0/1)
"""

import os
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = "/Users/maximolorenzoylosada/Documents/2026-03-03 DEMO"
DATA_PATH  = os.path.join(BASE_DIR, "Titanic-Dataset.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
JSON_DIR   = os.path.join(BASE_DIR, ".claude", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(JSON_DIR,   exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})

COLORS = {"survived": "#2196F3", "died": "#F44336",
          "male": "#42A5F5",    "female": "#EF9A9A"}

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

# Data fingerprint (SHA-256 of col names + dtypes + shape)
fingerprint_src = "|".join(df.columns.tolist()) + "|".join(df.dtypes.astype(str).tolist()) + str(df.shape)
data_fingerprint = hashlib.sha256(fingerprint_src.encode()).hexdigest()
print(f"Data fingerprint: {data_fingerprint}")

# ─────────────────────────────────────────────
# 2. COLUMN CLASSIFICATION
# ─────────────────────────────────────────────
id_cols   = ["PassengerId"]
text_cols = ["Name", "Ticket", "Cabin"]
cat_cols  = ["Survived", "Pclass", "Sex", "Embarked"]
num_cols  = ["Age", "SibSp", "Parch", "Fare"]

# ─────────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────────
missing = {}
for col in df.columns:
    cnt = int(df[col].isnull().sum())
    pct = round(cnt / len(df) * 100, 2)
    missing[col] = {"count": cnt, "percentage": pct}

missing_df = pd.DataFrame(missing).T
missing_df = missing_df[missing_df["count"] > 0].sort_values("count", ascending=False)
print("\nMissing values:")
print(missing_df)

# ─────────────────────────────────────────────
# 4. BASIC STATISTICS
# ─────────────────────────────────────────────
num_stats = df[num_cols].describe().round(4)
print("\nNumerical stats:\n", num_stats)

# Survival distribution
surv_counts = df["Survived"].value_counts().to_dict()
surv_pct    = df["Survived"].value_counts(normalize=True).mul(100).round(2).to_dict()
print(f"\nSurvival: died={surv_counts.get(0,0)}, survived={surv_counts.get(1,0)}")

# ─────────────────────────────────────────────
# 5. CORRELATIONS
# ─────────────────────────────────────────────
df_enc = df.copy()
df_enc["Sex_enc"]      = df_enc["Sex"].map({"male": 0, "female": 1})
df_enc["Embarked_enc"] = df_enc["Embarked"].map({"S": 0, "C": 1, "Q": 2})
corr_features = ["Survived", "Pclass", "Sex_enc", "Age", "SibSp", "Parch", "Fare", "Embarked_enc"]
corr_matrix   = df_enc[corr_features].corr().round(4)
target_corr   = corr_matrix["Survived"].drop("Survived").sort_values(key=abs, ascending=False)
print("\nCorrelations with Survived:\n", target_corr)

# ─────────────────────────────────────────────
# 6. SURVIVAL RATES BY KEY FEATURES
# ─────────────────────────────────────────────
surv_by_sex      = df.groupby("Sex")["Survived"].mean().round(4).to_dict()
surv_by_pclass   = df.groupby("Pclass")["Survived"].mean().round(4).to_dict()
surv_by_embarked = df.groupby("Embarked")["Survived"].mean().round(4).to_dict()

# Age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 120],
    labels=["Child(0-12)", "Teen(13-18)", "Adult(19-35)", "MiddleAge(36-60)", "Senior(61+)"]
)
surv_by_age_group = df.groupby("AgeGroup", observed=False)["Survived"].mean().round(4).to_dict()
surv_by_age_group = {str(k): v for k, v in surv_by_age_group.items()}

# ─────────────────────────────────────────────
# 7. DATA QUALITY ISSUES
# ─────────────────────────────────────────────
quality_issues = []

# Missing Age (19.9%)
age_missing_pct = missing["Age"]["percentage"]
quality_issues.append({
    "column": "Age",
    "issue": "Missing values",
    "detail": f"{missing['Age']['count']} missing ({age_missing_pct}%)",
    "severity": "high",
    "recommendation": "Impute with median grouped by Pclass and Sex"
})

# Missing Cabin (77.1%)
quality_issues.append({
    "column": "Cabin",
    "issue": "Missing values",
    "detail": f"{missing['Cabin']['count']} missing ({missing['Cabin']['percentage']}%)",
    "severity": "high",
    "recommendation": "Create binary 'HasCabin' feature; extract deck letter where available"
})

# Missing Embarked (0.2%)
quality_issues.append({
    "column": "Embarked",
    "issue": "Missing values",
    "detail": f"{missing['Embarked']['count']} missing ({missing['Embarked']['percentage']}%)",
    "severity": "low",
    "recommendation": "Impute with mode ('S')"
})

# Fare outliers
fare_q99 = df["Fare"].quantile(0.99)
fare_outliers = int((df["Fare"] > fare_q99).sum())
quality_issues.append({
    "column": "Fare",
    "issue": "Extreme outliers",
    "detail": f"{fare_outliers} values above 99th percentile ({fare_q99:.2f}); max={df['Fare'].max():.2f}",
    "severity": "medium",
    "recommendation": "Apply log transform or cap at 99th percentile"
})

# Fare zero
fare_zeros = int((df["Fare"] == 0).sum())
quality_issues.append({
    "column": "Fare",
    "issue": "Zero values",
    "detail": f"{fare_zeros} zero-fare passengers",
    "severity": "medium",
    "recommendation": "Investigate; may indicate staff or data entry errors"
})

# SibSp/Parch extremes
sibsp_max = int(df["SibSp"].max())
parch_max = int(df["Parch"].max())
quality_issues.append({
    "column": "SibSp/Parch",
    "issue": "Extreme family sizes",
    "detail": f"SibSp max={sibsp_max}, Parch max={parch_max}",
    "severity": "low",
    "recommendation": "Create 'FamilySize' feature = SibSp + Parch + 1; flag large families"
})

# Age skewness
age_skew = round(float(df["Age"].skew()), 4)
fare_skew = round(float(df["Fare"].skew()), 4)
quality_issues.append({
    "column": "Fare",
    "issue": "High positive skew",
    "detail": f"Skewness = {fare_skew}",
    "severity": "medium",
    "recommendation": "Apply log(Fare+1) transform before modelling"
})

# ─────────────────────────────────────────────
# 8. RECOMMENDATIONS
# ─────────────────────────────────────────────
recommendations = [
    {"action": "Impute Age with median grouped by Pclass and Sex", "priority": "high", "target_agent": "feature-engineering-analyst"},
    {"action": "Impute Embarked mode ('S'); encode as ordinal or one-hot", "priority": "low", "target_agent": "feature-engineering-analyst"},
    {"action": "Create binary HasCabin feature; extract Deck letter from Cabin", "priority": "high", "target_agent": "feature-engineering-analyst"},
    {"action": "Apply log1p transform to Fare to reduce skew", "priority": "medium", "target_agent": "feature-engineering-analyst"},
    {"action": "Engineer FamilySize = SibSp + Parch + 1 and IsAlone flag", "priority": "high", "target_agent": "feature-engineering-analyst"},
    {"action": "Extract Title from Name (Mr, Mrs, Miss, Master, Rare)", "priority": "high", "target_agent": "feature-engineering-analyst"},
    {"action": "Drop PassengerId, Name, Ticket as non-predictive identifiers", "priority": "high", "target_agent": "feature-engineering-analyst"},
    {"action": "Encode Sex as binary (0/1); one-hot encode Embarked and Pclass", "priority": "high", "target_agent": "feature-engineering-analyst"},
    {"action": "Address class imbalance (61.6% vs 38.4%) with stratified splits", "priority": "medium", "target_agent": "ml-theory-advisor"},
    {"action": "Consider ensemble methods (RandomForest, XGBoost) given mixed feature types", "priority": "medium", "target_agent": "ml-theory-advisor"},
]

# ─────────────────────────────────────────────
# 9. KEY FINDINGS
# ─────────────────────────────────────────────
key_findings = [
    f"Dataset has 891 passengers; {surv_counts.get(1,0)} survived ({surv_pct.get(1,0):.1f}%) and {surv_counts.get(0,0)} died ({surv_pct.get(0,0):.1f}%) — moderate class imbalance.",
    f"Sex is the strongest predictor: female survival rate {surv_by_sex.get('female',0)*100:.1f}% vs male {surv_by_sex.get('male',0)*100:.1f}%.",
    f"Pclass strongly predicts survival: 1st class {surv_by_pclass.get(1,0)*100:.1f}%, 2nd {surv_by_pclass.get(2,0)*100:.1f}%, 3rd {surv_by_pclass.get(3,0)*100:.1f}%.",
    f"Age has {missing['Age']['count']} missing values ({age_missing_pct}%); imputation is required.",
    f"Cabin is missing for {missing['Cabin']['percentage']}% of passengers — nearly useless as-is but deck letter can be salvaged.",
    f"Fare is heavily right-skewed (skewness={fare_skew}); log transform recommended.",
    f"Embarked port affects survival: Cherbourg (C) {surv_by_embarked.get('C',0)*100:.1f}%, Queenstown (Q) {surv_by_embarked.get('Q',0)*100:.1f}%, Southampton (S) {surv_by_embarked.get('S',0)*100:.1f}%.",
    f"Children (0-12) had highest survival rate ({surv_by_age_group.get('Child(0-12)', 0)*100:.1f}%), confirming 'women and children first'.",
    f"Correlation of Sex_enc with Survived: {target_corr.get('Sex_enc', 0):.3f} (strongest single feature).",
    f"Pclass and Fare are highly correlated (wealthier passengers in higher classes).",
    f"FamilySize engineering likely valuable: solo travellers and very large families had lower survival rates.",
]

print("\nKey findings generated:", len(key_findings))


# ═══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

# ── Plot 1: Target Distribution ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Variable: Survived Distribution", fontsize=14, fontweight="bold")

labels    = ["Did Not Survive (0)", "Survived (1)"]
counts    = [surv_counts.get(0, 0), surv_counts.get(1, 0)]
bar_colors = [COLORS["died"], COLORS["survived"]]

ax = axes[0]
bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white", linewidth=1.5)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{cnt}\n({cnt/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=12)
ax.set_title("Count by Survival Status")
ax.set_ylabel("Count")
ax.set_ylim(0, max(counts) * 1.2)

axes[1].pie(counts, labels=labels, autopct="%1.1f%%",
            colors=bar_colors, startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "01_target_distribution.png"))
plt.close()
print("Saved: 01_target_distribution.png")

# ── Plot 2: Missing Values ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
miss_plot = missing_df[missing_df["count"] > 0].copy()
miss_plot["percentage"] = miss_plot["percentage"].astype(float)
colors_miss = ["#EF5350" if p > 50 else "#FFA726" if p > 15 else "#66BB6A"
               for p in miss_plot["percentage"]]
bars = ax.barh(miss_plot.index, miss_plot["percentage"], color=colors_miss, edgecolor="white")
for bar, (idx, row) in zip(bars, miss_plot.iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{int(row['count'])} ({row['percentage']:.1f}%)",
            va="center", fontsize=11)
ax.set_xlabel("Missing Percentage (%)")
ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
ax.set_xlim(0, 100)
ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "02_missing_values.png"))
plt.close()
print("Saved: 02_missing_values.png")

# ── Plot 3: Survival by Sex ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Survival by Sex", fontsize=14, fontweight="bold")

sex_surv = df.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
sex_surv.plot(kind="bar", ax=axes[0], color=[COLORS["died"], COLORS["survived"]],
              edgecolor="white", rot=0)
axes[0].set_title("Count by Sex and Survival")
axes[0].set_xlabel("Sex")
axes[0].set_ylabel("Count")
axes[0].legend(["Did Not Survive", "Survived"])
for container in axes[0].containers:
    axes[0].bar_label(container, fontsize=10)

sex_rate = pd.DataFrame({"Survival Rate": surv_by_sex})
sex_rate.plot(kind="bar", ax=axes[1], color=[COLORS["female"], COLORS["male"]][::-1],
              edgecolor="white", rot=0, legend=False)
axes[1].set_title("Survival Rate by Sex")
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1.1)
for i, (sex, rate) in enumerate(surv_by_sex.items()):
    axes[1].text(i, rate + 0.02, f"{rate*100:.1f}%", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "03_survival_by_sex.png"))
plt.close()
print("Saved: 03_survival_by_sex.png")

# ── Plot 4: Survival by Pclass ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Survival by Passenger Class", fontsize=14, fontweight="bold")

pclass_surv = df.groupby(["Pclass", "Survived"]).size().unstack(fill_value=0)
pclass_surv.plot(kind="bar", ax=axes[0], color=[COLORS["died"], COLORS["survived"]],
                 edgecolor="white", rot=0)
axes[0].set_title("Count by Pclass and Survival")
axes[0].set_xlabel("Passenger Class")
axes[0].set_ylabel("Count")
axes[0].legend(["Did Not Survive", "Survived"])
for container in axes[0].containers:
    axes[0].bar_label(container, fontsize=10)

pclass_rate = pd.DataFrame({"Survival Rate": {str(k): v for k, v in surv_by_pclass.items()}})
pclass_rate.plot(kind="bar", ax=axes[1], color="#42A5F5", edgecolor="white", rot=0, legend=False)
axes[1].set_title("Survival Rate by Pclass")
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1.1)
for i, (cls, rate) in enumerate(surv_by_pclass.items()):
    axes[1].text(i, rate + 0.02, f"{rate*100:.1f}%", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "04_survival_by_pclass.png"))
plt.close()
print("Saved: 04_survival_by_pclass.png")

# ── Plot 5: Age Distribution ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Age Distribution", fontsize=14, fontweight="bold")

# Histogram with KDE by survival
for surv_val, label, color in [(0, "Did Not Survive", COLORS["died"]),
                                (1, "Survived",        COLORS["survived"])]:
    subset = df[df["Survived"] == surv_val]["Age"].dropna()
    axes[0].hist(subset, bins=30, alpha=0.5, color=color, label=label, edgecolor="white")

axes[0].set_title("Age Distribution by Survival")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].axvline(df["Age"].median(), color="black", linestyle="--", alpha=0.7,
                label=f"Median={df['Age'].median():.1f}")

# Survival rate by age group
age_group_rate = df.groupby("AgeGroup", observed=False)["Survived"].mean()
age_group_count = df.groupby("AgeGroup", observed=False)["Survived"].count()
colors_age = plt.cm.Blues(np.linspace(0.3, 0.9, len(age_group_rate)))
bars = axes[1].bar(range(len(age_group_rate)), age_group_rate.values, color=colors_age, edgecolor="white")
axes[1].set_xticks(range(len(age_group_rate)))
axes[1].set_xticklabels(age_group_rate.index, rotation=20, ha="right")
axes[1].set_title("Survival Rate by Age Group")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1.1)
for bar, (rate, cnt) in zip(bars, zip(age_group_rate.values, age_group_count.values)):
    axes[1].text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                 f"{rate*100:.1f}%\n(n={cnt})", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "05_age_distribution.png"))
plt.close()
print("Saved: 05_age_distribution.png")

# ── Plot 6: Fare Distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fare Distribution", fontsize=14, fontweight="bold")

axes[0].hist(df["Fare"].dropna(), bins=50, color="#42A5F5", edgecolor="white", alpha=0.8)
axes[0].set_title(f"Fare Histogram (skew={fare_skew:.2f})")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Count")
axes[0].axvline(df["Fare"].median(), color="red", linestyle="--",
                label=f"Median={df['Fare'].median():.2f}")
axes[0].legend()

axes[1].hist(np.log1p(df["Fare"].dropna()), bins=40, color="#66BB6A", edgecolor="white", alpha=0.8)
axes[1].set_title("log1p(Fare) Distribution (after transform)")
axes[1].set_xlabel("log1p(Fare)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "06_fare_distribution.png"))
plt.close()
print("Saved: 06_fare_distribution.png")

# ── Plot 7: Correlation Heatmap ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 10}, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "07_correlation_heatmap.png"))
plt.close()
print("Saved: 07_correlation_heatmap.png")

# ── Plot 8: Survival by Embarked ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Survival by Embarkation Port", fontsize=14, fontweight="bold")

embarked_surv = df.groupby(["Embarked", "Survived"]).size().unstack(fill_value=0)
embarked_surv.plot(kind="bar", ax=axes[0], color=[COLORS["died"], COLORS["survived"]],
                   edgecolor="white", rot=0)
axes[0].set_title("Count by Embarked and Survival")
axes[0].set_xlabel("Embarked (C=Cherbourg, Q=Queenstown, S=Southampton)")
axes[0].set_ylabel("Count")
axes[0].legend(["Did Not Survive", "Survived"])
for container in axes[0].containers:
    axes[0].bar_label(container, fontsize=10)

emb_rate = df.groupby("Embarked")["Survived"].mean()
emb_colors = ["#42A5F5", "#66BB6A", "#FFA726"]
bars = axes[1].bar(emb_rate.index, emb_rate.values, color=emb_colors, edgecolor="white")
axes[1].set_title("Survival Rate by Embarked")
axes[1].set_xlabel("Embarked Port")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1.1)
for bar, (port, rate) in zip(bars, emb_rate.items()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                 f"{rate*100:.1f}%", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "08_survival_by_embarked.png"))
plt.close()
print("Saved: 08_survival_by_embarked.png")

# ── Plot 9: Sex × Pclass Heatmap ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
            ax=ax, annot_kws={"size": 14})
ax.set_title("Survival Rate: Sex x Pclass", fontsize=13, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Sex")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "09_sex_pclass_heatmap.png"))
plt.close()
print("Saved: 09_sex_pclass_heatmap.png")

# ── Plot 10: Target Correlation Bar ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
tc = target_corr.sort_values()
bar_colors_corr = ["#EF5350" if v < 0 else "#42A5F5" for v in tc.values]
bars = ax.barh(tc.index, tc.values, color=bar_colors_corr, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, tc.values):
    ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=10)
ax.set_title("Feature Correlations with Survived", fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "10_target_correlations.png"))
plt.close()
print("Saved: 10_target_correlations.png")

# ── Plot 11: Fare Boxplot by Pclass ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
df.boxplot(column="Fare", by="Pclass", ax=ax,
           boxprops=dict(color="#1565C0"),
           medianprops=dict(color="red", linewidth=2),
           whiskerprops=dict(color="#1565C0"),
           capprops=dict(color="#1565C0"),
           flierprops=dict(marker="o", color="grey", alpha=0.3))
ax.set_title("Fare Distribution by Passenger Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Fare")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "11_fare_by_pclass.png"))
plt.close()
print("Saved: 11_fare_by_pclass.png")

# ── Plot 12: Family Size vs Survival ─────────────────────────
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
family_surv = df.groupby("FamilySize")["Survived"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Family Size and Survival", fontsize=14, fontweight="bold")

family_count = df.groupby(["FamilySize", "Survived"]).size().unstack(fill_value=0)
family_count.plot(kind="bar", ax=axes[0], color=[COLORS["died"], COLORS["survived"]],
                  edgecolor="white", rot=0)
axes[0].set_title("Count by Family Size")
axes[0].set_xlabel("Family Size")
axes[0].set_ylabel("Count")
axes[0].legend(["Did Not Survive", "Survived"])

axes[1].plot(family_surv["FamilySize"], family_surv["Survived"],
             marker="o", color="#42A5F5", linewidth=2, markersize=8)
axes[1].fill_between(family_surv["FamilySize"], family_surv["Survived"], alpha=0.2, color="#42A5F5")
axes[1].set_title("Survival Rate by Family Size")
axes[1].set_xlabel("Family Size")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "12_family_size_survival.png"))
plt.close()
print("Saved: 12_family_size_survival.png")

# ── Plot 13: Numerical Feature Distributions (4-panel) ───────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")

for ax, col in zip(axes.flatten(), num_cols):
    survived_vals = df[df["Survived"] == 1][col].dropna()
    died_vals     = df[df["Survived"] == 0][col].dropna()
    ax.hist(died_vals,     bins=30, alpha=0.6, color=COLORS["died"],     label="Did Not Survive", edgecolor="white")
    ax.hist(survived_vals, bins=30, alpha=0.6, color=COLORS["survived"], label="Survived",        edgecolor="white")
    ax.set_title(f"{col} Distribution by Survival")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "13_numerical_distributions.png"))
plt.close()
print("Saved: 13_numerical_distributions.png")

# ── Plot 14: Pairplot subset ──────────────────────────────────
pair_df = df[["Age", "Fare", "FamilySize", "Survived"]].dropna()
pair_df["Survived_label"] = pair_df["Survived"].map({0: "Did Not Survive", 1: "Survived"})
pg = sns.pairplot(pair_df, hue="Survived_label", vars=["Age", "Fare", "FamilySize"],
                  plot_kws={"alpha": 0.4},
                  palette={"Did Not Survive": COLORS["died"], "Survived": COLORS["survived"]})
pg.fig.suptitle("Pairplot: Age, Fare, FamilySize by Survival", y=1.02, fontsize=13, fontweight="bold")
pg.fig.savefig(os.path.join(REPORT_DIR, "14_pairplot.png"), bbox_inches="tight")
plt.close()
print("Saved: 14_pairplot.png")


# ═══════════════════════════════════════════════════════════════
# BUILD JSON REPORT
# ═══════════════════════════════════════════════════════════════

# Key statistics — serialisable
key_stats = {}
for col in num_cols:
    s = df[col].describe()
    key_stats[col] = {
        "count":    int(s["count"]),
        "mean":     round(float(s["mean"]), 4),
        "std":      round(float(s["std"]),  4),
        "min":      round(float(s["min"]),  4),
        "25%":      round(float(s["25%"]),  4),
        "50%":      round(float(s["50%"]),  4),
        "75%":      round(float(s["75%"]),  4),
        "max":      round(float(s["max"]),  4),
        "skewness": round(float(df[col].skew()), 4),
        "kurtosis": round(float(df[col].kurtosis()), 4),
        "missing":  int(df[col].isnull().sum()),
    }

correlations_dict = {
    "pearson_with_survived": {k: round(float(v), 4) for k, v in target_corr.items()},
    "top_positive": str(target_corr[target_corr > 0].idxmax()),
    "top_negative": str(target_corr[target_corr < 0].idxmin()),
}

json_report = {
    "metadata": {
        "generated_at":    datetime.now().isoformat(),
        "task_type":       "binary_classification",
        "data_fingerprint": data_fingerprint,
        "source_path":     DATA_PATH,
        "analyst":         "eda-analyst",
    },
    "dataset_info": {
        "rows":         len(df),
        "columns":      df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes":       {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    },
    "target_info": {
        "name":          "Survived",
        "type":          "binary",
        "distribution":  {str(k): int(v) for k, v in surv_counts.items()},
        "class_balance": {str(k): float(v) for k, v in surv_pct.items()},
        "majority_class": 0,
        "minority_class": 1,
        "imbalance_ratio": round(surv_counts.get(0, 0) / surv_counts.get(1, 1), 3),
    },
    "column_types": {
        "numerical":   num_cols,
        "categorical": ["Pclass", "Sex", "Embarked"],
        "target":      ["Survived"],
        "id_columns":  id_cols,
        "text_columns": text_cols,
    },
    "missing_values": missing,
    "key_statistics": key_stats,
    "correlations": correlations_dict,
    "survival_rates": {
        "by_sex":       {k: round(float(v), 4) for k, v in surv_by_sex.items()},
        "by_pclass":    {str(k): round(float(v), 4) for k, v in surv_by_pclass.items()},
        "by_embarked":  {k: round(float(v), 4) for k, v in surv_by_embarked.items()},
        "by_age_group": surv_by_age_group,
    },
    "data_quality_issues": quality_issues,
    "recommendations": recommendations,
    "key_findings": key_findings,
    "artifacts": {
        "plots": [os.path.join(REPORT_DIR, f) for f in [
            "01_target_distribution.png", "02_missing_values.png",
            "03_survival_by_sex.png",     "04_survival_by_pclass.png",
            "05_age_distribution.png",    "06_fare_distribution.png",
            "07_correlation_heatmap.png", "08_survival_by_embarked.png",
            "09_sex_pclass_heatmap.png",  "10_target_correlations.png",
            "11_fare_by_pclass.png",      "12_family_size_survival.png",
            "13_numerical_distributions.png", "14_pairplot.png",
        ]],
        "reports": [
            os.path.join(JSON_DIR, "eda-analyst_report.json"),
            os.path.join(REPORT_DIR, "eda_report.md"),
        ],
    },
    "enables": ["feature-engineering-analyst", "ml-theory-advisor", "frontend-ux-analyst"],
    "status": "completed",
}

# Save JSON report
json_path = os.path.join(JSON_DIR, "eda-analyst_report.json")
with open(json_path, "w") as f:
    json.dump(json_report, f, indent=2)
print(f"\nSaved JSON report: {json_path}")

# ═══════════════════════════════════════════════════════════════
# BUILD MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════

md = f"""# Titanic Dataset — Exploratory Data Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Task:** Binary Classification — Predict `Survived` (0 = Did Not Survive, 1 = Survived)
**Data Fingerprint:** `{data_fingerprint}`
**Source:** `{DATA_PATH}`

---

## 1. Dataset Overview

| Attribute      | Value              |
|----------------|--------------------|
| Rows           | {len(df):,}        |
| Columns        | {df.shape[1]}      |
| Memory Usage   | {round(df.memory_usage(deep=True).sum()/1024, 1)} KB |
| Task Type      | Binary Classification |
| Target Column  | `Survived`         |

### Column Inventory

| Column       | Type     | Category    | Notes                          |
|--------------|----------|-------------|--------------------------------|
| PassengerId  | int64    | ID          | Unique row identifier          |
| Survived     | int64    | Target      | 0=No, 1=Yes                   |
| Pclass       | int64    | Categorical | Ticket class (1=1st, 3=3rd)   |
| Name         | object   | Text        | Full name with title           |
| Sex          | object   | Categorical | male / female                  |
| Age          | float64  | Numerical   | 19.9% missing                  |
| SibSp        | int64    | Numerical   | Siblings/spouses aboard        |
| Parch        | int64    | Numerical   | Parents/children aboard        |
| Ticket       | object   | Text        | Ticket number (high cardinality)|
| Fare         | float64  | Numerical   | Ticket price in GBP            |
| Cabin        | object   | Text        | Cabin number — 77.1% missing   |
| Embarked     | object   | Categorical | Port: C=Cherbourg, Q=Queenstown, S=Southampton |

---

## 2. Data Quality Report

| Issue Type      | Columns Affected | Severity | Count / % Missing  | Recommended Action                              |
|-----------------|------------------|----------|--------------------|--------------------------------------------------|
| Missing values  | Age              | HIGH     | {missing['Age']['count']} ({missing['Age']['percentage']}%)   | Impute with median grouped by Pclass & Sex       |
| Missing values  | Cabin            | HIGH     | {missing['Cabin']['count']} ({missing['Cabin']['percentage']}%) | Create HasCabin flag; extract Deck letter        |
| Missing values  | Embarked         | LOW      | {missing['Embarked']['count']} ({missing['Embarked']['percentage']}%)    | Impute with mode (S)                            |
| Extreme outliers| Fare             | MEDIUM   | Max={df['Fare'].max():.2f}, skew={fare_skew}   | Log1p transform before modelling                |
| Zero values     | Fare             | MEDIUM   | {fare_zeros} rows                    | Investigate; may be staff or data entry errors  |
| High cardinality| Ticket           | LOW      | {df['Ticket'].nunique()} unique values         | Drop or extract prefix features                 |
| High cardinality| Name             | LOW      | All unique                         | Extract Title (Mr, Mrs, Miss, Master, Rare)     |

---

## 3. Target Variable — Survived

| Class | Label           | Count | Percentage |
|-------|-----------------|-------|------------|
| 0     | Did Not Survive | {surv_counts.get(0,0)}   | {surv_pct.get(0,0):.1f}%      |
| 1     | Survived        | {surv_counts.get(1,0)}   | {surv_pct.get(1,0):.1f}%      |

**Class Imbalance Ratio (0:1):** {round(surv_counts.get(0,0)/surv_counts.get(1,1), 2)}
**Assessment:** Moderate imbalance — use stratified train/test splits. Accuracy alone is not a sufficient metric; report ROC-AUC, F1-score, and precision/recall.

![Target Distribution](01_target_distribution.png)

---

## 4. Numerical Feature Statistics

| Feature  | Mean    | Median  | Std     | Min   | Max     | Skewness | Missing |
|----------|---------|---------|---------|-------|---------|----------|---------|
| Age      | {key_stats['Age']['mean']:.2f}  | {key_stats['Age']['50%']:.2f}  | {key_stats['Age']['std']:.2f}  | {key_stats['Age']['min']:.2f} | {key_stats['Age']['max']:.2f}  | {key_stats['Age']['skewness']:.2f}  | {key_stats['Age']['missing']} ({missing['Age']['percentage']}%) |
| Fare     | {key_stats['Fare']['mean']:.2f} | {key_stats['Fare']['50%']:.2f} | {key_stats['Fare']['std']:.2f} | {key_stats['Fare']['min']:.2f} | {key_stats['Fare']['max']:.2f} | {key_stats['Fare']['skewness']:.2f} | {key_stats['Fare']['missing']} |
| SibSp    | {key_stats['SibSp']['mean']:.2f} | {key_stats['SibSp']['50%']:.2f} | {key_stats['SibSp']['std']:.2f} | {key_stats['SibSp']['min']:.2f} | {key_stats['SibSp']['max']:.2f} | {key_stats['SibSp']['skewness']:.2f} | {key_stats['SibSp']['missing']} |
| Parch    | {key_stats['Parch']['mean']:.2f} | {key_stats['Parch']['50%']:.2f} | {key_stats['Parch']['std']:.2f} | {key_stats['Parch']['min']:.2f} | {key_stats['Parch']['max']:.2f} | {key_stats['Parch']['skewness']:.2f} | {key_stats['Parch']['missing']} |

---

## 5. Survival Rates by Key Features

### By Sex

| Sex    | Survival Rate |
|--------|--------------|
| Female | {surv_by_sex.get('female', 0)*100:.1f}%       |
| Male   | {surv_by_sex.get('male', 0)*100:.1f}%       |

**Insight:** Sex is the single most powerful predictor. The "women and children first" protocol is clearly reflected in the data.

![Survival by Sex](03_survival_by_sex.png)

### By Passenger Class

| Class | Survival Rate |
|-------|--------------|
| 1st   | {surv_by_pclass.get(1, 0)*100:.1f}%        |
| 2nd   | {surv_by_pclass.get(2, 0)*100:.1f}%        |
| 3rd   | {surv_by_pclass.get(3, 0)*100:.1f}%        |

**Insight:** Strong socioeconomic gradient — 1st class passengers were nearly 3x more likely to survive than 3rd class.

![Survival by Pclass](04_survival_by_pclass.png)

### By Age Group

| Age Group      | Survival Rate |
|----------------|--------------|
| Child (0-12)   | {surv_by_age_group.get('Child(0-12)', 0)*100:.1f}%  |
| Teen (13-18)   | {surv_by_age_group.get('Teen(13-18)', 0)*100:.1f}%   |
| Adult (19-35)  | {surv_by_age_group.get('Adult(19-35)', 0)*100:.1f}%  |
| Middle (36-60) | {surv_by_age_group.get('MiddleAge(36-60)', 0)*100:.1f}% |
| Senior (61+)   | {surv_by_age_group.get('Senior(61+)', 0)*100:.1f}%   |

**Insight:** Children had the highest survival rate, consistent with evacuation priorities. Note that age has 19.9% missing values, which may bias these estimates.

![Age Distribution](05_age_distribution.png)

### By Embarkation Port

| Port             | Survival Rate |
|------------------|--------------|
| Cherbourg (C)    | {surv_by_embarked.get('C', 0)*100:.1f}%  |
| Queenstown (Q)   | {surv_by_embarked.get('Q', 0)*100:.1f}%  |
| Southampton (S)  | {surv_by_embarked.get('S', 0)*100:.1f}%  |

**Insight:** Cherbourg passengers had a higher survival rate, likely because they were predominantly 1st class (Embarked is a proxy for Pclass).

![Survival by Embarked](08_survival_by_embarked.png)

---

## 6. Correlation Analysis

| Feature     | Correlation with Survived |
|-------------|--------------------------|
| Sex_enc     | {target_corr.get('Sex_enc', 0):.3f}                      |
| Pclass      | {target_corr.get('Pclass', 0):.3f}                       |
| Fare        | {target_corr.get('Fare', 0):.3f}                         |
| Embarked_enc| {target_corr.get('Embarked_enc', 0):.3f}                 |
| Parch       | {target_corr.get('Parch', 0):.3f}                        |
| Age         | {target_corr.get('Age', 0):.3f}                          |
| SibSp       | {target_corr.get('SibSp', 0):.3f}                        |

**Note:** Pearson correlations for binary target — low correlations don't imply low predictive power (non-linear relationships exist).

![Correlation Heatmap](07_correlation_heatmap.png)
![Target Correlations](10_target_correlations.png)

---

## 7. Key Visualisations

| File | Description |
|------|-------------|
| `01_target_distribution.png`    | Survival class balance (bar + pie) |
| `02_missing_values.png`         | Missing value severity by column |
| `03_survival_by_sex.png`        | Counts and rates by sex |
| `04_survival_by_pclass.png`     | Counts and rates by passenger class |
| `05_age_distribution.png`       | Age histograms by survival + age group rates |
| `06_fare_distribution.png`      | Raw vs log-transformed Fare |
| `07_correlation_heatmap.png`    | Full feature correlation matrix |
| `08_survival_by_embarked.png`   | Counts and rates by embarkation port |
| `09_sex_pclass_heatmap.png`     | Survival rate by Sex x Pclass interaction |
| `10_target_correlations.png`    | Ranked feature correlations with Survived |
| `11_fare_by_pclass.png`         | Fare distribution boxplots by class |
| `12_family_size_survival.png`   | Family size vs survival counts and rates |
| `13_numerical_distributions.png`| All numerical features by survival group |
| `14_pairplot.png`               | Pairwise scatter: Age, Fare, FamilySize |

---

## 8. Key Findings

{''.join([f"{i+1}. {finding}{chr(10)}" for i, finding in enumerate(key_findings)])}

---

## 9. Preprocessing Recommendations

| Priority | Action |
|----------|--------|
| HIGH     | Impute Age using median grouped by Pclass and Sex |
| HIGH     | Create binary HasCabin feature; extract Deck letter from Cabin |
| HIGH     | Engineer FamilySize = SibSp + Parch + 1; create IsAlone flag |
| HIGH     | Extract Title from Name (Mr, Mrs, Miss, Master, Rare) |
| HIGH     | Drop PassengerId, Name, Ticket |
| MEDIUM   | Apply log1p transform to Fare |
| MEDIUM   | Address class imbalance with stratified splits; use ROC-AUC as primary metric |
| MEDIUM   | Consider ensemble models (RandomForest, XGBoost, LightGBM) |
| LOW      | Impute Embarked with mode (S) |
| LOW      | One-hot encode Embarked; encode Sex as binary |

---

## 10. Questions for Domain Experts

1. Why do {fare_zeros} passengers have a zero Fare? Were they crew or complimentary passengers?
2. Should Cabin deck be treated as an ordinal feature (A=highest deck, G=lowest)?
3. Are the {missing['Age']['count']} missing Age values randomly distributed, or are they concentrated in a particular class/sex group?
4. Does the Ticket prefix carry meaningful information about groupings or pricing tiers?
5. Should SibSp and Parch be interpreted differently for children vs. adults?

---

*Report generated by EDA Analyst | Data fingerprint: `{data_fingerprint}`*
"""

md_path = os.path.join(REPORT_DIR, "eda_report.md")
with open(md_path, "w") as f:
    f.write(md)
print(f"Saved Markdown report: {md_path}")

print("\nAll EDA artefacts saved successfully.")
print(f"  JSON report : {json_path}")
print(f"  MD report   : {md_path}")
print(f"  Plots dir   : {REPORT_DIR}")
