"""
Titanic data processing pipeline.

Implements leakage-safe feature engineering and preprocessing.
All transformations are fit on training data only.
"""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer for Titanic dataset.

    Fits imputation statistics on training data only to prevent leakage.
    """

    def __init__(self):
        self.age_medians_ = None
        self.fare_bins_ = None
        self.global_age_median_ = None

    @staticmethod
    def _extract_title(name):
        match = re.search(r' ([A-Za-z]+)\.', name)
        if match:
            title = match.group(1)
            if title in ['Mr']:
                return 'Mr'
            elif title in ['Mrs', 'Mme']:
                return 'Mrs'
            elif title in ['Miss', 'Ms', 'Mlle']:
                return 'Miss'
            elif title in ['Master']:
                return 'Master'
            else:
                return 'Rare'
        return 'Mr'

    def fit(self, X, y=None):
        df = X.copy()
        # Compute age medians by Pclass+Sex on training data only
        self.age_medians_ = df.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        self.global_age_median_ = df['Age'].median()

        # Compute fare quartile bins on training data only
        df_fare = df['Fare'].dropna()
        df_fare = df_fare[df_fare > 0]
        self.fare_bins_ = [
            -np.inf,
            df_fare.quantile(0.25),
            df_fare.quantile(0.50),
            df_fare.quantile(0.75),
            np.inf,
        ]
        return self

    def transform(self, X):
        df = X.copy()

        # 1. Title extraction
        df['Title'] = df['Name'].apply(self._extract_title)
        title_map = {'Mr': 0, 'Master': 1, 'Miss': 2, 'Mrs': 3, 'Rare': 2}
        df['Title_encoded'] = df['Title'].map(title_map)

        # 2. Sex binary
        df['Sex_binary'] = (df['Sex'] == 'female').astype(int)

        # 3. Age imputation (using training-fit medians)
        for (pclass, sex), median_age in self.age_medians_.items():
            mask = (df['Age'].isna()) & (df['Pclass'] == pclass) & (df['Sex'] == sex)
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(self.global_age_median_)
        df.rename(columns={'Age': 'Age_imputed'}, inplace=True)

        # 4. Age bins
        df['AgeBin'] = pd.cut(
            df['Age_imputed'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=[0, 1, 2, 3, 4],
        ).astype(float).fillna(2)

        # 5. IsChild
        df['IsChild'] = (df['Age_imputed'] < 13).astype(int)

        # 6. Family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['FamilyType'] = pd.cut(
            df['FamilySize'],
            bins=[0, 1, 4, 11],
            labels=[0, 1, 2],
        ).astype(float).fillna(0)

        # 7. IsMother
        df['IsMother'] = (
            (df['Sex'] == 'female')
            & (df['Parch'] > 0)
            & (df['Age_imputed'] > 18)
            & (df['Title'] != 'Miss')
        ).astype(int)

        # 8. Cabin features
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['Deck'] = df['Cabin'].str.extract(r'^([A-Z])').fillna('U')
        deck_map = {'U': 0, 'G': 1, 'F': 2, 'A': 3, 'C': 4, 'E': 5, 'D': 6, 'B': 7, 'T': 0}
        df['Deck_encoded'] = df['Deck'].map(deck_map).fillna(0)

        # 9. Fare features
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare_log1p'] = np.log1p(df['Fare'])
        df['FareBin'] = pd.cut(
            df['Fare'],
            bins=self.fare_bins_,
            labels=[0, 1, 2, 3],
        ).astype(float).fillna(1)

        # 10. Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
        df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
        df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)

        # 11. Sex_Pclass interaction
        df['Sex_Pclass'] = df['Sex_binary'] * 10 + df['Pclass']

        # Select final feature columns
        feature_cols = [
            'Sex_binary', 'Title_encoded', 'Pclass', 'HasCabin', 'Deck_encoded',
            'Fare_log1p', 'FareBin', 'FamilySize', 'IsAlone', 'FamilyType',
            'Age_imputed', 'AgeBin', 'IsChild', 'IsMother',
            'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_Pclass',
        ]
        return df[feature_cols].astype(float)


def build_pipeline():
    """Build the full preprocessing pipeline."""
    return Pipeline([
        ('features', TitanicFeatureEngineer()),
        ('scaler', StandardScaler()),
    ])


def load_and_split(data_path, test_size=0.2, random_state=42):
    """Load data and perform stratified train/test split."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def process_data(data_path, test_size=0.2, random_state=42):
    """Load, split, and process data. Returns processed arrays and pipeline."""
    X_train, X_test, y_train, y_test = load_and_split(data_path, test_size, random_state)

    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, pipeline
