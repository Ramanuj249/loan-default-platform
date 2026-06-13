import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def drop_high_missing_columns(df, threshold = 40):
    missing_percentage = (df.isnull().sum()/len(df)) *100
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    print(f"Dropping {len(cols_to_drop)} columns with more than the {threshold}% missing values.")
    df = df.drop(columns=cols_to_drop)

    return df

def fix_days_columns(df):
    df["AGE_YEARS"]=df["DAYS_BIRTH"]/-365
    df["DAYS_EMPLOYED"]=df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df["YEARS_EMPLOYED"]=df["DAYS_EMPLOYED"]/-365
    return df

def fill_missing_values(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encode_categorical_columns(df):
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def scale_features(df, target_col = ["TARGET"]):
    X = df.drop(columns=target_col)
    y = df[target_col].values.ravel()

    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y, scaler

def apply_smote(X,y):
    print(f"before smote - class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"After Smote - class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    return X_resampled, y_resampled

def preprocess_data(df):
    print("Starting Preprocessing...")
    df = drop_high_missing_columns(df)
    df = fix_days_columns(df)
    df = fill_missing_values(df)
    df = encode_categorical_columns(df)
    X, y, scaler = scale_features(df)
    X, y = apply_smote(X, y)
    print("Preprocessing complete!")
    return X, y, scaler

if __name__ == "__main__":
    df = pd.read_csv("../../data/application_train.csv")
    X, y, scaler = preprocess_data(df)
    print("X shape:", X.shape)
    print("y shape:", y.shape)