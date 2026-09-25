import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.csv"
)

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.csv"
)

TARGET_COLUMN = "Churn"


def load_train_test_data():

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    return train_df, test_df


def separate_features_target(train_df, test_df):

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def prepare_feature_types(X_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["TotalCharges"] = pd.to_numeric(
        X_train["TotalCharges"],
        errors="coerce"
    )

    X_test["TotalCharges"] = pd.to_numeric(
        X_test["TotalCharges"],
        errors="coerce"
    )

    # SeniorCitizen is a binary categorical feature.
    X_train["SeniorCitizen"] = X_train[
        "SeniorCitizen"
    ].astype(str)

    X_test["SeniorCitizen"] = X_test[
        "SeniorCitizen"
    ].astype(str)

    return X_train, X_test



def get_feature_columns(X_train):

    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        column
        for column in X_train.columns
        if column not in numerical_features
    ]

    return numerical_features, categorical_features



def build_preprocessor(
    numerical_features,
    categorical_features
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(
                    strategy="constant",
                    fill_value=0
                )
            ),
            (
                "scaler", StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical", numerical_pipeline, numerical_features
            ),
            (
                "categorical", categorical_pipeline, categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


def preprocess_data():

    train_df, test_df = load_train_test_data()

    X_train, X_test, y_train, y_test = (
        separate_features_target(
            train_df,
            test_df
        )
    )

    X_train, X_test = prepare_feature_types(
        X_train,
        X_test
    )

    numerical_features, categorical_features = (
        get_feature_columns(X_train)
    )

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )


    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )



if __name__ == "__main__":

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data()

    print("\n========== PREPROCESSING VERIFICATION ==========")

    print(
        f"\nOriginal training rows: {len(y_train)}"
    )

    print(
        f"Original test rows: {len(y_test)}"
    )

    print(
        f"\nProcessed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed test shape: "
        f"{X_test_processed.shape}"
    )

    print(
        "\nProcessed training missing values:",
        np.isnan(X_train_processed).sum()
    )

    print(
        "Processed test missing values:",
        np.isnan(X_test_processed).sum()
    )

    print(
        "\nNumber of output features:",
        X_train_processed.shape[1]
    )

    print(
        "\n==============================================="
    )