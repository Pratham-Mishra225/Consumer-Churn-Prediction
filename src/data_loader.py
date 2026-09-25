import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

TEST_SIZE = 0.20
RANDOM_STATE = 42

def load_data():

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)
    return df

def prepare_data(df):
    df = df.copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Convert target variable to binary
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
            "No": 0,
            "Yes": 1
        })

    else:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    return df

def split_data(df):

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    return train_df, test_df


def save_data(train_df, test_df):

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_DATA_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_DATA_PATH,
        index=False
    )




def verify_split(train_df, test_df):

    print("\n========== DATA SPLIT VERIFICATION ==========")

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")   

    print("\nTrain target distribution:")
    print(
        train_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTest target distribution:")
    print(
        test_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTrain missing values:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])

    print("\nTest missing values:")
    print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

    print("\nTrain duplicate rows:", train_df.duplicated().sum())
    print("Test duplicate rows:", test_df.duplicated().sum())


def main():

    print("Loading dataset...")

    df = load_data()

    print(f"Raw dataset shape: {df.shape}")

    df = prepare_data(df)

    print(f"Dataset after basic preparation: {df.shape}")

    train_df, test_df = split_data(df)

    save_data(train_df, test_df)

    verify_split(train_df, test_df)

    print("\nProcessed datasets saved successfully.")
    print(f"Train: {TRAIN_DATA_PATH}")
    print(f"Test:  {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
    