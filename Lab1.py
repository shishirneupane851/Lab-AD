import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sample_dataframe():
    data = {
        "Customer ID": [f"C{100+i}" for i in range(10)],
        "Age": [25, 34, np.nan, 45, 23, 52, 33, np.nan, 40, 29],
        "Gender": ["Male", "Female", "Female", "Male", np.nan, "Male", "Female", "Female", "Male", "Male"],
        "Income": [50000, 72000, 61000, np.nan, 30000, 98000, 54000, 45000, np.nan, 67000],
        "City": ["Urban", "Rural", "Urban", "Urban", "Rural", "Urban", np.nan, "Rural", "Urban", "Rural"],
        "Subscription Status": [
            "Subscribed", "Not Subscribed", "Subscribed", "Subscribed",
            "Not Subscribed", np.nan, "Subscribed", "Not Subscribed", "Subscribed", "Not Subscribed"
        ]
    }
    return pd.DataFrame(data)

def check_missing_values(df):
    print(df.isna().sum())
    print((df.isna().mean() * 100).round(2))

def handle_missing_values(df):
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Income"] = df["Income"].fillna(df["Income"].median())
    for col in ["Gender", "City", "Subscription Status"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def encode_categorical(df):
    df = df.copy()
    gender_map = {"Male": 0, "Female": 1}
    df["Gender_Label"] = df["Gender"].map(gender_map)
    df = pd.get_dummies(df, columns=["City", "Subscription Status"], prefix=["City", "Subscription"])
    df = df.drop(columns=["Gender"])
    return df

def scale_features(df, features):
    df = df.copy()
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def main():
    df = create_sample_dataframe()
    check_missing_values(df)
    df_clean = handle_missing_values(df)
    check_missing_values(df_clean)
    df_encoded = encode_categorical(df_clean)
    df_final = scale_features(df_encoded, ["Age", "Income"])
    out_path = "customer_processed.csv"
    df_final.to_csv(out_path, index=False)
    print(df_final)

if __name__ == "__main__":
    main()
