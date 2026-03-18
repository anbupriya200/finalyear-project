# preprocess.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    label_col = [c for c in df.columns if "label" in c.lower()][0]
    df["BinaryLabel"] = df[label_col].apply(
        lambda x: 0 if str(x).lower() in ["benign", "normal"] else 1
    )
    
    X = df.drop(columns=[label_col, "BinaryLabel"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["BinaryLabel"]
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "X_train_torch": X_train_torch,
        "X_test_torch": X_test_torch,
        "scaler": scaler
    }

if __name__ == "__main__":
    # Example usage
    data = preprocess_data("CICIDS_Merged_80K.csv")
    print("Training features shape:", data["X_train"].shape)
    print("Test features shape:", data["X_test"].shape)
    print("Training labels distribution:")
    print(pd.Series(data["y_train"]).value_counts())
    print("Test labels distribution:")
    print(pd.Series(data["y_test"]).value_counts())