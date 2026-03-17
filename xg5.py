# =====================================================
# Hybrid VAE + XGBoost Intrusion Detection
# =====================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- DEVICE --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =====================================================
# STEP 1: LOAD AND PREPROCESS DATA
# =====================================================
print("\nLoading dataset...")
df = pd.read_csv("CICIDS_Merged_80K.csv")
df.columns = df.columns.str.strip()

# Find label column automatically
label_col = [c for c in df.columns if "label" in c.lower()][0]

# Convert into binary (0 = Normal, 1 = Attack)
df["BinaryLabel"] = df[label_col].apply(lambda x: 0 if str(x).lower() in ["benign","normal"] else 1)

# Keep only numeric features
X = df.drop(columns=[label_col,"BinaryLabel"], errors="ignore").select_dtypes(include=[np.number])
y = df["BinaryLabel"]

# Clean data
X.replace([np.inf,-np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_train_np = y_train.values
y_test_np = y_test.values

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =====================================================
# STEP 2: DEFINE VAE MODEL
# =====================================================
class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

vae = VAE(X_train.shape[1]).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# =====================================================
# STEP 3: TRAIN VAE ON NORMAL DATA
# =====================================================
print("\nTraining VAE on normal traffic...")
X_normal = X_train_torch[y_train_np==0]

epochs = 40
for epoch in range(epochs):
    recon = vae(X_normal)
    loss = loss_fn(recon, X_normal)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{epochs} Loss: {loss.item():.4f}")

# =====================================================
# STEP 4: TRAIN XGBOOST CLASSIFIER
# =====================================================
print("\nTraining XGBoost classifier...")
scale_pos_weight = (len(y_train_np) - sum(y_train_np)) / sum(y_train_np)
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train_np)

# =====================================================
# STEP 5: PREDICTION AND FINAL SCORE CALCULATION
# =====================================================
print("\nGenerating predictions...")
# XGBoost probabilities
xgb_probs = xgb_model.predict_proba(X_test)[:,1]

# VAE reconstruction errors
with torch.no_grad():
    recon_test = vae(X_test_torch)
    vae_error = torch.mean((X_test_torch - recon_test)**2, dim=1).cpu().numpy()

# WADE-style final score
final_scores = vae_error + xgb_probs
threshold = np.percentile(final_scores, 85)
y_pred = (final_scores >= threshold).astype(int)

# =====================================================
# STEP 6: EVALUATION
# =====================================================
accuracy = accuracy_score(y_test_np, y_pred)
precision = precision_score(y_test_np, y_pred)
recall = recall_score(y_test_np, y_pred)
f1 = f1_score(y_test_np, y_pred)
roc = roc_auc_score(y_test_np, final_scores)

print("\n===== RESULTS =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc:.4f}")

cm = confusion_matrix(y_test_np, y_pred)

# =====================================================
# STEP 7: VISUALIZATION
# =====================================================
# Feature importance
plt.figure(figsize=(12,5))
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test_np, final_scores)
plt.figure()
plt.plot(fpr, tpr, label="Hybrid VAE+XGBoost")
plt.plot([0,1],[0,1],'--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =====================================================
# STEP 8: SAVE OUTPUT
# =====================================================
label_map = {0:"Normal",1:"Attack"}
output = pd.DataFrame({
    "Actual_Label":[label_map[i] for i in y_test_np],
    "Predicted_Label":[label_map[i] for i in y_pred],
    "Attack_Probability": xgb_probs,
    "VAE_Error": vae_error,
    "Final_Score": final_scores
})
output.to_csv("Hybrid_VAE_XGBoost_Detection.csv", index=False)
print("\nSaved results to 'Hybrid_VAE_XGBoost_Detection.csv'")