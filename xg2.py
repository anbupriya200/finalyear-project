# =====================================================
# WADE: VAE + XGBoost Intrusion Detection (Corrected)
# =====================================================

# -------------------- IMPORTS --------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from xgboost import XGBClassifier

# -------------------- DEVICE --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# STEP 1: LOAD DATA
# =====================================================
print("Loading dataset...")

df = pd.read_csv("CICIDS_Merged_80K.csv")
df.columns = df.columns.str.strip()

# Find label column automatically
label_col = [c for c in df.columns if "label" in c.lower()][0]

# Convert into binary (0 = Normal, 1 = Attack)
df["BinaryLabel"] = df[label_col].apply(
    lambda x: 0 if str(x).lower() in ["benign","normal"] else 1
)

# Keep only numeric features
X = df.drop(columns=[label_col,"BinaryLabel"],errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df["BinaryLabel"]

# Clean data
X.replace([np.inf,-np.inf],np.nan,inplace=True)
X.fillna(0,inplace=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# STEP 2: TRAIN-TEST SPLIT
# =====================================================
X_train,X_test,y_train,y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to PyTorch tensors for VAE
X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

y_train_np = y_train.values
y_test_np = y_test.values

print("Training samples:", len(X_train))

# =====================================================
# STEP 3: VAE (ANOMALY DETECTION)
# =====================================================
class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

vae = VAE(X_train.shape[1]).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("\nTraining VAE (only on normal data)...")
X_normal = X_train_torch[y_train_np==0]

for epoch in range(40):
    recon = vae(X_normal)
    loss = loss_fn(recon, X_normal)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# =====================================================
# STEP 4: XGBOOST (CLASSIFICATION)
# =====================================================
print("\nTraining XGBoost...")

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
# STEP 5: PREDICTION
# =====================================================
# XGBoost probabilities
xgb_probs = xgb_model.predict_proba(X_test)[:,1]

# VAE reconstruction errors
with torch.no_grad():
    recon_test = vae(X_test_torch)
    vae_error = torch.mean((X_test_torch - recon_test)**2, dim=1).cpu().numpy()

# =====================================================
# STEP 6: CORRECT FINAL SCORE (WADE Style)
# =====================================================
# Following WADE: Final_Score = rit + ret + MSR optimization

r_prev = 0
sigma = 1e-8
eta = 2
lambda_val = 0
final_scores = []

for i in range(len(X_test)):
    rit = vae_error[i]

    # External reward (ret)
    if y_test_np[i] == 1 and xgb_probs[i] >= 0.5:      # Correct attack
        ret = 1
    elif y_test_np[i] == 1 and xgb_probs[i] < 0.5:     # Missed attack
        ret = -1
    elif y_test_np[i] == 0 and xgb_probs[i] >= 0.5:    # False positive
        ret = rit - (1 - rit)
    else:                                               # Correct normal
        ret = (1 - rit) - rit

    # Total reward before MSR
    rt = rit + ret

    # MSR reward optimizer
    rho = (rt - r_prev) / min(abs(rt + sigma), abs(r_prev + sigma))
    x = rho + np.sign(rt - r_prev)
    kx = np.arctan((x * np.pi) / (2 * eta))
    r_star = rt + (kx - lambda_val) * abs(rt)

    final_scores.append(r_star)
    r_prev = rt

final_scores = np.array(final_scores)

# Threshold for classification
threshold = np.percentile(final_scores, 85)
y_pred = (final_scores >= threshold).astype(int)

# =====================================================
# STEP 7: EVALUATION
# =====================================================
accuracy = accuracy_score(y_test_np, y_pred)
precision = precision_score(y_test_np, y_pred, zero_division=0)
recall = recall_score(y_test_np, y_pred, zero_division=0)
f1 = f1_score(y_test_np, y_pred, zero_division=0)
roc = roc_auc_score(y_test_np, final_scores)

print("\n===== RESULTS =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc)

cm = confusion_matrix(y_test_np, y_pred)

# =====================================================
# STEP 8: VISUALIZATION
# =====================================================
plt.figure(figsize=(10,5))
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.title("Feature Importance (XGBoost)")
plt.show()

fpr, tpr, _ = roc_curve(y_test_np, final_scores)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.show()

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================================
# STEP 9: SAVE OUTPUT
# =====================================================
label_map = {0:"Normal",1:"Attack"}

output = pd.DataFrame({
    "Actual_Label":[label_map[i] for i in y_test_np],
    "Predicted_Label":[label_map[i] for i in y_pred],
    "Attack_Probability": xgb_probs,
    "VAE_Error": vae_error,
    "Final_Score": final_scores
})

output.to_csv("WADE_Attack_Detection_Corrected.csv", index=False)
print("\nSaved: WADE_Attack_Detection_Corrected.csv")
print("\nResults saved successfully!")