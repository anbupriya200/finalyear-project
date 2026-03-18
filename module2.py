# module21.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from xgboost import XGBClassifier

# Import preprocessing function (optional, for standalone testing)
from modul1 import preprocess_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- VAE (Autoencoder) Definition ----------
class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ---------- Main Training Function ----------
def build_models(data, vae_epochs=40, vae_lr=0.001, xgb_params=None):
    """
    data : dict from preprocess_data() containing:
        - X_train, y_train
        - X_train_torch
    Returns dict with trained models: 'vae', 'xgb'
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_train_torch = data["X_train_torch"]

    input_dim = X_train.shape[1]

    # 1. Train VAE on normal samples only
    vae = VAE(input_dim).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    loss_fn = nn.MSELoss()

    # Extract normal samples (label 0)
    normal_mask = (y_train == 0)
    X_normal = X_train_torch[normal_mask]

    print(f"Training VAE on {len(X_normal)} normal samples...")
    vae.train()
    for epoch in range(vae_epochs):
        recon = vae(X_normal)
        loss = loss_fn(recon, X_normal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}, Loss: {loss.item():.6f}")

    # --- Added: Print reconstruction error on normal training data ---
    vae.eval()
    with torch.no_grad():
        recon_normal = vae(X_normal)
        recon_error = torch.mean((X_normal - recon_normal) ** 2, dim=1).cpu().numpy()
    print(f"\nAverage reconstruction error on normal training data: {recon_error.mean():.6f} (+/- {recon_error.std():.6f})")
    # ----------------------------------------------------------------

    # 2. Train XGBoost on all training data
    print("\nTraining XGBoost...")
    default_xgb_params = {
        'n_estimators': 400,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'eval_metric': 'logloss'
    }
    if xgb_params:
        default_xgb_params.update(xgb_params)

    # Compute scale_pos_weight for imbalance
    n_normal = np.sum(y_train == 0)
    n_attack = np.sum(y_train == 1)
    scale_pos_weight = n_normal / n_attack if n_attack > 0 else 1.0
    default_xgb_params['scale_pos_weight'] = scale_pos_weight

    xgb_model = XGBClassifier(**default_xgb_params)
    xgb_model.fit(X_train, y_train)

    # --- Added: Print XGBoost probabilities on training data ---
    train_probs = xgb_model.predict_proba(X_train)[:, 1]  # probability of attack
    normal_probs = train_probs[y_train == 0]
    attack_probs = train_probs[y_train == 1]
    print(f"\nAverage predicted attack probability for normal samples: {normal_probs.mean():.4f}")
    print(f"Average predicted attack probability for attack samples: {attack_probs.mean():.4f}")
    print(f"Sample probabilities (first 5 normal): {normal_probs[:5]}")
    print(f"Sample probabilities (first 5 attack): {attack_probs[:5]}")
    # -----------------------------------------------------------

    return {
        'vae': vae,
        'xgb': xgb_model
    }

if __name__ == "__main__":
    # Test the module
    data = preprocess_data("CICIDS_Merged_80K.csv")
    models = build_models(data)
    print("\nModels trained successfully!")
    print(f"VAE device: {next(models['vae'].parameters()).device}")
    print(f"XGBoost classes: {models['xgb'].classes_}")