# module3.py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# Import from previous modules (optional, for standalone testing)
from modul1 import preprocess_data
from module2 import build_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_models(data, models, threshold_percentile=85):
    """
    data : dict from preprocess_data() containing test data
    models : dict from build_models() containing 'vae' and 'xgb'
    threshold_percentile : percentile to use for final score threshold
    Returns: dict with metrics, final_scores, y_pred, and saves CSV
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_test_torch = data["X_test_torch"]

    vae = models['vae']
    xgb = models['xgb']

    # 1. XGBoost probabilities
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    # 2. VAE reconstruction errors
    vae.eval()
    with torch.no_grad():
        recon_test = vae(X_test_torch)
        vae_error = torch.mean((X_test_torch - recon_test) ** 2, dim=1).cpu().numpy()

    # 3. WADE final score calculation
    r_prev = 0
    sigma = 1e-8
    eta = 2
    lambda_val = 0
    final_scores = []

    for i in range(len(X_test)):
        rit = vae_error[i]

        # External reward (ret)
        if y_test[i] == 1 and xgb_probs[i] >= 0.5:      # Correct attack
            ret = 1
        elif y_test[i] == 1 and xgb_probs[i] < 0.5:     # Missed attack
            ret = -1
        elif y_test[i] == 0 and xgb_probs[i] >= 0.5:    # False positive
            ret = rit - (1 - rit)
        else:                                            # Correct normal
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

    # 4. Convert to binary predictions
    threshold = np.percentile(final_scores, threshold_percentile)
    y_pred = (final_scores >= threshold).astype(int)

    # 5. Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, final_scores)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== RESULTS =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc:.4f}")

    # 6. Visualizations
    # Feature importance
    plt.figure(figsize=(10,5))
    plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
    plt.title("Feature Importance (XGBoost)")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, final_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc:.3f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Confusion matrix heatmap
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 7. Save output CSV
    label_map = {0: "Normal", 1: "Attack"}
    output = pd.DataFrame({
        "Actual_Label": [label_map[i] for i in y_test],
        "Predicted_Label": [label_map[i] for i in y_pred],
        "Attack_Probability": xgb_probs,
        "VAE_Error": vae_error,
        "Final_Score": final_scores
    })
    output.to_csv("WADE_Attack_Detection_Corrected.csv", index=False)
    print("\nSaved: WADE_Attack_Detection_Corrected.csv")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
        'confusion_matrix': cm,
        'final_scores': final_scores,
        'y_pred': y_pred
    }

if __name__ == "__main__":
    # Test the full pipeline
    data = preprocess_data("CICIDS_Merged_80K.csv")
    models = build_models(data)
    metrics = evaluate_models(data, models)