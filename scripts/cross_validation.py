from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"

X_TRAIN_PATH = REPORTS_DIR / "X_train.csv"
Y_TRAIN_PATH = REPORTS_DIR / "y_train.csv"


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

print("\n===== DONNÉES =====")
print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")


# ============================================================
# 3. CRÉATION DU MODÈLE
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced"
)


# ============================================================
# 4. STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("\n===== CROSS-VALIDATION =====")

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)


# ============================================================
# 6. AFFICHAGE DES RÉSULTATS
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

results = []

for metric in metrics:

    scores = cv_results[f"test_{metric}"]

    mean_score = scores.mean()
    std_score = scores.std()

    results.append({
        "metric": metric,
        "mean": mean_score,
        "std": std_score
    })

    print(
        f"{metric.upper():10s} "
        f"Mean = {mean_score:.4f} "
        f"| Std = {std_score:.4f}"
    )


# ============================================================
# 7. SAUVEGARDE
# ============================================================

cv_results_df = pd.DataFrame(results)

output_path = REPORTS_DIR / "cross_validation_results.csv"

cv_results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 8. FIN
# ============================================================

print("\n===== PHASE 12 TERMINÉE =====")
print(f"Résultats sauvegardés dans : {output_path}")