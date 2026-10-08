from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

X_train = pd.read_csv(
    REPORTS_DIR / "X_train.csv"
)

y_train = pd.read_csv(
    REPORTS_DIR / "y_train.csv"
).squeeze()


print("\n===== DONNÉES =====")
print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")


# ============================================================
# 3. MODÈLE DE BASE
# ============================================================

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# 4. GRILLE DES HYPERPARAMÈTRES
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        5,
        10,
        15,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]
}


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)


print("\n===== RECHERCHE DES MEILLEURS PARAMÈTRES =====")

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 7. MEILLEURS PARAMÈTRES
# ============================================================

print("\n===== MEILLEURS PARAMÈTRES =====")

print(
    grid_search.best_params_
)


# ============================================================
# 8. MEILLEUR SCORE
# ============================================================

print("\n===== MEILLEUR ROC-AUC =====")

print(
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 9. MEILLEUR MODÈLE
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 10. SAUVEGARDE DU MODÈLE
# ============================================================

model_path = MODEL_DIR / "best_random_forest_model.pkl"

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# 11. SAUVEGARDE DES PARAMÈTRES
# ============================================================

best_params_df = pd.DataFrame(
    [
        {
            "parameter": key,
            "value": value
        }
        for key, value in grid_search.best_params_.items()
    ]
)

params_path = REPORTS_DIR / "best_random_forest_parameters.csv"

best_params_df.to_csv(
    params_path,
    index=False
)


# ============================================================
# 12. FIN
# ============================================================

print("\n===== PHASE 14 TERMINÉE =====")

print(
    f"Meilleur modèle sauvegardé : {model_path}"
)

print(
    f"Paramètres sauvegardés : {params_path}"
)