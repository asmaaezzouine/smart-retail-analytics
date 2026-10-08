from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)


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
# 3. DÉFINITION DES MODÈLES
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 5. COMPARAISON
# ============================================================

results = []

print("\n===== COMPARAISON DES MODÈLES =====")

for model_name, model in models.items():

    print(f"\nModèle : {model_name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    result = {
        "model": model_name,
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "precision_mean": scores["test_precision"].mean(),
        "recall_mean": scores["test_recall"].mean(),
        "f1_mean": scores["test_f1"].mean(),
        "roc_auc_mean": scores["test_roc_auc"].mean()
    }

    results.append(result)

    print(
        f"Accuracy : {result['accuracy_mean']:.4f}"
    )

    print(
        f"Precision : {result['precision_mean']:.4f}"
    )

    print(
        f"Recall : {result['recall_mean']:.4f}"
    )

    print(
        f"F1 : {result['f1_mean']:.4f}"
    )

    print(
        f"ROC-AUC : {result['roc_auc_mean']:.4f}"
    )


# ============================================================
# 6. TABLEAU FINAL
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="roc_auc_mean",
    ascending=False
)

print("\n===== CLASSEMENT DES MODÈLES =====")

print(results_df)


# ============================================================
# 7. SAUVEGARDE
# ============================================================

output_path = REPORTS_DIR / "model_comparison.csv"

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 8. MEILLEUR MODÈLE
# ============================================================

best_model = results_df.iloc[0]

print("\n===== MEILLEUR MODÈLE =====")

print(
    f"Modèle : {best_model['model']}"
)

print(
    f"ROC-AUC : {best_model['roc_auc_mean']:.4f}"
)


# ============================================================
# 9. FIN
# ============================================================

print("\n===== PHASE 13 TERMINÉE =====")
print(f"Résultats sauvegardés dans : {output_path}")