from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 2. CHARGEMENT DES DONNÉES DE TEST
# ============================================================

X_test_path = REPORTS_DIR / "X_test.csv"
y_test_path = REPORTS_DIR / "y_test.csv"

X_test = pd.read_csv(X_test_path)

y_test = pd.read_csv(
    y_test_path
).squeeze()


print("\n===== DONNÉES DE TEST =====")

print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# 3. VÉRIFICATION DES FEATURES
# ============================================================

print("\n===== FEATURES UTILISÉES =====")

print(X_test.columns.tolist())


print("\n===== VALEURS MANQUANTES =====")

print(X_test.isnull().sum())


# ============================================================
# 4. CHARGEMENT DU MODÈLE OPTIMISÉ
# ============================================================

model_path = (
    MODEL_DIR /
    "best_gradient_boosting_model.pkl"
)

model = joblib.load(model_path)


print("\n===== MODÈLE FINAL =====")

print(
    f"Modèle chargé depuis : {model_path}"
)

print(
    f"Type de modèle : {type(model).__name__}"
)


# ============================================================
# 5. PRÉDICTIONS
# ============================================================

print("\n===== GÉNÉRATION DES PRÉDICTIONS =====")

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]


print("Prédictions générées.")


# ============================================================
# 6. CALCUL DES MÉTRIQUES
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_proba
)


# ============================================================
# 7. AFFICHAGE DES MÉTRIQUES
# ============================================================

print("\n===== MÉTRIQUES DU MODÈLE FINAL =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 8. MATRICE DE CONFUSION
# ============================================================

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)


print("\n===== MATRICE DE CONFUSION =====")

print(conf_matrix)


# ============================================================
# 9. RAPPORT DE CLASSIFICATION
# ============================================================

print("\n===== RAPPORT DE CLASSIFICATION =====")

classification_result = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal Sale",
        "High Value Sale"
    ],
    zero_division=0
)

print(classification_result)


# ============================================================
# 10. SAUVEGARDE DES MÉTRIQUES
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})


metrics_path = (
    REPORTS_DIR /
    "final_model_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 11. SAUVEGARDE DE LA MATRICE DE CONFUSION
# ============================================================

confusion_df = pd.DataFrame(
    conf_matrix,
    index=[
        "Actual_Normal",
        "Actual_High_Value"
    ],
    columns=[
        "Predicted_Normal",
        "Predicted_High_Value"
    ]
)


confusion_path = (
    REPORTS_DIR /
    "final_confusion_matrix.csv"
)

confusion_df.to_csv(
    confusion_path
)


# ============================================================
# 12. SAUVEGARDE DES PRÉDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "probability_high_value": y_proba
})


predictions_path = (
    REPORTS_DIR /
    "final_model_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 13. FIN
# ============================================================

print("\n===== ÉVALUATION FINALE TERMINÉE =====")

print(
    f"Métriques sauvegardées : {metrics_path}"
)

print(
    f"Matrice de confusion : {confusion_path}"
)

print(
    f"Prédictions sauvegardées : {predictions_path}"
)