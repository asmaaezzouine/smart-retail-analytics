from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. CHARGEMENT DES DONNÉES TRAIN / TEST
# ============================================================

X_train_path = REPORTS_DIR / "X_train.csv"
X_test_path = REPORTS_DIR / "X_test.csv"
y_train_path = REPORTS_DIR / "y_train.csv"
y_test_path = REPORTS_DIR / "y_test.csv"

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

y_train = pd.read_csv(y_train_path).squeeze()
y_test = pd.read_csv(y_test_path).squeeze()


print("\n===== DONNÉES D'ENTRAÎNEMENT =====")
print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")

print("\n===== DONNÉES DE TEST =====")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# 3. VÉRIFICATION DES FEATURES
# ============================================================

print("\n===== FEATURES UTILISÉES =====")
print(X_train.columns.tolist())

print("\n===== VALEURS MANQUANTES =====")
print(X_train.isnull().sum())


# ============================================================
# 4. CRÉATION DU MODÈLE
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
# 5. ENTRAÎNEMENT
# ============================================================

print("\n===== ENTRAÎNEMENT DU MODÈLE =====")

model.fit(X_train, y_train)

print("Entraînement terminé.")


# ============================================================
# 6. PRÉDICTIONS SUR TRAIN
# ============================================================

train_predictions = model.predict(X_train)

print("\n===== EXEMPLE DE PRÉDICTIONS TRAIN =====")
print(train_predictions[:10])


# ============================================================
# 7. PRÉDICTIONS SUR TEST
# ============================================================

test_predictions = model.predict(X_test)

print("\n===== EXEMPLE DE PRÉDICTIONS TEST =====")
print(test_predictions[:10])


# ============================================================
# 8. IMPORTANCE DES FEATURES
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n===== IMPORTANCE DES FEATURES =====")
print(feature_importance)


# ============================================================
# 9. SAUVEGARDE DES IMPORTANCES
# ============================================================

importance_path = REPORTS_DIR / "feature_importance.csv"

feature_importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# 10. SAUVEGARDE DU MODÈLE
# ============================================================

model_path = MODEL_DIR / "random_forest_model.pkl"

joblib.dump(
    model,
    model_path
)


# ============================================================
# 11. FIN
# ============================================================

print("\n===== PHASE 10 TERMINÉE =====")

print(f"Modèle sauvegardé dans : {model_path}")
print(f"Importance des features : {importance_path}")