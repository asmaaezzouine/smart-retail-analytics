import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. Chemins du projet
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 2. Charger le modèle final
# ============================================================

model_path = MODEL_DIR / "best_gradient_boosting_model.pkl"

model = joblib.load(model_path)

print("Modèle chargé :", model_path.name)
print("Type de modèle :", type(model).__name__)


# ============================================================
# 3. Nouvelle transaction à prédire
# ============================================================

new_sale = pd.DataFrame({
    "year": [2026],
    "month": [5],
    "day": [10],
    "day_of_week": [6],
    "week_of_year": [19],
    "is_weekend": [1],
    "price_vs_category_mean": [1.5],
    "log_price": [5.5]
})


# ============================================================
# 4. Prédiction
# ============================================================

prediction = model.predict(new_sale)[0]

probabilities = model.predict_proba(new_sale)[0]

probability_normal = probabilities[0]
probability_high_value = probabilities[1]


# ============================================================
# 5. Affichage du résultat
# ============================================================

if prediction == 1:
    result = "High Value Sale"
else:
    result = "Normal Sale"


print("\n===== PREDICTION =====")
print("Résultat :", result)
print(f"Probabilité Normal Sale : {probability_normal:.2%}")
print(f"Probabilité High Value Sale : {probability_high_value:.2%}")