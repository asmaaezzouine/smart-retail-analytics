import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR /
    "reports" /
    "features_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR /
    "reports"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


print("\n===== DATASET INITIAL =====")

print(
    f"Nombre de lignes : {len(df)}"
)

print(
    f"Nombre de colonnes : {len(df.columns)}"
)


# ============================================================
# 3. TRAIN / TEST SPLIT INITIAL
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=None
)


print("\n===== TRAIN / TEST =====")

print(
    f"Train : {train_df.shape}"
)

print(
    f"Test  : {test_df.shape}"
)


# ============================================================
# 4. CRÉATION DE LA TARGET
# ============================================================

# Le seuil est appris uniquement sur TRAIN.
threshold = train_df["total_price"].median()


print("\n===== SEUIL DE CLASSIFICATION =====")

print(
    f"Médiane de total_price sur TRAIN : {threshold}"
)


train_df["high_value_sale"] = (
    train_df["total_price"] > threshold
).astype(int)


test_df["high_value_sale"] = (
    test_df["total_price"] > threshold
).astype(int)


print("\n===== DISTRIBUTION TARGET TRAIN =====")

print(
    train_df["high_value_sale"]
    .value_counts()
)


print("\n===== DISTRIBUTION TARGET TEST =====")

print(
    test_df["high_value_sale"]
    .value_counts()
)


print("\n===== POURCENTAGE TARGET TRAIN =====")

print(
    train_df["high_value_sale"]
    .value_counts(normalize=True)
    .mul(100)
)


print("\n===== POURCENTAGE TARGET TEST =====")

print(
    test_df["high_value_sale"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# 5. CALCUL DE LA MOYENNE DE PRIX PAR CATÉGORIE
# ============================================================

# IMPORTANT :
# La moyenne est calculée UNIQUEMENT sur TRAIN.

category_mean_price = (
    train_df
    .groupby("category")["price"]
    .mean()
)


print("\n===== MOYENNE DES PRIX PAR CATÉGORIE =====")

print(
    category_mean_price
)


# ============================================================
# 6. CRÉATION DE price_vs_category_mean
# ============================================================

train_df["category_mean_price"] = (
    train_df["category"]
    .map(category_mean_price)
)

test_df["category_mean_price"] = (
    test_df["category"]
    .map(category_mean_price)
)


train_df["price_vs_category_mean"] = (
    train_df["price"] /
    train_df["category_mean_price"]
)


test_df["price_vs_category_mean"] = (
    test_df["price"] /
    test_df["category_mean_price"]
)


# ============================================================
# 7. SÉLECTION DES FEATURES
# ============================================================

features = [
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "price_vs_category_mean",
    "log_price"
]

target = "high_value_sale"


X_train = train_df[features]

X_test = test_df[features]

y_train = train_df[target]

y_test = test_df[target]


# ============================================================
# 8. VÉRIFICATION DES FEATURES
# ============================================================

print("\n===== FEATURES =====")

print(
    X_train.columns.tolist()
)


print("\n===== TARGET =====")

print(target)


# ============================================================
# 9. VÉRIFICATION DES VALEURS MANQUANTES
# ============================================================

print("\n===== VALEURS MANQUANTES TRAIN =====")

print(
    X_train.isnull().sum()
)


print("\n===== VALEURS MANQUANTES TEST =====")

print(
    X_test.isnull().sum()
)


# ============================================================
# 10. VÉRIFICATION DES DIMENSIONS
# ============================================================

print("\n===== DIMENSIONS FINALES =====")

print(
    f"X_train : {X_train.shape}"
)

print(
    f"X_test  : {X_test.shape}"
)

print(
    f"y_train : {y_train.shape}"
)

print(
    f"y_test  : {y_test.shape}"
)


# ============================================================
# 11. DISTRIBUTION DES CLASSES
# ============================================================

print("\n===== DISTRIBUTION TRAIN =====")

print(
    y_train.value_counts(normalize=True)
)


print("\n===== DISTRIBUTION TEST =====")

print(
    y_test.value_counts(normalize=True)
)


# ============================================================
# 12. SAUVEGARDE
# ============================================================

X_train.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)


X_test.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)


y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)


y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# ============================================================
# 13. SAUVEGARDE DES DATASETS POUR TRAÇABILITÉ
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train_dataset.csv",
    index=False
)


test_df.to_csv(
    OUTPUT_DIR / "test_dataset.csv",
    index=False
)


# ============================================================
# 14. FIN
# ============================================================

print(
    "\n===== TRAIN / TEST SPLIT TERMINÉ ====="
)

print("Fichiers créés :")

print("- X_train.csv")
print("- X_test.csv")
print("- y_train.csv")
print("- y_test.csv")
print("- train_dataset.csv")
print("- test_dataset.csv")