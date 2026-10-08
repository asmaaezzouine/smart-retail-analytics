import os
from pathlib import Path
import math

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


# Vérification de la configuration
required_variables = {
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
}

missing_variables = [
    name
    for name, value in required_variables.items()
    if value is None
]

if missing_variables:
    raise ValueError(
        f"Variables manquantes dans le fichier .env : "
        f"{', '.join(missing_variables)}"
    )


# ============================================================
# 2. CONNEXION À POSTGRESQL
# ============================================================

connection_string = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_string)


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

query = """
SELECT
    s.sale_id,
    s.product_id,
    p.name,
    p.category,
    p.price,
    s.quantity,
    s.sale_date,
    s.total_price
FROM sales AS s
LEFT JOIN products AS p
    ON s.product_id = p.product_id
ORDER BY s.sale_date;
"""

df = pd.read_sql(query, engine)


print("\n===== DONNÉES BRUTES =====")

print(df.head())

print(f"\nNombre de lignes : {len(df)}")


# ============================================================
# 4. CONVERSION DES TYPES
# ============================================================

df["sale_date"] = pd.to_datetime(
    df["sale_date"]
)


# ============================================================
# 5. FEATURE ENGINEERING TEMPOREL
# ============================================================

# Année
df["year"] = df["sale_date"].dt.year

# Mois
df["month"] = df["sale_date"].dt.month

# Jour
df["day"] = df["sale_date"].dt.day

# Jour de la semaine
# 0 = lundi
# 6 = dimanche
df["day_of_week"] = (
    df["sale_date"].dt.dayofweek
)

# Semaine de l'année
df["week_of_year"] = (
    df["sale_date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Weekend
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# 6. FEATURE ENGINEERING QUANTITATIF
# ============================================================

# Transformation logarithmique du prix
df["log_price"] = (
    df["price"]
    .clip(lower=0)
    .apply(math.log1p)
)


# Transformation logarithmique de la quantité
df["log_quantity"] = (
    df["quantity"]
    .clip(lower=0)
    .apply(math.log1p)
)


# ============================================================
# 7. VÉRIFICATION DES FEATURES
# ============================================================

print("\n===== COLONNES APRÈS FEATURE ENGINEERING =====")

print(df.columns.tolist())


print("\n===== APERÇU DES FEATURES =====")

feature_columns = [
    "sale_date",
    "price",
    "quantity",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "log_price",
    "log_quantity"
]

print(
    df[feature_columns].head()
)


# ============================================================
# 8. VÉRIFICATION DES VALEURS MANQUANTES
# ============================================================

print("\n===== VALEURS MANQUANTES =====")

print(
    df.isnull().sum()
)


# ============================================================
# 9. SAUVEGARDE
# ============================================================

reports_dir = BASE_DIR / "reports"

reports_dir.mkdir(
    exist_ok=True
)


output_path = (
    reports_dir /
    "features_dataset.csv"
)


df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 10. FIN
# ============================================================

print(
    "\n===== FEATURE ENGINEERING TERMINÉ ====="
)

print(
    f"Nombre de lignes : {len(df)}"
)

print(
    f"Nombre de colonnes : {len(df.columns)}"
)

print(
    f"Fichier créé : {output_path}"
)