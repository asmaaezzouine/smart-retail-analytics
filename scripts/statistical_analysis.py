import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Le fichier .env se trouve à la racine du projet.
# __file__ correspond à :
# smart-retail-analytics/scripts/statistical_analysis.py
#
# parent.parent permet donc de remonter à :
# smart-retail-analytics/

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


# Récupération des informations de connexion
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


# Vérification simple de la configuration
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
    f"?client_encoding=utf8"
    
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


# ============================================================
# 4. PRÉPARATION DES DONNÉES
# ============================================================

# Conversion de la date
df["sale_date"] = pd.to_datetime(df["sale_date"])


# Vérification du nombre de valeurs manquantes
missing_values = df.isnull().sum()

print("\n===== VALEURS MANQUANTES =====")
print(missing_values)


# ============================================================
# 5. STATISTICAL ANALYSIS
# ============================================================

numerical_columns = [
    "price",
    "quantity",
    "total_price"
]


# ------------------------------------------------------------
# 5.1 Statistiques descriptives
# ------------------------------------------------------------

descriptive_statistics = df[numerical_columns].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print("\n===== STATISTIQUES DESCRIPTIVES =====")
print(descriptive_statistics)


# ------------------------------------------------------------
# 5.2 Skewness
# ------------------------------------------------------------

skewness = df[numerical_columns].skew()

print("\n===== SKEWNESS =====")
print(skewness)


# ------------------------------------------------------------
# 5.3 Quartiles
# ------------------------------------------------------------

quartiles = df[numerical_columns].quantile(
    [0.25, 0.50, 0.75]
)

print("\n===== QUARTILES =====")
print(quartiles)


# ------------------------------------------------------------
# 5.4 IQR et détection des outliers
# ------------------------------------------------------------

q1 = df[numerical_columns].quantile(0.25)
q3 = df[numerical_columns].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (
    (df[numerical_columns] < lower_bound)
    | (df[numerical_columns] > upper_bound)
)

outlier_counts = outlier_mask.sum()

print("\n===== NOMBRE D'OUTLIERS =====")
print(outlier_counts)


# ============================================================
# 6. SAUVEGARDE DES RÉSULTATS
# ============================================================

reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(exist_ok=True)


# Statistiques descriptives
descriptive_statistics.to_csv(
    reports_dir / "descriptive_statistics.csv"
)


# Skewness
skewness.to_csv(
    reports_dir / "skewness.csv",
    header=["skewness"]
)


# Quartiles
quartiles.to_csv(
    reports_dir / "quartiles.csv"
)


# Outliers
outlier_counts.to_csv(
    reports_dir / "outlier_counts.csv",
    header=["outlier_count"]
)


# Dataset utilisé pour l'analyse
df.to_csv(
    reports_dir / "statistical_analysis_dataset.csv",
    index=False
)


print("\n===== ANALYSE TERMINÉE =====")
print(f"Nombre de lignes analysées : {len(df)}")
print(f"Résultats sauvegardés dans : {reports_dir}")