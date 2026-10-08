import os
from pathlib import Path

import html
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Retail Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_gradient_boosting_model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "final_model_metrics.csv"
CONFUSION_PATH = BASE_DIR / "reports" / "final_confusion_matrix.csv"

FEATURE_NAMES = [
    "year", "month", "day", "day_of_week", "week_of_year",
    "is_weekend", "price_vs_category_mean", "log_price",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Bootstrap 5 (CDN). Les composants Bootstrap sont rendus dans des
# iframes isolées (components.html) pour ne pas entrer en conflit avec le CSS de Streamlit.
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
# Palette Bootstrap pour les graphiques Plotly
BS_COLORS = ["#0d6efd", "#198754", "#ffc107", "#dc3545", "#0dcaf0", "#6f42c1", "#fd7e14", "#20c997"]

CSS = """
<style>
.stApp { background-color: #f8f9fa; }
h1, h2, h3 { font-weight: 700; }

/* Sidebar : thème "dark" Bootstrap */
section[data-testid="stSidebar"] { background-color: #212529; }
section[data-testid="stSidebar"] * { color: #f8f9fa; }
/* Champs de la sidebar : fond sombre pour que le texte blanc reste lisible */
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #343a40 !important;
    border-color: #495057 !important;
}
section[data-testid="stSidebar"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background-color: #343a40 !important;
}
section[data-testid="stSidebar"] button {
    background-color: #343a40 !important;
    border: 1px solid #495057 !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] button:hover {
    background-color: #0d6efd !important;
    border-color: #0d6efd !important;
}
.sidebar-title { font-size: 22px; font-weight: 700; margin-bottom: 5px; }
.sidebar-subtitle { font-size: 13px; color: #adb5bd !important; margin-bottom: 20px; }
.badge {
    display: inline-block; padding: .35em .65em; border-radius: 50rem;
    font-size: .75em; font-weight: 700; background-color: #0d6efd; color: #fff !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# BASE DE DONNÉES
# ============================================================

@st.cache_resource
def get_engine():
    load_dotenv(BASE_DIR / ".env")
    params = {k: os.getenv(k) for k in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")}
    missing = [k for k, v in params.items() if not v]
    if missing:
        st.error(f"Variables d'environnement manquantes dans .env : {', '.join(missing)}")
        st.stop()

    # URL.create gère correctement les caractères spéciaux du mot de passe
    url = URL.create(
        "postgresql",
        username=params["DB_USER"],
        password=params["DB_PASSWORD"],
        host=params["DB_HOST"],
        port=int(params["DB_PORT"]),
        database=params["DB_NAME"],
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=3600, show_spinner="Chargement des données…")
def load_data() -> pd.DataFrame:
    # Jointure faite côté SQL ; INNER JOIN pour exclure les produits jamais vendus
    # (le LEFT JOIN d'origine créait des lignes vides qui faussaient les statistiques).
    query = """
        SELECT s.sale_id, s.sale_date, s.quantity, s.total_price,
               p.product_id, p.name, p.category, p.price
        FROM sales s
        INNER JOIN products p ON p.product_id = s.product_id
    """
    data = pd.read_sql(query, get_engine())
    data["sale_date"] = pd.to_datetime(data["sale_date"])
    data["year_month"] = data["sale_date"].dt.to_period("M").dt.to_timestamp()
    data["week_start"] = data["sale_date"].dt.to_period("W").dt.start_time
    data["day_of_week"] = data["sale_date"].dt.day_name()
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


try:
    df_all = load_data()
except Exception as exc:  # connexion, requête, table absente…
    st.error("Impossible de charger les données depuis PostgreSQL.")
    st.exception(exc)
    st.stop()

if df_all.empty:
    st.warning("Aucune vente trouvée dans la base de données.")
    st.stop()


# ============================================================
# COMPOSANTS UI
# ============================================================

def bs_html(body: str, height: int) -> None:
    """Affiche un bloc HTML avec Bootstrap 5 chargé."""
    components.html(
        f"""<!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="{BOOTSTRAP_CSS}" rel="stylesheet">
        <style>body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; }}</style>
        </head><body><div class="container-fluid p-1">{body}</div></body></html>""",
        height=height,
    )


def page_header(title: str, subtitle: str) -> None:
    bs_html(
        f"""<div class="card border-0 shadow-sm">
              <div class="card-body p-4">
                <h1 class="h2 fw-bold mb-1">{html.escape(title)}</h1>
                <p class="text-muted mb-0">{html.escape(subtitle)}</p>
              </div>
            </div>""",
        height=130,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items = [(titre, valeur, couleur Bootstrap)]"""
    cols = "".join(
        f"""<div class="col-12 col-sm-6 col-lg">
              <div class="card border-0 border-start border-4 border-{color} shadow-sm h-100">
                <div class="card-body">
                  <div class="text-muted small text-uppercase">{html.escape(title)}</div>
                  <div class="fs-4 fw-bold text-break">{html.escape(value)}</div>
                </div>
              </div>
            </div>"""
        for title, value, color in items
    )
    bs_html(f'<div class="row g-3">{cols}</div>', height=110)


def show_fig(fig, height: int | None = None) -> None:
    """Point d'entrée unique pour l'affichage des graphiques (style cohérent)."""
    fig.update_layout(template="plotly_white", colorway=BS_COLORS, margin=dict(l=10, r=10, t=50, b=10))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SIDEBAR + FILTRES GLOBAUX
# ============================================================

with st.sidebar:
    st.markdown(
        """<div class="sidebar-title">Smart Retail</div>
        <div class="sidebar-subtitle">Retail Analytics Platform</div>""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Dashboard", "Sales Analysis", "Exploratory Analysis",
         "Machine Learning", "Prediction"],
    )

    st.markdown("---")
    st.markdown("**Filtres**")

    min_d, max_d = df_all["sale_date"].min().date(), df_all["sale_date"].max().date()
    date_range = st.date_input("Période", (min_d, max_d), min_value=min_d, max_value=max_d)
    categories = sorted(df_all["category"].dropna().unique())
    selected_cats = st.multiselect("Catégories", categories, default=categories)

    if st.button("Rafraîchir les données", use_container_width=True):
        load_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown('<span class="badge">Data Science Project</span>', unsafe_allow_html=True)

# date_input renvoie un seul élément tant que la plage n'est pas complète
start, end = (date_range if len(date_range) == 2 else (date_range[0], max_d))
df = df_all[
    df_all["sale_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    & df_all["category"].isin(selected_cats)
]

if df.empty and page != "Prediction":
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()


# ============================================================
# PAGES
# ============================================================

def page_dashboard() -> None:
    page_header("Smart Retail Analytics", "Vue globale des performances commerciales")

    kpi_row([
        ("Total Revenue", f"${df['total_price'].sum():,.0f}", "success"),
        ("Total Sales", f"{df['sale_id'].nunique():,}", "primary"),
        ("Average Order", f"${df['total_price'].mean():,.2f}", "info"),
        ("Best Category", str(df.groupby("category")["total_price"].sum().idxmax()).title(),
         "warning"),
        ("Units Sold", f"{df['quantity'].sum():,}", "secondary"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.subheader("Revenue by Category")
        data = df.groupby("category", as_index=False)["total_price"].sum().sort_values(
            "total_price", ascending=False)
        show_fig(px.bar(data, x="category", y="total_price",
                        labels={"category": "Catégorie", "total_price": "Revenu"}), 400)

    with right:
        st.subheader("Daily Revenue")
        data = df.groupby("sale_date", as_index=False)["total_price"].sum()
        data["moyenne_7j"] = data["total_price"].rolling(7, min_periods=1).mean()
        fig = px.line(data, x="sale_date", y=["total_price", "moyenne_7j"],
                      labels={"sale_date": "Date", "value": "Revenu", "variable": ""})
        show_fig(fig, 400)

    st.subheader("Top 10 Products")
    metric = st.radio("Classer par", ["Quantité", "Revenu"], horizontal=True)
    col_name = "quantity" if metric == "Quantité" else "total_price"
    top = (df.groupby("name")[col_name].sum().sort_values(ascending=False)
           .head(10).sort_values().reset_index())
    show_fig(px.bar(top, x=col_name, y="name", orientation="h",
                    labels={col_name: metric, "name": "Produit"}), 450)


def page_sales() -> None:
    page_header("Sales Analysis", "Analyse des produits, catégories et tendances de revenus")

    st.subheader("Category Performance")
    analysis = (
        df.groupby("category")
        .agg(revenue=("total_price", "sum"), quantity=("quantity", "sum"),
             transactions=("sale_id", "count"), average_price=("price", "mean"))
        .sort_values("revenue", ascending=False)
    )
    analysis["part_du_CA"] = analysis["revenue"] / analysis["revenue"].sum()
    st.dataframe(
        analysis.style.format({
            "revenue": "{:,.2f}", "quantity": "{:,.0f}", "transactions": "{:,.0f}",
            "average_price": "{:,.2f}", "part_du_CA": "{:.1%}",
        }),
        use_container_width=True,
    )
    st.download_button("Exporter (CSV)", analysis.to_csv().encode("utf-8"),
                       "category_performance.csv", "text/csv")

    st.subheader("Monthly Revenue")
    monthly = df.groupby("year_month", as_index=False)["total_price"].sum()
    show_fig(px.bar(monthly, x="year_month", y="total_price",
                    labels={"year_month": "Mois", "total_price": "Revenu"}))

    st.subheader("Weekly Revenue")
    weekly = df.groupby("week_start", as_index=False)["total_price"].sum()
    show_fig(px.line(weekly, x="week_start", y="total_price", markers=True,
                     labels={"week_start": "Semaine", "total_price": "Revenu"}))

    st.subheader("Revenue by Day of Week")
    dow = (df.groupby("day_of_week")["total_price"].sum()
           .reindex(DAY_ORDER).fillna(0).reset_index())
    show_fig(px.bar(dow, x="day_of_week", y="total_price",
                    labels={"day_of_week": "Jour", "total_price": "Revenu"}))


def page_eda() -> None:
    page_header("Exploratory Data Analysis", "Analyse statistique et exploration des données")
    num_cols = ["price", "quantity", "total_price"]
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Statistics", "Correlation", "Distributions", "Data Quality"])

    with tab1:
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

    with tab2:
        corr = df[num_cols].corr()
        show_fig(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                           color_continuous_scale="RdBu_r", aspect="auto"))
        st.info("La corrélation entre price et total_price est élevée car "
                "total_price dépend directement de price et quantity.")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            show_fig(px.histogram(df, x="quantity", nbins=10,
                                  title="Distribution des quantités vendues"))
        with c2:
            show_fig(px.histogram(df, x="total_price", nbins=30,
                                  title="Distribution du revenu par transaction"))
        st.subheader("Price vs Revenue")
        sample = df.sample(min(len(df), 5000), random_state=42)  # évite un graphique trop lourd
        show_fig(px.scatter(sample, x="price", y="total_price", color="category",
                            hover_data=["name", "quantity"]))
        if len(sample) < len(df):
            st.caption(f"Échantillon aléatoire de {len(sample):,} points sur {len(df):,}.")

    with tab4:
        st.subheader("Valeurs manquantes")
        missing = df.isnull().sum().rename_axis("column").reset_index(name="missing_values")
        st.dataframe(missing, use_container_width=True)
        st.write(f"Lignes dupliquées : **{df.duplicated().sum()}**")
        st.subheader("Transactions par catégorie")
        counts = df["category"].value_counts().rename_axis("category").reset_index(name="count")
        show_fig(px.bar(counts, x="category", y="count"))


def page_ml() -> None:
    page_header("Machine Learning", "Classification des ventes à forte valeur")
    st.markdown("### Objectif\nPrédire si une transaction est une "
                "**High Value Sale** ou une **Normal Sale**.")

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        md = dict(zip(metrics["metric"], metrics["value"]))
        kpi_row([
            ("Accuracy", f"{md.get('accuracy', 0):.2%}", "primary"),
            ("Precision", f"{md.get('precision', 0):.2%}", "info"),
            ("Recall", f"{md.get('recall', 0):.2%}", "success"),
            ("F1 Score", f"{md.get('f1', 0):.2%}", "warning"),
            ("ROC-AUC", f"{md.get('roc_auc', 0):.2%}", "danger"),
        ])
    else:
        st.warning(f"Fichier de métriques introuvable : {METRICS_PATH.name}")

    if CONFUSION_PATH.exists():
        st.subheader("Confusion Matrix")
        cm = pd.read_csv(CONFUSION_PATH, index_col=0)
        show_fig(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels={"x": "Prédit", "y": "Réel"}))

    st.subheader("Feature Importance")
    model = load_model()
    if model is None:
        st.warning("Modèle introuvable.")
        return
    importances = model.feature_importances_
    names = list(getattr(model, "feature_names_in_", FEATURE_NAMES))
    if len(names) != len(importances):  # garde-fou si le modèle a changé
        names = [f"feature_{i}" for i in range(len(importances))]
    imp = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance")
    show_fig(px.bar(imp, x="importance", y="feature", orientation="h"))


def page_prediction() -> None:
    page_header("Sales Prediction",
                "Prédire la probabilité qu'une vente soit une vente à forte valeur")

    model = load_model()
    if model is None:
        st.error("Le modèle Gradient Boosting est introuvable.")
        st.stop()

    st.subheader("Transaction Information")
    left, right = st.columns(2)
    with left:
        price = st.number_input("Product Price", min_value=0.0, value=100.0, step=10.0)
        category = st.selectbox("Category", sorted(df_all["category"].dropna().unique()))
        sale_date = st.date_input("Sale Date")
    with right:
        st.markdown("### Prediction\nLe modèle utilise les variables construites "
                    "pendant la phase de Machine Learning.")
        go_predict = st.button("Predict Sale", use_container_width=True)

    if not go_predict:
        return

    # Moyennes calculées sur TOUTES les données (pas sur les filtres de la sidebar).
    # Idéalement, sauvegarder ces moyennes au moment de l'entraînement.
    cat_mean = df_all.loc[df_all["category"] == category, "price"].mean()
    if not cat_mean or np.isnan(cat_mean):
        cat_mean = df_all["price"].mean()

    d = pd.Timestamp(sale_date)
    row = {
        "year": d.year, "month": d.month, "day": d.day,
        "day_of_week": d.dayofweek, "week_of_year": int(d.isocalendar().week),
        "is_weekend": int(d.dayofweek >= 5),
        "price_vs_category_mean": price / cat_mean,
        "log_price": np.log1p(price),
    }
    X = pd.DataFrame([row])
    # Respecte l'ordre exact des colonnes vu à l'entraînement
    X = X[list(getattr(model, "feature_names_in_", FEATURE_NAMES))]

    proba = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])

    st.markdown("---")
    (st.success if pred == 1 else st.info)(
        "High Value Sale" if pred == 1 else "Normal Sale")

    c1, c2 = st.columns(2)
    c1.metric("Normal Sale Probability", f"{proba[0]:.2%}")
    c2.metric("High Value Probability", f"{proba[1]:.2%}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba[1] * 100,
        number={"suffix": " %"},
        title={"text": "Probabilité de vente à forte valeur"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#198754"},
               "steps": [{"range": [0, 50], "color": "#e9ecef"},
                         {"range": [50, 100], "color": "#d1e7dd"}]},
    ))
    show_fig(gauge, 320)


# ============================================================
# ROUTAGE
# ============================================================

PAGES = {
    "Dashboard": page_dashboard,
    "Sales Analysis": page_sales,
    "Exploratory Analysis": page_eda,
    "Machine Learning": page_ml,
    "Prediction": page_prediction,
}
PAGES[page]()