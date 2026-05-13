import random
from datetime import datetime, timedelta
from db_connection import connect_db

# connexion PostgreSQL
conn, cur = connect_db()

# récupérer produits
cur.execute("SELECT product_id, price FROM products;")
products = cur.fetchall()

# fonction pour générer une date aléatoire
def random_date(start_days_ago=30):
    start_date = datetime.now() - timedelta(days=start_days_ago)
    random_days = random.randint(0, start_days_ago)
    return start_date + timedelta(days=random_days)

# requête insert
query = """
INSERT INTO sales (
    product_id,
    quantity,
    sale_date,
    total_price
)
VALUES (%s, %s, %s, %s);
"""

# génération des ventes
for product in products:
    product_id = product[0]
    price = product[1]

    # nombre de ventes par produit
    for _ in range(random.randint(10, 30)):

        quantity = random.randint(1, 5)
        sale_date = random_date()
        total_price = price * quantity

        values = (
            product_id,
            quantity,
            sale_date,
            total_price
        )

        cur.execute(query, values)

# sauvegarde
conn.commit()

# fermeture
cur.close()
conn.close()

print("Sales generated successfully")