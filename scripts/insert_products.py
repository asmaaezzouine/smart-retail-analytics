import requests
from db_connection import connect_db
conn, cur = connect_db()
url = "https://fakestoreapi.com/products"
response = requests.get(url)
products = response.json()
query = """
INSERT INTO products (
    product_id,
    name,
    category,
    price
)
VALUES (%s, %s, %s, %s)
ON CONFLICT (product_id) DO NOTHING;
"""
for product in products :

    values = (
    product["id"],
    product["title"],
    product["category"],
    product["price"]
    )
    cur.execute(query,values)
conn.commit()
cur.close()
conn.close()
