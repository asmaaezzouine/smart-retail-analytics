from db_connection import connect_db

conn, cur = connect_db()

print("Connected successfully")

cur.execute("SELECT version();")
print(cur.fetchone())

cur.close()
conn.close()