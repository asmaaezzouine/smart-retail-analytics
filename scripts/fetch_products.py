import requests
url = "https://fakestoreapi.com/products"
response = requests.get(url)
products = response.json()
print(len(products))
print(products[0])
for product in products:
    print(product["title"])
    print(product["id"])
    print(product["price"])
    print(product["category"])