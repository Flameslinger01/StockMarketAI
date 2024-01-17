import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

soup = BeautifulSoup(requests.get(URL).content, "html.parser")

print("[")
for tag in soup.select("td:nth-of-type(2)"):
    print("'" + tag.text + "', ", end="")
print("]")