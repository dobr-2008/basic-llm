import requests 
from bs4 import BeautifulSoup
debug_mode = True 
url = "https://en.wikipedia.org/wiki/Earth"


headers = { 
    "User-Agent": "MyLearningScraper/1.0 (https://github.com/my-username; my-email@example.com)" ,
    "User-Agent1": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
#debug stuff
if debug_mode == True:
    print(f"HTTP response: {response.status_code}")

soup = BeautifulSoup(response.text , "html.parser")

content_div = soup.find("div" , class_ = "mw-parser-output")
pgraph = soup.find_all("p")

if debug_mode == True:
    print(f" amount of paragraphs:{len(pgraph)}")

text_data =  soup.get_text()

with open("output.txt" ,  "w" , encoding= "utf-8") as file:
    file.write(str(soup.get_text()))

print(text_data)
