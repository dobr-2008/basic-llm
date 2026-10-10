import nltk 
import os 
os.getcwd()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('omw-1.4')

text_file = open("output.txt" , encoding= "utf-8")

tokens = word_tokenize(text_file.read())

tokens = [word.lower() for word in tokens if word.isalnum]
filtered_tokens = [word for word in tokens if word not in stopwords.words("english")]


print(filtered_tokens)