import re
import nltk
import requests
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

# Text collection
res = requests.get('https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt')
res.encoding = 'utf-8'
text = res.text

# Tokenization
tokens = re.findall("[\w]+", text)  #regex

# Lowercasing
lowercase = [x.lower() for x in tokens]

# Stemming_Porter’s algorithm
ps = PorterStemmer()
stemming = [ps.stem(i) for i in lowercase]

# Stopword removal
stops = stopwords.words('english')
filtered = [w for w in stemming if w not in stops]
print(filtered)

# Save the result as a txt file
txt = open("result.txt", "w")
for item in filtered:
    txt.write(item + "\n")
txt.close()