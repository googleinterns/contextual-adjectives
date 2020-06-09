"""Generate IDF for adjectives"""
import pickle
from nltk.corpus import wordnet
from nltk.text import TextCollection, Text

f = open("generated_files/all.txt", "r")
lines = f.readlines()
tokens = []

for line in lines:
    line = line.lower()
    line = line.split("\n")[0].split(" ")
    tokens.extend(line)

text_tokens = Text(tokens)
all_tokens = TextCollection(text_tokens)
adj_idf = {}

for adjective_set in list(wordnet.all_synsets(wordnet.ADJ)):
    adj = adjective_set.lemmas()[0].name().lower()
    adj_idf[adj] = all_tokens.idf(adj)

with open("generated_files/adj_idf.dat", 'wb') as f:
    pickle.dump(adj_idf, f)
