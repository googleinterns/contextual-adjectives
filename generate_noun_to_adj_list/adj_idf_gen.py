"""Generate IDF for adjectives"""
import pickle
from nltk.corpus import wordnet
from nltk.text import TextCollection, Text
import os

# Folder where generated files are stored
generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

f = open(generated_file + 'books_data.txt', "r")
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

with open(generated_file + 'adj_idf.dat', 'wb') as f:
    pickle.dump(adj_idf, f)
