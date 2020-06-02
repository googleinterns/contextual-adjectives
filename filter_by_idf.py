from nltk.text import TextCollection, Text
from nltk.corpus import wordnet
f = open("all.txt", "r")
lines = f.readlines()
tokens = []
for line in lines:
	line = line.lower()
	line = line.split("\n")[0].split(" ")
	tokens.extend(line)
text = Text(tokens)
mytexts = TextCollection(text)
# Print the IDF of a word
adj_idf = {}
for synset in list(wordnet.all_synsets(wordnet.ADJ)):
	adj = synset.lemmas()[0].name().lower()
	adj_idf[adj] = mytexts.idf(adj)

with open("adj_idf.dat", 'wb') as f:
	pickle.dump(adj_idf, f)