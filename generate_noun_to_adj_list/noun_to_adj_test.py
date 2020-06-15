"""To test the code for noun to adjective generation"""
import pickle
from utils import sentence_list_gen, noun_list_gen, adjective_list_gen, filter_by_idf, save_as_csv
from noun_to_adj_gen import NounToAdjGen
import os

# Folder where generated files are stored
generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')
# Generating noun list and adjective list from wordnet
noun_list = noun_list_gen()
adj_list = adjective_list_gen()

# Loading dataset into sentences.
sentences = sentence_list_gen(generated_file + 'books_data.txt')

generator = NounToAdjGen(noun_list, adj_list) # an instance of class noun_to_adj_gen

# 10 = Number of perturbations you want to make for a word in a sentence
# Generating only for first 100 sentences for testing purposes.
generator.add_to_dic(sentences[:100], 10) 

# Get Adjectives IDF values.
with open(generated_file + 'adj_idf.dat', 'rb') as f:
    adj_idf = pickle.load(f)

# filtering by IDF (threshold = 8.125) , sorting by BERT score and returning top k=20 adjectives.
noun_to_adj = filter_by_idf(getattr(generator, 'noun_to_adj'), adj_idf, 8.125, 20)

# Asserting value
for noun, adjs in noun_to_adj.items():
	if adjs != []:
		assert noun == 'chamber'
		assert len(adjs) == 4
