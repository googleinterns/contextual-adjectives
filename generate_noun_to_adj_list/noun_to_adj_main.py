"""Generate csv file with nouns as first coloumn and other columns correspond to adjectives"""

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
generator.add_to_dictionary(sentences, 10) 

# Saving the dictionary to a pickle file
with open(generated_file + 'noun_to_adj_score.dat', 'wb') as f:
    pickle.dump(getattr(generator, 'noun_to_adj'), f)

# Get Adjectives IDF values.
with open(generated_file + 'adj_idf.dat', 'rb') as f:
    adj_idf = pickle.load(f)

# Filtering by IDF (threshold = 8.125) , sorting by BERT score and returning top k=20 adjectives.
noun_to_adj = filter_by_idf(getattr(generator, 'noun_to_adj'), adj_idf, 8.125, 20)

save_as_csv(generated_file + 'noun_to_adj_sort.csv', noun_to_adj)
