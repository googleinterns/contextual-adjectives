"""Generate csv file with nouns as first coloumn and other columns correspond to adjectives"""

import pickle
from utils import sentence_list_gen, noun_list_gen, adjective_list_gen, filter_by_idf, save_as_csv, sentiment_calculator
from noun_to_adj_gen import NounToAdjGen

# Generating noun list and adjective list from wordnet
noun_list = noun_list_gen()
adj_list = adjective_list_gen()

sentences = sentence_list_gen("generated_files/books_data.txt") # Loading dataset into sentences.

generator = NounToAdjGen(noun_list, adj_list) # an instance of class noun_to_adj_gen

# 10 = Number of perturbations you want to make for a word in a sentence
generator.add_to_dic(sentences, 10) 

#Saving the dictionary to a pickle file
with open("generated_files/noun_to_adj_score.dat", 'wb') as f:
    pickle.dump(getattr(generator, 'noun_to_adj'), f)

# Get Adjectives IDF values.
with open("generated_files/adj_idf.dat", 'rb') as f:
    adj_idf = pickle.load(f)

# Filtering by IDF (threshold = 8.125) , sorting by BERT score and returning top k=20 adjectives.
noun_to_adj = filter_by_idf(getattr(generator, 'noun_to_adj'), adj_idf, 8.125, 20)

save_as_csv("generated_files/noun_to_adj_sort.csv", noun_to_adj)

# Generating polarity of adjectives for a noun
noun_to_adj_polarity = sentiment_calculator(noun_to_adj)

save_as_csv("generated_files/noun_to_adj_polarity.csv", noun_to_adj_polarity)