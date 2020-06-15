import pickle
from utils import sentence_list_gen, noun_list_gen, adjective_list_gen, filter_by_idf, save_as_csv, sentiment_calculator

with open("generated_files/noun_to_adj_score.dat", 'rb') as f:
    noun_to_adj = pickle.load(f)

# Get Adjectives IDF values.
with open("generated_files/adj_idf.dat", 'rb') as f:
    adj_idf = pickle.load(f)

# Filtering by IDF (threshold = 8.125) , sorting by BERT score and returning top k=20 adjectives.
noun_to_adj = filter_by_idf(noun_to_adj, adj_idf, 8.125, 20)

# Generating polarity of adjectives for a noun
noun_to_adj_polarity = sentiment_calculator(noun_to_adj)

for noun, adjs in noun_to_adj_polarity.items():
  dic = {'positive' : [], 'negative' : [], 'neutral' : []}
  for adj, val in adjs:
    if val > 0:
      dic['positive'].append(adj)
    if val < 0:
      dic['negative'].append(adj)
    if val == 0:
      dic['neutral'].append(adj)
  noun_to_adj_polarity[noun] = dic

with open("generated_files/noun_to_adj_polarity.dat", 'wb') as f:
    pickle.dump(noun_to_adj_polarity, f)    