"""Generate csv file with nouns as first coloumn and other columns correspond to adjectives"""

import pickle
from utils import save_as_csv, sentiment_calculator
import os

# Folder where generated files are stored
generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

# Fetching the noun to adj dictionary from pickle file
with open(generated_file + 'noun_to_adj_score.dat', 'wb') as f:
    noun_to_adj = pickle.load(f)

# Generating polarity of adjectives for a noun
noun_to_adj_polarity = sentiment_calculator(noun_to_adj)

save_as_csv(generated_file + 'noun_to_adj_polarity.csv', noun_to_adj_polarity)
