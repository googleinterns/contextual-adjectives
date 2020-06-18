"""Categorize Adjectives using K-medoids algorithm
Here we only try to categorize most frequent 100 adjectives
"""
import os
import random
import pickle
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from libraries.bert import BertSimilarity

# Folder where generated files are stored
generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

# Initializing the BERT Similarity function and also storing calculated distances in array.
BERT = BertSimilarity()
dis_array = np.zeros((150, 150)) - 1

def distance(w_1, w_2):
    """
    Calulates and Returns the distance between w1 and w2
    Also stores the distance in dis_array
    """
    if dis_array[w_1, w_2] == -1:
        dis_array[w_1, w_2] = 1-(BERT.get_similarity([(selected_adj[int(w_1)], selected_adj[int(w_2)])],
                                 layer=-2)[0])
        dis_array[w_2, w_1] = dis_array[w_1, w_2]
    return dis_array[w_1, w_2]

# Fetching the noun to adj dictionary from pickle file
with open(generated_file + 'noun_to_adj_score.dat', 'wb') as f:
    noun_to_adj = pickle.load(f)

# Selecting the most frequent 100 adjectives
NUM_ADJ = 100
selected_adj = []
adj_count = {}
for noun in noun_to_adj.keys():
    for adj, val in noun_to_adj[noun]:
        try:
            adj_count[adj] += -1
        except KeyError:
            adj_count[adj] = -1
sorted_adjs = sorted(adj_count.items(), key=lambda kv: (kv[1], kv[0]))[:NUM_ADJ]
for adj, val in sorted_adjs:
    selected_adj.append(adj)

# Choosing number of Clusters
NUM_CLUSTER = 10
sample = [np.array(f) for f in range(NUM_ADJ)]
initial_medoids = random.sample(range(NUM_ADJ), NUM_CLUSTER)
metric = distance_metric(type_metric.USER_DEFINED, func=distance)
kmedoids_instance = kmedoids(sample, initial_medoids, metric=metric)
# run cluster analysis and obtain results
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

# Save the clusters in a csv file
f = open(generated_file + "kmedoids_clusters.csv", "w")
for cluster in clusters:
    sentence = ""
    for idx in cluster:
        sentence += selected_adj[idx] + ","
    sentence = sentence[:-1]
    sentence += "\n"
    f.write(sentence)
f.close()
