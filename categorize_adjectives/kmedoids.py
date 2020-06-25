"""Categorize Adjectives using K-medoids algorithm
Here we only try to categorize most frequent 150 adjectives
"""
import os
import random
import pickle
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

def distance(w_1, w_2):
    """
    Calulates and Returns the BERT contexual distance between w1 and w2
    Also stores the distance in dis_array

    w_1, w_2: index of adjectives in the list of adjectives between which we need to find distance.

    returns distance between two words
    """
    return dis_array[w_1, w_2]

def get_frequent_adjectives(noun_to_adj, num_adj):
    """Selecting the most frequent num_adj adjectives from the noun_to_adj dictionary"""
    selected_adjs = []
    adj_count = {}
    for noun in noun_to_adj.keys():
        for adj, _ in noun_to_adj[noun]:
            try:
                adj_count[adj] += -1
            except KeyError:
                adj_count[adj] = -1
    sorted_adjs = sorted(adj_count.items(), key=lambda kv: (kv[1], kv[0]))[:num_adj]
    for adj, _ in sorted_adjs:
        selected_adjs.append(adj)
    return selected_adjs

def kmedoids_clustering(num_adj, num_cluster):
    """Performs kmedoids clustering

    Uses pyclustering library.

    num_adj: Number of adjectives being clustered.
    num_cluster: Number of clusters they need to be clustered to.

    returns clusters list
    """
    sample = [np.array(f) for f in range(num_adj)]
    initial_medoids = random.sample(range(num_adj), num_cluster)
    metric = distance_metric(type_metric.USER_DEFINED, func=distance)
    kmedoids_instance = kmedoids(sample, initial_medoids, metric=metric)
    # run cluster analysis and obtain results
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    return clusters

def save_to_csv(clusters, file_name):
    """Save the clusters in a csv file"""
    file = open(file_name, "w")
    for cluster in clusters:
        sentence = ""
        for idx in cluster:
            sentence += selected_adj[idx] + ","
        sentence = sentence[:-1]
        sentence += "\n"
        file.write(sentence)
    file.close()

if __name__ == "__main__":
    NUM_ADJ = 150
    NUM_CLUSTER = 10

    # Folder where generated files are stored
    generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

    # Initializing the BERT Similarity function and also storing calculated distances in array.
    with open(generated_file + "adj_distance.dat", 'rb') as f:
        dis_array = pickle.load(f)

    # Fetching the noun to adj dictionary from pickle file
    with open(generated_file + 'noun_to_adj_score.dat', 'rb') as f:
        noun_to_adj_dictionary = pickle.load(f)
    selected_adj = get_frequent_adjectives(noun_to_adj_dictionary, NUM_ADJ)
    generated_clusters = kmedoids_clustering(NUM_ADJ, NUM_CLUSTER)
    save_to_csv(generated_clusters, generated_file + "kmedoids_clusters.csv")
