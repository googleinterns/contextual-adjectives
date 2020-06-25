"""Clustering the adjectives in a hierarchical manner"""
import os
import pickle

def compute_cluster_distance(cluster_1, cluster_2):
    """Calculate distance between clusters

    It is defined as average over distance between adj1, adj2
    pairs where adj1 belongs to cluster_1 and adj2 belongs to cluster_2

    returns distance between the two clusters
    """
    average_distance = 0
    for adj_1 in cluster_1:
        for adj_2 in cluster_2:
            average_distance += distance_metric[(adj_1, adj_2)]
    average_distance /= (len(cluster_1)*len(cluster_2))
    return average_distance

def optimize_clusters(clusters, min_threshold, max_threshold):
    """Optimize the number of clusters by combining clusters that are at minimum distance
    
    The method used for threshold here is min(min_threshold + 0.005*(len1*len2 - len1 -len2 + 1)
    len1 is length of first cluster
    len2 is length of second cluster


    clsuters: initial clusters list
    min_threshold : minimum threshold for combining of clusters
    max_threshold: maximum threshold for combining of clusters

    returns the list of minimized clusters
    """
    is_changing = True
    while is_changing:
        is_changing = False
        for _, cluster1 in enumerate(clusters):
            min_distance = 1
            cluster_idx = -1
            for idx1, cluster2 in enumerate(clusters):
                if cluster2 == cluster1:
                    continue
                distance = compute_cluster_distance(cluster1, cluster2)
                if distance < min_distance:
                    min_distance = distance
                    cluster_idx = idx1
            len1 = len(cluster1)
            len2 = len(clusters[cluster_idx])
            # We can also vary the condition for clustering here
            if min_distance < min(min_threshold + 0.005*(len1*len2 - len1 -len2 + 1),
                                  max_threshold):
                is_changing = True
                cluster2 = clusters[cluster_idx]
                new_cluster = cluster1 | cluster2
                clusters.remove(cluster1)
                clusters.remove(cluster2)
                clusters.append(new_cluster)
                break
            if is_changing:
                break
    return clusters

def save_adj_clusters_csv(clusters, file_name):
    """Saves the clusters in a csv file"""
    file = open(file_name, "w")
    for cluster in clusters:
        sentence = ""
        for adjective in cluster:
            sentence += adjective + ","
        sentence = sentence[:-1]
        sentence += "\n"
        file.write(sentence)
    file.close()

if __name__ == "__main__":
    # Folder where generated files are stored
    generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

    with open(generated_file + "adj_distance.dat", 'rb') as f:
        distance_metric = pickle.load(f)

    adj_list = []
    for adj1, adj2 in distance_metric.keys():
        adj_list.append(adj1)
        adj_list.append(adj2)
    adj_list = list(set(adj_list))

    # These values were chosen after trying clustering with several thresholds
    # However, I don't think they are optimal and hence there must exists better
    # values than these
    MIN_THRESHOLD = 0.05
    MAX_THRESHOLD = 0.15

    initial_clusters = []
    for adj in adj_list:
        single_adj_cluster = set([adj])
        if single_adj_cluster not in initial_clusters:
            initial_clusters.append(single_adj_cluster)

    final_clusters = minimize_clusters(initial_clusters, MIN_THRESHOLD, MAX_THRESHOLD)
    save_adj_clusters_csv(final_clusters, generated_file + "hierarchial_clusters.csv")
