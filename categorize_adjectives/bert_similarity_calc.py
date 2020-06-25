"""Generate dictionary of distances between each pair of adjective"""
import os
import pickle
from libraries.bert_embeddings import BertEmbedding
from scipy.spatial.distance import cosine

def get_most_frequent_adjs(noun_to_adj, num_adjs):
    """Generate the list of most frequent num_adjs in noun_to_adj dictionary
    
    noun_to_adj: dictionary of noun to adjective
    num_adjs: number of most frequent adjectives need to be returned

    returns the list of most frequent adjectives
    """
    selected_adj = []
    adj_count = {}
    for noun in noun_to_adj.keys():
        for adj, _ in noun_to_adj[noun]:
            # Used -1 to sort in descending order
            try:
                adj_count[adj] += -1
            except KeyError:
                adj_count[adj] = -1
    sorted_adjs = sorted(adj_count.items(), key=lambda kv: (kv[1], kv[0]))[:num_adjs]
    for adj, _ in sorted_adjs:
        selected_adj.append(adj)
    return selected_adj

def compute_adj_distance_matrix(adj_list):
    """generate the distance metric between each pair of adjective from adj_list

    First it calls the get_embeddings library to get embeddings of each adjective
    and then takes cosine distance between each pair of adjective from adj_list

    adj_list: list of adjective for which distance metric is being generated

    returns a dicitonary that contains tuple of two words and point to their distance
    """
    calculated_dis = {}
    bert_embeddings_gen = BertEmbedding()
    tuple_adj = []
    for adj in adj_list:
        tuple_adj.append((adj,))
    # Used -2 layer as it contains contexual embeddings in BERT
    embeddings = bert_embeddings_gen.get_embeddings(tuple_adj, layer=-2)
    for i, adj_1 in enumerate(adj_list):
        for j, adj_2 in enumerate(adj_list, i+1):
            distance = cosine(embeddings[i], embeddings[j])
            calculated_dis[(adj_1, adj_2)] = distance
            calculated_dis[(adj_2, adj_1)] = distance
    return calculated_dis


if __name__ == "__main__":
    # Folder where generated files are stored
    generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

    with open(generated_file + 'noun_to_adj_score.dat', 'rb') as f:
        noun_to_adj_dictionary = pickle.load(f)

    NUM_ADJS = 15000

    adjective_list = get_most_frequent_adjs(noun_to_adj_dictionary, NUM_ADJS)

    distance_metric = compute_adj_distance_matrix(adjective_list)

    with open(generated_file + "adj_distance.dat", 'wb') as f:
        pickle.dump(distance_metric, f)
