from utils import adjective_list_gen
from bert import BertSimilarity
import pickle
from scipy.spatial.distance import cosine
adj_list = adjective_list_gen()

calculated_dis = {}
x = BertSimilarity()
tuple_adj = []
for adj in adj_list:
    tuple_adj.append((adj,))
embeddings = x.get_sim(tuple_adj, layer = -2)
for i in range(len(adj_list)):
    for j in range(len(adj_list)):
        if j<i:
            continue
        w1 = adj_list[i]
        w2 = adj_list[j]    
        dis = cosine(embeddings[i], embeddings[j])
        calculated_dis[(w1, w2)] = dis
        calculated_dis[(w2, w1)] = dis  

with open("adj_distance.dat", 'wb') as f:
    pickle.dump(calculated_dis, f)