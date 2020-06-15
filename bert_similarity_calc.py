from utils import adjective_list_gen
from bert import BertSimilarity
import pickle
adj_list = adjective_list_gen()

with open("noun_to_adj_score.dat", 'rb') as f:
    noun_to_adj = pickle.load(f)

selected_adj = []
adj_count = {}
for noun in noun_to_adj.keys():
    for adj, val in noun_to_adj[noun]:
        try:
            adj_count[adj] += -1
        except:
            adj_count[adj] = -1
sorted_adjs = sorted(adj_count.items(), key = lambda kv:(kv[1], kv[0]))[:1000]
for adj,val in sorted_adjs:
    selected_adj.append(adj)

adj_list = selected_adj

calculated_dis = {}
x = BertSimilarity()
for i in range(len(adj_list)):
    for j in range(len(adj_list)):
        if j<=i:
            continue
        w1 = adj_list[i]
        w2 = adj_list[j]    
        dis = 1-(x.get_sim([(w1, w2)], layer = -2)[0])
        calculated_dis[(w1, w2)] = dis
        calculated_dis[(w2, w1)] = dis  

with open("adj_distance.dat1", 'wb') as f:
    pickle.dump(calculated_dis, f)