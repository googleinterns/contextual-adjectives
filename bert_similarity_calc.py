from utils import adjective_list_gen
from bert import BertSimilarity
import pickle
adj_list = adjective_list_gen()
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

with open("adj_distance.dat", 'wb') as f:
    pickle.dump(calculated_dis, f)