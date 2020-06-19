import pickle

with open('adj_distance.dat1', 'rb') as f:
	distance_metric = pickle.load(f)

adj_list = []

for adj1, adj2 in distance_metric.keys():
	adj_list.append(adj1)
	adj_list.append(adj2)

adj_list = list(set(adj_list))

threshold = 0.09

clusters = {}
clusters[1] = set()

for adj in adj_list:
	cluster = set(adj)
	clusters.insert(cluster)

