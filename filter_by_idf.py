import pickle
with open("adj_idf.dat", 'rb') as f:
	adj_idf = pickle.load(f)


# threshold = 8
# count = 0
# for adj in adj_idf.keys():
# 	if adj_idf[adj] < threshold and adj_idf[adj]!=0:
# 		print(adj)
# 		count += 1



sorted_adjs = sorted(adj_idf.items(), key = lambda kv:(kv[1], kv[0]))[:500]
for adj,val in sorted_adjs:
	print(adj, val)