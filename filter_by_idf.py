import pickle
with open("adj_idf.dat", 'rb') as f:
	adj_idf = pickle.load(f)

with open("noun_to_adj_score.dat", "rb") as f:
	noun_to_adj = pickle.load(f)

for noun in noun_to_adj.keys():
	if noun_to_adj[noun]!=[]:
		print(noun)

for noun in noun_to_adj.keys():
	temp = {}
	total = {}
	for adj, score in noun_to_adj[noun]:
		try:
			temp[adj] += score
			total[adj] += 1
		except:
			temp[adj] = score
			total[adj] = 1
	final_score = {}
	for adj in temp.keys():
		final_score[adj] = -1*temp[adj]/total[adj]
	try:	
		sorted_adjs = sorted(final_score.items(), key = lambda kv:(kv[1], kv[0]))[:20]
	except:
		sorted_adjs = sorted(final_score.items(), key = lambda kv:(kv[1], kv[0]))
	sorted_lis = []
	for key, val in sorted_adjs:
		if adj_idf[key] >= 8.125:
			sorted_lis.append((key,-1*val))
	noun_to_adj[noun] = sorted_lis

f = open("noun_to_adj_sort.csv","w")
for noun in noun_to_adj.keys():
	if noun_to_adj[noun] !=[]:
		s = noun + ","
		adjs = noun_to_adj[noun]
		for adj, val in adjs:
			s += adj + "(" + str(round(val,2)) + "),"
		s += "\n"
		f.write(s)
f.close()      