import pickle

with open("adj_distance.dat", 'rb') as f:
  distance_metric = pickle.load(f)

def cluster_cluster_dis(cluster1, cluster2):
  average_dis = 0
  for a in cluster1:
    for b in cluster2:
      average_dis += distance_metric[(a, b)]
  average_dis /= (len(cluster1)*len(cluster2))
  return average_dis

adj_list = []

for adj1, adj2 in distance_metric.keys():
  adj_list.append(adj1)
  adj_list.append(adj2)

adj_list = list(set(adj_list))[:100]
threshold = 0.15

clusters = []

for adj in adj_list:
  cluster = set([adj])
  if cluster not in clusters:
    clusters.append(cluster)
#decreasing = True
#i = 0
is_changing = True
while is_changing:
  is_changing = False
  for idx, cluster in enumerate(clusters):
    min_dis = 1
    cluster_idx = -1
    for idx1, cluster1 in enumerate(clusters):
      if cluster1 == cluster:
        continue
      dis = cluster_cluster_dis(cluster, cluster1)
      if dis < min_dis:
        min_dis = dis
        cluster_idx = idx1
    l1 = len(cluster)
    l2 = len(clusters[cluster_idx])
    if min_dis < min(0.05 + 0.005*(l1*l2 - l1 -l2 + 1), threshold):
      is_changing = True
      cluster1 = clusters[cluster_idx]
      new_cluster = cluster | cluster1
      clusters.remove(cluster)
      clusters.remove(cluster1)
      clusters.append(new_cluster)
      break
    if is_changing:
      break

f = open("hierarchial_clusters.csv", "w")

for cluster in clusters:
  s = ""
  for ele in cluster:
    s += ele + ","
  s = s[:-1]
  s += "\n"
  f.write(s)
f.close()

