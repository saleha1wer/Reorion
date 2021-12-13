import numpy as np
from csv import reader
from networkx import read_edgelist 
from networkx import parse_edgelist
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length as short_from_to_all
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import itertools


too = '/data/s3305139/olion/'
path = 'author.txt' 

# # Read graph
# lines = []
# #Path to csv edge list
# with open(path, 'r') as f:
#     csv_reader = reader(f)
#     header = next(csv_reader)
#     if header != None:
#         for row in csv_reader:
#             lines.append(str(row[0]) + ' ' + str(row[1]))

G = read_edgelist(path, nodetype=int)


# select set of 1000 random nodes
nodes = np.random.choice(G.nodes, size=1000)
possible_pairs = list(itertools.combinations(nodes, 2))
num_pairs = len(possible_pairs)
# compute ground-truth 500,000 pairwise distances using networkx 
pair_wise_dist = dict()
total = 0 
max_distance = 0 
for i in range(num_pairs):
    node1 = possible_pairs[i][0]
    node2 = possible_pairs[i][1]
    # while node1 == node2:
    #     node2 = np.random.choice(nodes)
    # while (str(node1) + ' '+ str(node2)) in pair_wise_dist.keys():
    #     node1 = np.random.choice(nodes)
    #     node2 = np.random.choice(nodes)
    #     while node1 == node2:
    #         node2 = np.random.choice(nodes)
    dist = shortest_path_length(G, source = node1, target=node2)
    total = total + dist
    if dist > max_distance:
        max_distance = dist
    pair_wise_dist[str(node1) + ' '+ str(node2)] = dist
print('Real Average: ', total/num_pairs)
print('Real Diameteer: ', max_distance)

strategies = ['random', 'degree', 'max_ave_distance', 'max_min_distance', 'convex_hull']

for strategy in strategies:
    new_dict = np.load('/Users/mrsalwer/Desktop/Uni/Leiden Uni/Year-1/SNACS/proj/cords/dogster/dogster.txt_'+strategy+'.npy', allow_pickle='TRUE')
    # slice dict keys by space
    # for every pair you slice, calc euclideaan distance and calculate relative error 
    # average relative error
    total = 0
    total_rel_error = 0
    max_distance = 0 
    count = 0
    for i in pair_wise_dist.keys():
        nodes_to_estimate = i.split()
        node1 = int(nodes_to_estimate[0])
        node2 = int(nodes_to_estimate[1])
        cords_node1 = new_dict.item().get(node1)
        cords_node2 = new_dict.item().get(node2)

        if cords_node1 is None or cords_node2 is None:
            count = count + 1
            continue

        if node1 == node2:
            count = count + 1
            continue
        olion_prediction = np.linalg.norm(cords_node1 - cords_node2)
        if olion_prediction > max_distance:
            max_distance = olion_prediction
        total = total + olion_prediction
        real_distance = pair_wise_dist[i]
        relative_error = (abs(real_distance - olion_prediction)) / real_distance

        total_rel_error = total_rel_error + relative_error


    average_relative_error = total_rel_error / (len(pair_wise_dist.keys()) - count)
    print(strategy, 'ARE: ', average_relative_error)
    print(count)
    print(strategy, 'Average Path: ', total/(num_pairs-count))
    print(strategy, 'Diameter: ', max_distance)
