import numpy as np
import multiprocessing as mp
import time
from csv import reader
import itertools
import time
import sys
import heapq
import os


from networkx import read_edgelist 
from networkx import parse_edgelist
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length as short_from_to_all

from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from functools import partial




def select_lm(Graph,k_value, strategy):
    """ Select k landmarks from graph G using either strategy = 'random' or 'degree' 
        Returns landmarks and their shortest path lengths to all other nodes"""
    nodes_list = list(Graph.nodes)
    landmarks = []
    landmark_shortest_paths = []
    
    if strategy == 'random':
        for i in range(k_value):
            p_landmark = np.random.choice(nodes_list) #Pick candidate

            if len(landmarks) > 1:
                # Make sure no other landmark is within 3-hop
                while is_bad(landmark_shortest_paths, p_landmark): 
                    p_landmark = np.random.choice(nodes_list)

            # Add landmark
            landmarks.append(p_landmark)
            # Compute landmark shortest paths
            landmark_shortest_paths.append(short_from_to_all(Graph,landmarks[i]))
            nodes_list.remove(p_landmark)

    elif strategy == 'degree':
        degrees =  Graph.degree()
        degrees = dict(degrees)

        for i in range(k_value):
            # pick first landmark as the node with highest degree
            p_landmark = heapq.nlargest(1, degrees, key=degrees.get)[0] 
            # remove from list of possible landmarks
            degrees.pop(p_landmark) 
            if len(landmarks) > 1: 
                while is_bad(landmark_shortest_paths, p_landmark): # Make sure no other landmark is within 3-hop
                    p_landmark = heapq.nlargest(1, degrees, key=degrees.get)[0]
                    degrees.pop(p_landmark)

            landmarks.append(p_landmark) # Add landmark 

            landmark_shortest_paths.append(short_from_to_all(Graph,landmarks[i])) # Compute landmark shortest paths

    elif strategy == 'max_min_distance':
        degrees =  Graph.degree()
        degrees = dict(degrees)

        highest_degrees = heapq.nlargest(300, degrees, key=degrees.get)
        first_landmark = np.random.choice(highest_degrees)
        second_landmark = np.random.choice(highest_degrees)
        landmarks.append(first_landmark)
        landmarks.append(second_landmark)
        landmark_shortest_paths.append(short_from_to_all(Graph,first_landmark))
        landmark_shortest_paths.append(short_from_to_all(Graph,second_landmark))
        for i in range(k_value-2):
            node_min_d_to_landmark = dict()

            #look at all possible landmarks
            for node in nodes_list:
                node_distances = []
                #calculate distance of node to all landmarks and store the length of the minimum one
                for dist_to_landmark in landmark_shortest_paths:
                    node_distances.append(dist_to_landmark[node])
                node_min_d_to_landmark[node] = min(node_distances)

            #pick node with maximum minimum distance as the new landmarks
            new_landmark = max(node_min_d_to_landmark, key=node_min_d_to_landmark.get)
            landmarks.append(new_landmark)
            landmark_shortest_paths.append(short_from_to_all(Graph,new_landmark))

    elif strategy == 'max_ave_distance':
        degrees =  Graph.degree()
        degrees = dict(degrees)
        highest_degrees = heapq.nlargest(300, degrees, key=degrees.get)

        first_landmark = np.random.choice(highest_degrees)
        second_landmark = np.random.choice(highest_degrees)
        landmarks.append(first_landmark)
        landmarks.append(second_landmark)
        landmark_shortest_paths.append(short_from_to_all(Graph,first_landmark))
        landmark_shortest_paths.append(short_from_to_all(Graph,second_landmark))
        for i in range(k_value-2):
            node_ave_d_to_landmark = dict()
            #look at all possible landmarks
            for node in nodes_list:
                node_distances = []
                #calculate distance of node to all landmarks and store the length of the minimum one
                for dist_to_landmark in landmark_shortest_paths:
                    node_distances.append(dist_to_landmark[node])
                node_ave_d_to_landmark[node] = (sum(node_distances)/len(node_distances))
            #pick node with maximum minimum distance as the new landmarks
            new_landmark = max(node_ave_d_to_landmark, key=node_ave_d_to_landmark.get)
            landmarks.append(new_landmark)
            landmark_shortest_paths.append(short_from_to_all(Graph,new_landmark))
    elif strategy == 'convex_hull':
        temp_k = 10
        temp_k_initial = 5
        temp_d = 2
        #Select landmarks 
        temp_k_landmarks, temp_k_landmarks_shortest_paths= select_lm(Graph,temp_k,strategy='random')
        #Calculate coordinates for initial group (k) of landmarks
        temp_initial_landmarks_cords = plot_initial(temp_k_landmarks[:temp_k_initial], temp_k_landmarks_shortest_paths[:temp_k_initial],temp_d)
        #Calculate coordinates for secondary group (k - ki) of landmarks
        temp_secondary_landmarks_cords =  plot(temp_initial_landmarks_cords, temp_k_landmarks_shortest_paths, temp_k_landmarks[temp_k_initial:],temp_d)
        # Merge dictionaries together
        temp_landmark_cords = {**temp_initial_landmarks_cords, **temp_secondary_landmarks_cords}
        #Calculate coordinates for rest of the nodes in the graph
        temp_rest_of_graph = list(set(Graph.nodes) - set(temp_k_landmarks))
        temp_restOF_nodes_cords = plot(temp_landmark_cords, temp_k_landmarks_shortest_paths, temp_rest_of_graph, temp_d)
        #plotting to 2D using random landmark selection
        temp_all_cords = {**temp_landmark_cords, **temp_restOF_nodes_cords}
    
        list_of_cords = list(temp_all_cords.values())
        hull = ConvexHull(list_of_cords)
        for idx, vertex in enumerate(hull.vertices):
            cord_of_vertex = list_of_cords[vertex]
            landmarks.append(int(get_key(temp_all_cords, cord_of_vertex)))
            landmark_shortest_paths.append(short_from_to_all(Graph,landmarks[idx]))

        # If we need more landmarks
        print(len(landmarks), 'landmarks selected using Convex Hull')
        while len(landmarks) < k_value:
            new_landmark = np.random.choice(nodes_list)
            if new_landmark not in landmarks:
                landmarks.append(new_landmark)
                landmark_shortest_paths.append(short_from_to_all(Graph,new_landmark))

        # If we have too many landmarks
        if len(landmarks)  > k_value:
            landmarks = landmarks[:k_value]
            landmark_shortest_paths = landmark_shortest_paths[:k_value]

    
    return landmarks, landmark_shortest_paths


def get_key(dic, val):
    # https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    for key, value in dic.items():
        if np.array_equal(val, value):
            return key
 
    return "key doesn't exist"

def is_bad(lm_shortestpaths, candidate):
    """ Returns true if a landmark-candidate is within 3-hop of another landmark"""
    bad = False
    for z in lm_shortestpaths:
            
        if z[candidate] < 3: 
            bad = True
            break
    return bad

def chunks(lst, n):
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def plot_initial(landmarks, landmark_shortest_paths, dimension):
    """ Plots landmarks in euclidean space, minimmizing the difference between their pairwise shortest path lengths and the euclidean distance """
    guess = np.random.uniform(low=-15,high=15,size=(dimension*len(landmarks)))
    temp = minimize(sq_error_initial, guess, args=(landmarks, landmark_shortest_paths, dimension), method='Nelder-Mead', options={ 'adaptive' : True, 'return_all' : False, 'maxiter' : 1000*d*len(landmarks) })
    cords= temp['x']
    cords_split = list(chunks(cords,dimension))

    #Make dictionary of landmarks' coordinates so that dict(landmark) return coordinates of landmark
    cords_dict = dict()

    for idx, cord in enumerate(cords_split):
        cords_dict[str(landmarks[idx])] = cord
    return cords_dict

def sq_error_initial(cords, landmarks, landmark_shortest_paths, dimension):
    """ Returns the sum of the squared error between euclidean distance between the landmark coordinates (cords) and ther shortest paths' lengths for all possible pairs of landmarks"""

    total_error = 0
    cords_by_idx = list(chunks(cords,dimension))

    possible_pairs = list(itertools.combinations(range(len(landmarks)), 2))

    for i in possible_pairs:
        cord1 = cords_by_idx[i[0]]
        cord2 = cords_by_idx[i[1]]
    
        e_distance = np.linalg.norm( cord1 - cord2)
        error = (e_distance - landmark_shortest_paths[i[0]][i[1]])**2
        total_error = total_error + error 

    return total_error

def plot(anchors_cords, anchors_shortest_paths, nodes_to_plot,dimension):
    """ Plots nodes to already-plotted anchors, minimmizing the difference between their shortest path lengths and the euclidean distance"""

    #Make dictionary of landmarks' coordinates so that dict(landmark) return coordinates of landmark
    
    n_cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = mp.Pool(processes=int(n_cores))

    plot_one_node_fixed = partial(plot_one_node, anchors_cords= anchors_cords, anchors_shortest_paths= anchors_shortest_paths, dimension=dimension)
    results = pool.map(plot_one_node_fixed, nodes_to_plot)


    pool.close()
    pool.join()
    cords_dict = dict(results)

    # cords_dict = dict()
    # for node in nodes_to_plot:
    #     guess = np.random.uniform(low=-15,high=15,size=dimension)
    #     temp = minimize(sq_error, guess, args=(anchors_cords,anchors_shortest_paths, node), method='Nelder-Mead', options={ 'adaptive' : True, 'maxiter' : 700*dimension })
    #     cord = temp['x']
    #     cords_dict[str(node)] = cord


    return cords_dict

def plot_one_node(node, anchors_cords,anchors_shortest_paths,dimension):
    guess = np.random.uniform(low=-15,high=15,size=dimension)
    temp = minimize(sq_error, guess, args=(anchors_cords,anchors_shortest_paths, node), method='Nelder-Mead', options={ 'adaptive' : True, 'maxiter' : 500*dimension })
    cord = temp['x']
    return [node,cord]

def sq_error(cord,anchors_cords, anchors_shortest_paths, node):
    """ Returns the squared error between the euclidean distance of node coordindates (cord) and their shortest paths to the achors"""
    total_error = 0
    for idx,anchor in enumerate(anchors_cords):
        anchor_shortest_paths = anchors_shortest_paths[idx]
        anchor_cord = anchors_cords[anchor] 
        e_distance = np.linalg.norm(cord - anchor_cord) 
        error = (e_distance - anchor_shortest_paths[node])**2
        total_error = total_error + error
    return total_error

if __name__ == '__main__':
    strategies = ['random', 'degree', 'max_ave_distance', 'max_min_distance', 'convex_hull']
    too = '/data/s3305139/olion/'
    path = 'petster.txt' 


    for strategy in strategies:
        # # Read graph
        # lines = []
        # #Path to csv edge list
        # with open(path, 'r') as f:
        #     csv_reader = reader(f)
        #     header = next(csv_reader)
        #     if header != None:
        #         for row in csv_reader:
        #             lines.append(str(row[0]) + ' ' + str(row[1]))
        start = time.time()
        G = read_edgelist(path, nodetype=int)
        now = time.time()
        print('Graph read as: ', G, 'Time Taken:', int(now - start), 'second(s)')
        # dimension
        d = 5

        k = 30
        k_initial = 16

        #Select landmarks 
        k_landmarks, k_landmarks_shortest_paths= select_lm(G,k,strategy=strategy)
        now1 = time.time()


        print(len(k_landmarks), ' Landmarks Selected', 'Time Taken:', int(now1 - now), 'second(s)') 
        print(k_landmarks)


        #Calculate coordinates for initial group (k) of landmarks
        initial_landmarks_cords = plot_initial(k_landmarks[:k_initial], k_landmarks_shortest_paths[:k_initial],d)
        now2 = time.time()


        print('Initial', k_initial, 'Landmarks Plotted', 'Time Taken:', int(now2 - now1), 'second(s)')


        #Calculate coordinates for secondary group (k - ki) of landmarks
        secondary_landmarks_cords =  plot(initial_landmarks_cords, k_landmarks_shortest_paths, k_landmarks[k_initial:],d)
        now3 = time.time()


        print('All', k, 'Landmarks Plotted', 'Time Taken:', int(now3 - now2), 'second(s)')


        # Merge dictionaries together
        landmark_cords = {**initial_landmarks_cords, **secondary_landmarks_cords}

        #Calculate coordinates for rest of the nodes in the graph

        rest_of_graph = list(set(G.nodes) - set(k_landmarks))
        restOF_nodes_cords = plot(landmark_cords, k_landmarks_shortest_paths, rest_of_graph, d)

        all_cords = {**landmark_cords, **restOF_nodes_cords}
        now4 = time.time()

        print('All',len(G.nodes),'nodes plotted', 'Total Time Taken For',strategy, ':', int(now4 - start), 'second(s)')

        np.save('/data/s3305139/olion/'+path+'_'+ strategy +'.npy', all_cords)










