import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import LFR_benchmark_graph
from community import community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import asyn_lpa_communities
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import os
np.random.seed(50)

# Generate a graph for girvan newman benchmark
def benchmark_girvan_newman():
    N = 128
    tau1 = 3
    tau2 = 1.5
    mu = 0.04
    k =16
    minc = 32
    maxc = 32
    return LFR_benchmark_graph(n = N, tau1 = tau1, tau2 = tau2, mu = mu, min_degree = k,
                            max_degree = k, min_community=minc, max_community = maxc, seed = 10)

# Louvain's community detection method
def detect_communities_louvain(G):
    partition = community_louvain.best_partition(G)
    communities = list()
    for com in set(partition.values()) :
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        communities.append(sorted(list_nodes))
    return sorted(communities)

# Girvan Newman's community detection method
def detect_communities_girvan_newman(G):
    communities = community.girvan_newman(G)
    return sorted(sorted(c) for c in next(communities))

# Fast Greedy community detection method
def detect_communities_greedy(G):
    communities = greedy_modularity_communities(G)
    return sorted(map(sorted, communities))

# Label propagation community detection method
def detect_communities_label_propagation(G):
    communities = list()
    #for c in asyn_lpa_communities(G):
    for c in label_propagation_communities(G):
        communities.append(sorted(c))
    return sorted(communities)

# Plot graph with communities, receives a list of communities, where each community is a list of nodes (ints)
def show_communities(G, communities, name='title'):
    pos=nx.spring_layout(G)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure()
    plt.title(name, fontsize=20)
    aux = 0
    for community in communities:
        nx.draw_networkx_nodes(G, pos, community, node_size = 50, node_color = colors[aux])
        aux = aux + 1
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show(block=True)

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

    '''
    https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf
    '''

if __name__ == "__main__":
    # Create the graph for benchmark
    G = benchmark_girvan_newman()

    # Get the true set of communities
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    communities = sorted(map(sorted, communities))

    bla = np.histogram2d([0,1,2], [0,1,2], bins=3)
    print(bla)


    #com_true = [0, 1, 2]
    #com_pred = [4, 5, 6]
    #com_true = [0,0,0,0,0,0,0,0,0,0]
    com_true = [2,2,2,2,2,2,2,2,2,2]
    #com_true = [0,0,0,1,1,1,2,2,2,2]
    com_pred = [2,2,2,2,2,2,2,2,2,2]
    #com_pred = [0,1,1,2,2,2,2,2,2,2]
    #com_pred = [1]

    #nmi = normalized_mutual_info_score(com_true, com_pred, average_method='arithmetic')
    nmi = adjusted_mutual_info_score(com_true, com_pred, average_method='arithmetic')
    #adjusted_mutual_info_score
    print(nmi)

    exit()

    # List of community detection methods
    methods = [ detect_communities_louvain,
    detect_communities_girvan_newman,
    detect_communities_greedy,
    detect_communities_label_propagation
    ]
    # For each method in the list
    for method in methods:
        # Apply community detection method on graph
        result = method(G)
        # Plot graph with its communities and name it
        show_communities(G, result, name=method.__name__[19:])
