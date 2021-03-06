#| # Exercise 6

#| Import the libraries that we'll use
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
import pandas as pd
from IPython.display import display, HTML, display_pretty
import os
np.random.seed(50)

#| ## Benchmark function
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

#| ## Method functions
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

#| ## Function to plot communities
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

#| This function will help us to turn the partitions generated by the methods into
#| classification lists as explained
# Turn a list of communities, where each community is a list of nodes, into a classification.
#  Classification is a list of classes, where the value v in the position i means that the node
#  i belongs to the community v.
def communitiesToClassification(communities):
    # Get number of nodes
    nNodes = len(G)
    # Initialize a list of size nNodes
    classification = [0]*nNodes
    # Go through the list filling it with the classes
    for i in range(len(communities)):
        for j in communities[i]:
            classification[j] = i
    # Return classification
    return classification

#| Now that we did almost everything we'll need, let's call our functions
if __name__ == "__main__":
    # Create the graph for benchmark
    G = benchmark_girvan_newman()

    # Get the true set of communities
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    communities = sorted(map(sorted, communities))
    # Turn partition into classification
    realClassification = communitiesToClassification(communities)

    # List of method names
    methodNames = [
        'Louvain',
        'Girvan Newman',
        'Fast Greedy',
        'Label Propagation'
    ]

    # List of community detection methods
    methods = [
        detect_communities_louvain,
        detect_communities_girvan_newman,
        detect_communities_greedy,
        detect_communities_label_propagation
    ]

    # Dict where we'll put the results
    data = {'Method': methodNames, 'Normalized Mutual Information':[]}

    # For each method in the list
    for i in range(len(methods)):
        # Apply community detection method on graph
        result = methods[i](G)
        # Plot graph with its communities and name it
        #show_communities(G, result, name=methodNames[i])
        # Turn communities into a classification list
        classification = communitiesToClassification(result)
        # Calculate Normalized Mutual Information
        nmi = normalized_mutual_info_score(realClassification, classification, average_method='arithmetic')
        # Append NMI
        data['Normalized Mutual Information'].append(nmi)

    # Display DataFrame
    df = pd.DataFrame(data)
    display(df)

#| As we can see, the girvan newman method was the worse on accuracy and also on time of execution. The other methods did a good job on classifying the communities, with 100% accuracy.
