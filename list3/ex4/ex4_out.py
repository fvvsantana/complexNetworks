import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random as rd
import igraph as ig

#| # Exercise 4
#| ## Functions for statistics
def second_momment_of_degree_distribution(G):
    k,Pk = degree_distribution(G)
    M = sum((k**2)*Pk)
    return M

def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

#| ## Graph generator and processor
def generate_barabasi_albert(N, avg_degree, dataset):
    m = int(avg_degree/2)
    GBA = nx.barabasi_albert_graph(N, m)
    dataset.append(GBA)

def generate_igraph_BA(N, n_networks, alpha, dataset):
    for i in np.arange(n_networks):
        G = ig.Graph.Barabasi(n = N, power = alpha)
        dataset.append(G)

def generate_configuration_model(N, gamma, dataset):
    seq = np.random.zipf(gamma, N) #Zipf distribution
    if(sum(seq)%2 != 0): # the sum of stubs have to be even
        pos = rd.randint(0, len(seq) - 1)
        seq[pos] = seq[pos]+ 1
    GCM=nx.configuration_model(seq)
    Gcc=sorted(nx.connected_component_subgraphs(GCM), key = len, reverse=True)
    dataset.append(Gcc[0])

def process_graph(G, n_networks, mean_matrix, line):
    n_nodes_list = list()
    avg_degree_list = list()
    smd_distribution_list = list()
    avg_shortest_list = list()
    avg_clustering_list = list()
    transitivity_list = list()
    assortativity_list = list()
    for i in range(0, n_networks):
        N = len(G[i])
        M = G[i].number_of_edges()
        n_nodes_list.append(N)
        avg_degree_list.append(2*M/N)
        smd_distribution_list.append(second_momment_of_degree_distribution(G[i]))
        avg_shortest_list.append(nx.average_shortest_path_length(G[i]))
        avg_clustering_list.append(nx.average_clustering(G[i]))
        transitivity_list.append(nx.transitivity(G[i]))
        assortativity_list.append(nx.degree_assortativity_coefficient(G[i]))
    a = np.array(n_nodes_list)
    mean_matrix[line][0] = np.mean(a)

    a = np.array(avg_degree_list)
    mean_matrix[line][1] = np.mean(a)

    a = np.array(smd_distribution_list)
    mean_matrix[line][2] = np.mean(a)

    a = np.array(avg_shortest_list)
    mean_matrix[line][3] = np.mean(a)

    a = np.array(avg_clustering_list)
    mean_matrix[line][4] = np.mean(a)

    a = np.array(transitivity_list)
    mean_matrix[line][5] = np.mean(a)

    a = np.array(assortativity_list)
    mean_matrix[line][6] = np.mean(a)

def process_igraph(G, n_networks, mean_matrix, line):
    n_nodes_list = list()
    avg_degree_list = list()
    smd_distribution_list = list()
    avg_shortest_list = list()
    avg_clustering_list = list()
    transitivity_list = list()
    assortativity_list = list()
    for i in range(0, n_networks):
        Gi = nx.Graph([edge.tuple for edge in G[i].es])
        N = len(Gi)
        M = Gi.number_of_edges()
        n_nodes_list.append(N)
        avg_degree_list.append(2*M/N)
        smd_distribution_list.append(second_momment_of_degree_distribution(Gi))
        avg_shortest_list.append(nx.average_shortest_path_length(Gi))
        avg_clustering_list.append(nx.average_clustering(Gi))
        transitivity_list.append(nx.transitivity(Gi))
        assortativity_list.append(nx.degree_assortativity_coefficient(Gi))
    a = np.array(n_nodes_list)
    mean_matrix[line][0] = np.mean(a)

    a = np.array(avg_degree_list)
    mean_matrix[line][1] = np.mean(a)

    a = np.array(smd_distribution_list)
    mean_matrix[line][2] = np.mean(a)

    a = np.array(avg_shortest_list)
    mean_matrix[line][3] = np.mean(a)

    a = np.array(avg_clustering_list)
    mean_matrix[line][4] = np.mean(a)

    a = np.array(transitivity_list)
    mean_matrix[line][5] = np.mean(a)

    a = np.array(assortativity_list)
    mean_matrix[line][6] = np.mean(a)

def plot_alpha_curve(G, title):
    max_k = list()
    aux_max_k = list()

    N = np.arange(10, 10000, 10)
    for n in N:
        for i in range(0, len(G)):
            aux_max_k.append(max(G[i].degree()))
        max_k.append(np.mean(aux_max_k))
    plt.loglog(N, np.asarray(max_k), 'o' ,color = "red", label = title)

    plt.xlabel("N")
    plt.ylabel("Max_K")
    plt.legend()

def main():
    #| ## Information about the networks
    graphIndex=["Barabási-Albert", "Configuration Model"]
    measureIndex=["Number of nodes", "Average Degree", "2nd Moment of Degree", "Average Shortest Path Length", "Average Clustering", "Transitivity", "Assortativity"]
    n_networks = 30
    N = 10
    avg_degree = 10
    gamma = 3

    #| ## Creating the networks
    barabasi_albert_graphs = list()
    configuration_model_graphs = list()
    for i in range(0, n_networks):
        generate_barabasi_albert(N, avg_degree, barabasi_albert_graphs)
        generate_configuration_model(N, gamma, configuration_model_graphs)

    #| ## Processing the networks and generating the statistics
    w, h = 7, 2
    mean_matrix = [[0 for x in range(w)] for y in range(h)]

    print('barabasi:')
    print(barabasi_albert_graphs)
    process_graph(barabasi_albert_graphs, n_networks, mean_matrix, 0)

    print('config:')
    configuration_model_graphs = list(map(nx.Graph, configuration_model_graphs))
    print(configuration_model_graphs)
    process_graph(configuration_model_graphs, n_networks, mean_matrix, 1)

    #| ## Table with the statistics
    # Create DataFrame
    data = {'Mean ': measureIndex, 'Barabási-Albert': np.array(mean_matrix[0]).tolist(), 'Configuration Model': np.array(mean_matrix[1]).tolist()}
    df = pd.DataFrame(data)
    # Print the output.
    print(df)

    #| ## generate with igraph
    BA_graphs_05 = list()
    BA_graphs_1 = list()
    BA_graphs_15 = list()
    BA_graphs_25 = list()

    generate_igraph_BA(N, n_networks, 0.5, BA_graphs_05)
    generate_igraph_BA(N, n_networks, 1, BA_graphs_1)
    generate_igraph_BA(N, n_networks, 1.5, BA_graphs_15)
    generate_igraph_BA(N, n_networks, 2.5, BA_graphs_25)

    #| ## Plot igraph curves
    plot_alpha_curve(BA_graphs_05, "Alpha = 0.5")
    plot_alpha_curve(BA_graphs_1, "Alpha = 1")
    plot_alpha_curve(BA_graphs_25, "Alpha = 2.5")

    #| ## Processing the igraphs to generate table
    w, h = 7, 4
    mean_matrix = [[0 for x in range(w)] for y in range(h)]
    process_igraph(BA_graphs_05, n_networks, mean_matrix, 0)
    process_igraph(BA_graphs_1 , n_networks, mean_matrix, 1)
    process_igraph(BA_graphs_15, n_networks, mean_matrix, 2)
    process_igraph(BA_graphs_25, n_networks, mean_matrix, 3)

    #| ## Table with igraph statistics
    # Create DataFrame
    data = {'Mean ': measureIndex, 'BA (Alpha = 0.5)': np.array(mean_matrix[0]).tolist(), 'BA (Alpha = 1)': np.array(mean_matrix[1]).tolist(), 'BA (Alpha = 1.5)': np.array(mean_matrix[2]).tolist(), 'BA (Alpha = 2.5)': np.array(mean_matrix[3]).tolist()}
    df = pd.DataFrame(data)
    # Print the output.
    print(df)

    #| ## Comments

if __name__ == '__main__':
    main()
