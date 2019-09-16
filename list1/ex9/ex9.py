#| # Exercise 9
#| Import the libraries that we'll use
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import expm
import pandas as pd
import time
from IPython.display import display, HTML, display_pretty
#import warnings
#warnings.simplefilter("ignore")

# A flag to use timer, just to prevent your system to freeze while running heavy processes
useTimer = False #this feature is disabled

#| Add some functions to help us
# Calculate the n-th moment of a probability distribution
def nth_moment_of_probability_distribution(xList, pxList, n):
    return expected_value([x**n for x in xList], pxList)

# Calculate the expected value of a random variable
def expected_value(xList, pxList):
    return sum([xList[i]*pxList[i] for i in range(len(xList))])

# Calculate the shannon entropy of an array of probabilities
def shannon_entropy(pxList):
    H = 0
    for p in pxList:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

#| Create a class to encapsulate and do the graph operations
class Network:

    def __init__(self, name=None):
        self.name = name

    # Read graph from file and apply transformations
    def read_graph(self, inputFile):
        # To read the network from a file, we use the command read_edgelist.
        G= nx.read_edgelist(inputFile, comments='%', nodetype=int, data=(('weight',float),))
        # We transfor the network into the undirected version.
        G = G.to_undirected()
        # Here we consider only the largest component.
        Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
        G=Gcc[0]
        # Sometimes the node labels are not in the sequential order or strings are used. To facilitate our implementation, let us convert the labels to integers starting with the index zero, because Python uses 0-based indexing.
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        # Save graph to the network
        self.graph = G
        return self.graph

    # Store and return the transitivity of the graph
    def transitivity(self):
        return nx.transitivity(self.graph)

    # Store and return the average clustering coefficient of the graph
    def average_clustering(self):
        return nx.average_clustering(self.graph)

    # Get number of edges of the graph
    def number_of_edges(self):
        return self.graph.number_of_edges()

    # Get number of nodes of the graph
    def number_of_nodes(self):
        return len(self.graph)

    # Calculate average node degree
    def average_node_degree(self):
        # Get the degrees
        degrees = [d for n,d in self.graph.degree()]
        # Calculate the average
        return sum(degrees)/len(degrees)


    # Calculate the nth moment of degree distribution
    def nth_moment_of_degree_distribution(self, n):
        kvalues, pk = self.degree_distribution()
        return nth_moment_of_probability_distribution(kvalues, pk, n)

    # Calculate average shortest path length
    def average_shortest_path_length(self, weight=None):
        return nx.average_shortest_path_length(self.graph, weight)

    # Calculate diameter
    def diameter(self, e=None):
        return nx.diameter(self.graph, e)

    # Calculate degree distribution
    def degree_distribution(self):
        vk = dict(self.graph.degree())
        vk = list(vk.values())  # we get only the degree values
        vk = np.array(vk)
        maxk = np.max(vk)
        mink = np.min(vk)
        kvalues= np.arange(0,maxk+1) # possible values of k
        Pk = np.zeros(maxk+1) # P(k)
        for k in vk:
            Pk[k] = Pk[k] + 1
        Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
        return (kvalues, Pk)


    # Plot degree distribution
    def plot_degree_distribution(self):
        degree_distribution = self.degree_distribution()
        fig = plt.subplot(1,1,1)
        fig.set_xscale('log')
        fig.set_yscale('log')
        plt.suptitle(self.name, fontsize=16)
        plt.plot(degree_distribution[0], degree_distribution[1],'bo')
        plt.xlabel("k", fontsize=20)
        plt.ylabel("P(k)", fontsize=20)
        plt.title('Degree distribution', fontsize=20)
        plt.show(block=True)
        plt.clf()

    # Plot the graph
    def plot_graph(self):
        plt.figure(figsize=(10,10))
        pos=nx.spring_layout(self.graph)
        nx.draw(self.graph, with_labels = True, pos = pos)
        plt.show(block=True)
        plt.clf()

#| Now we'll calculate the measures for each network. Then we store it in a dict to present it at the end.
def main():
    # List of files to open
    networkFiles = [
                    #"../data/out.ca-AstroPh",
                    #'../data/out.ego-facebook',
                    #'../data/out.petster-friendships-hamster-uniq',
                    '../data/out.subelj_euroroad_euroroad'
    ]

    # List of names of the networks
    networkNames = [
                    #'ArXiv’s Astrophysics',
                    #'Facebook user-user friendships',
                    #'Hamsterster friendships',
                    'E-road network'
    ]

    # Dict with data to show
    data = {'Network':[],
            'Nodes':[], 
            'Average degree':[],
            'Second moment of degree distribution':[],
            'Average clustering coefficient':[],
            'Transitivity':[],
            'Average shortest path length':[],
            'Diameter':[],
            }

    # For each network
    for i in range(len(networkFiles)):
        # Load network
        network = Network(name=networkNames[i])
        network.read_graph(networkFiles[i])

        # Let us verify the number of nodes and edges of the network.
        #print('Number of nodes:', network.number_of_nodes())
        #print('Number of edges:', network.number_of_edges())

        # Append network name
        data['Network'].append(networkNames[i])

        # Get number of nodes
        data['Nodes'].append(network.number_of_nodes())

        # Get number of nodes
        data['Average degree'].append(network.average_node_degree())

        # Get Second moment of degree distribution
        data['Second moment of degree distribution'].append(network.nth_moment_of_degree_distribution(2))

        # Calculate Average Clustering Cofficient
        data['Average clustering coefficient'].append(network.average_clustering())

        # Calculate transitivity
        data['Transitivity'].append(network.transitivity())

        # Calculate average shortest path length
        data['Average shortest path length'].append(network.average_shortest_path_length())

        # Calculate diameter
        data['Diameter'].append(network.diameter())

    # Display DataFrame
    df = pd.DataFrame(data)
    display(df)


#| Here we create a process to run the main function. The advantage of using this strategy, instead of directly invoking main(), is that it gives us more control to stop the program.
#| If we set the useTimer variable at the beginning of the program to True, the timer will stop the main process after 60 seconds. It's useful when we hit Ctrl+C in the terminal and the program refuses to stop.
#| Also, it's important to catch the KeyboardInterrupt exception, that is raised when we hit Ctrl+C.
#| By default, Ctrl+C would kill our program and the process that we've created would still continue to run.
#| When we catch the exception, we also terminate the process of the function main. This way, Ctrl+C works again.


from multiprocessing import Process
from threading import Timer
if __name__ == "__main__":
    # Create the main process
    proc = Process(target=main)

    # Use timer if set
    if(useTimer):
        #declare timer, it gets the time in seconds
        timer = Timer(300, proc.terminate)
        timer.start()

    try:
        # Start process
        proc.start()

        # Block until process terminate
        proc.join()
    except KeyboardInterrupt:
        proc.terminate()

    # If process ends in time, cancel timer
    if(useTimer):
        timer.cancel()

