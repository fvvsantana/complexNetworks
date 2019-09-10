
#| # Exercise 1
#| Import the libraries that we'll use
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
#import warnings
#warnings.simplefilter("ignore")

#| Create a graph as specified in the exercise
G = nx.Graph()
G.add_edge(0,1,weight=1)
G.add_edge(0,3,weight=1)
G.add_edge(1,2,weight=1)
G.add_edge(1,3,weight=1)
G.add_edge(2,3,weight=1)
G.add_edge(2,4,weight=1)
G.add_edge(2,5,weight=1)
G.add_edge(2,6,weight=1)
G.add_edge(5,6,weight=1)

#| Plot the graph
# Return a dict keyed by node with its coordinates
pos=nx.spring_layout(G)
# Draw the graph using matplot in the background
nx.draw_networkx(G, with_labels=True, node_color='r', edge_color='b', 
        node_size=500, font_size=16, pos=pos, width=6)   # default spring_layout
# The parameter block=True blocks the execution until the user closes it
plt.show(block=False)
plt.clf()



#| Let's start calculating the measures

#| ### Degree distribution
# Degree distribution function
def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values())  # we get only the degree values
    vk = np.array(vk)
    maxk = np.max(vk)
    mink = np.min(vk)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

#| Calculate
ks, Pk = degree_distribution(G)
print('Degrees = ', ks)
print('Probabilities = ', Pk)

#| Plot
fig = plt.subplot(1,1,1)
fig.set_xscale('log')
fig.set_yscale('log')
plt.plot(ks,Pk,'bo')
plt.xlabel("k", fontsize=20)
plt.ylabel("P(k)", fontsize=20)
plt.title("Degree distribution", fontsize=20)
plt.show(block=False)
plt.clf()


#| ### Transitivity
#| Calculate transitivity
CC = (nx.transitivity(G)) 
print("Transitivity = ","%3.4f"%CC)


#| ### Distance matrix

#| Distance matrix function
def distance_matrix(G):
    N=len(G)
    D = np.zeros(shape=(N,N)) # D is the matrix of distances
    for i in np.arange(0,N):
        for j in np.arange(i+1, N):
            D[i][j] = D[j][i] = nx.shortest_path_length(G,i,j) - 1
    return D

#| Calculate and print distance matrix
if nx.is_connected(G) == True:
    D = distance_matrix(G)
    print('Distance Matrix =\n', D)
else:
    print("The graph has more than one connected component")


#| ### Shannon Entropy
#| Shannon Entropy function
def shannon_entropy(G):
    k,Pk = degree_distribution(G)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

#| Calculate
H = shannon_entropy(G)
print("Shannon Entropy = ", "%3.4f"%H)


#| ### Second moment of the degree distribution
#| Second moment of the degree distribution
def momment_of_degree_distribution(G,m):
    k,Pk = degree_distribution(G)
    M = sum((k**m)*Pk)
    return M

#| Calculate
k2 = momment_of_degree_distribution(G,2)
print("Second moment of the degree distribution = ", k2)





