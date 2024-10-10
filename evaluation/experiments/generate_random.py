import pickle
import networkx as nx
import random
import itertools
import numpy as np

def generate_random_graphs(input_graphs):
    """
    Generate a list of lists of networkx graphs with random adjacency matrices
    of the same size as the input graphs.
    
    Args:
        input_graphs (list of list of nx.Graph): A list of lists where each element is a nx.Graph.
        
    Returns:
        list of list of nx.Graph: A list of lists of graphs with random adjacency matrices.
    """
    random_graphs = []
    
    for graph_list in input_graphs:
        random_graph_list = []
        for graph in graph_list:
            num_nodes = graph.number_of_nodes()
            random_adj_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
            
            # Making sure the adjacency matrix is symmetric for an undirected graph
            random_adj_matrix = np.triu(random_adj_matrix) + np.triu(random_adj_matrix, 1).T
            
            # Create a new random graph from the adjacency matrix
            random_graph = nx.from_numpy_array(random_adj_matrix)
            random_graph_list.append(random_graph)
        
        random_graphs.append(random_graph_list)
    
    return random_graphs

# Load the list of lists of graphs from a pickle file
input_pickle_file = '/workdir/generated_graphs_test/ba-test-graphs.pkl'
with open(input_pickle_file, 'rb') as f:
    graphs = pickle.load(f)

Gs = graphs
# Get the adjacency matrix as a sparse matrix
RGs = generate_random_graphs(Gs)
# print((nx.adjacency_matrix(RGs[0][0]).todense()))

# Optionally, save the random binary graphs to a new pickle file
output_pickle_file = '/workdir/generated_graphs/random-ba.pkl'
with open(output_pickle_file, 'wb') as f:
    pickle.dump(RGs, f)

print(f"Random binary graphs generated and saved to {output_pickle_file}")
