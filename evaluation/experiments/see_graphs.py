import pickle
import networkx as nx
import random
import itertools


# Load the list of lists of graphs from a pickle file
input_pickle_file = '/workdir/generated_graphs/damnets-3comm.pkl'
with open(input_pickle_file, 'rb') as f:
    graphs = pickle.load(f)

print(len(graphs[0]))
