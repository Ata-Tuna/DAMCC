import networkx as nx
import toponetx as tnx
import numpy as np
import torch

def to_device(tensor, device):
    return tensor.to(device)


def incidence_matrix_to_graph(incidence_matrix):
    """
    Converts a transposed incidence matrix to a NetworkX graph.

    Args:
    - incidence_matrix (torch.Tensor or np.ndarray): A transposed incidence matrix of size (m, n), where
      m is the number of edges and n is the number of nodes.

    Returns:
    - G (nx.Graph): A NetworkX graph constructed from the incidence matrix.
    """
    # Ensure the incidence matrix is a dense matrix
    if isinstance(incidence_matrix, torch.Tensor):
        incidence_matrix = incidence_matrix.to_dense() if incidence_matrix.is_sparse else incidence_matrix
        incidence_matrix = incidence_matrix.numpy()
    
    # Number of edges (rows) and nodes (columns)
    m, n = incidence_matrix.shape

    # Initialize an undirected graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(n))

    # Iterate through each edge (row in the incidence matrix)
    for edge_idx in range(m):
        # Get the non-zero entries in the row, which correspond to the nodes incident to the edge
        incident_nodes = np.where(incidence_matrix[edge_idx] != 0)[0]

        # If two nodes are connected by this edge, add an edge between them in the graph
        if len(incident_nodes) == 2:
            G.add_edge(incident_nodes[0], incident_nodes[1])

    return G

def generate_cc_from_transposed_incidence(incidence_0_1_T, incidence_0_2_T):
    """
    Generates a combinatorial complex (CC) from transposed incidence matrices between 0-cells and 1-cells, and 0-cells and 2-cells.

    Args:
    - incidence_0_1_T (torch.Tensor or np.ndarray): Transposed incidence matrix between 0-cells and 1-cells (edges).
    - incidence_0_2_T (torch.Tensor or np.ndarray): Transposed incidence matrix between 0-cells and 2-cells (faces).

    Returns:
    - cc (toponetx.CombinatorialComplex): A combinatorial complex generated from the transposed incidence matrices.
    """
    # Ensure the incidence matrices are dense if they are sparse
    if isinstance(incidence_0_1_T, torch.Tensor):
        incidence_0_1_T = incidence_0_1_T.to_dense() if incidence_0_1_T.is_sparse else incidence_0_1_T
        incidence_0_1_T = incidence_0_1_T.numpy()
    
    if isinstance(incidence_0_2_T, torch.Tensor):
        incidence_0_2_T = incidence_0_2_T.to_dense() if incidence_0_2_T.is_sparse else incidence_0_2_T
        incidence_0_2_T = incidence_0_2_T.numpy()

    # Create an empty combinatorial complex
    cc = tnx.CombinatorialComplex()

    # Add 0-cells (nodes) as cells of rank 0 individually
    num_nodes = incidence_0_1_T.shape[1]  # Number of 0-cells (columns of the incidence matrix)
    for i in range(num_nodes):
        cc.add_cell([i], rank=0)

    # Add 1-cells (edges) based on incidence_0_1_T
    num_edges = incidence_0_1_T.shape[0]  # Number of 1-cells (rows of the transposed incidence matrix)
    for edge_idx in range(num_edges):
        incident_nodes = list(np.where(incidence_0_1_T[edge_idx, :] != 0)[0])  # Get nodes for this edge
        if len(incident_nodes) == 2:  # Add edge only if there are exactly 2 nodes
            cc.add_cell(incident_nodes, rank=1)

    # Add 2-cells (faces) based on incidence_0_2_T
    num_faces = incidence_0_2_T.shape[0]  # Number of 2-cells (rows of the transposed incidence matrix)
    for face_idx in range(num_faces):
        incident_nodes = list(np.where(incidence_0_2_T[face_idx, :] != 0)[0])  # Get nodes for this face
        if len(incident_nodes) > 2:  # Add face if there are more than 2 nodes
            cc.add_cell(incident_nodes, rank=2)

    return cc

def divide_tensors_into_lists(list_of_lists, list_of_tensors):
    """
    Divide a list of tensors into smaller lists to match the shape of the list of lists of tensors.

    Parameters
    ----------
    list_of_lists : list of list of torch.Tensor
        A list containing inner lists of tensors, where each inner list has tensors of the same shape.
    list_of_tensors : list of torch.Tensor
        A list of tensors to be divided into smaller lists.

    Returns
    -------
    list of list of torch.Tensor
        A list of lists of tensors, where each inner list has the same length as the inner lists in list_of_lists.
    """
    # Get the length of the first inner list in list_of_lists
    inner_list_length = len(list_of_lists[0])

    # Check if the total number of tensors is divisible by the inner list length
    if len(list_of_tensors) % inner_list_length != 0:
        raise ValueError("The number of tensors in list_of_tensors must be divisible by the length of the inner lists in list_of_lists.")

    # Divide the list_of_tensors into smaller lists
    divided_tensors = [list_of_tensors[i:i + inner_list_length] for i in range(0, len(list_of_tensors), inner_list_length)]

    return divided_tensors

def divide_tensors(list_of_lists, list_of_tensors, sample=True):
    """
    Divide a list of tensors into smaller lists to match the shape of the list of lists of tensors.

    Parameters
    ----------
    list_of_lists : list of list of torch.Tensor
        A list containing inner lists of tensors, where each inner list has tensors of the same shape.
    list_of_tensors : list of torch.Tensor
        A list of tensors to be divided into smaller lists.

    Returns
    -------
    list of list of torch.Tensor
        A list of lists of tensors, where each inner list has the same length as the inner lists in list_of_lists.
    """
    # Get the length of the first inner list in list_of_lists
    inner_list_length = len(list_of_lists[0])

    # Check if the total number of tensors is divisible by the inner list length
    # if len(list_of_tensors) % inner_list_length != 0:
    #     raise ValueError("The number of tensors in list_of_tensors must be divisible by the length of the inner lists in list_of_lists.")

    # Divide the list_of_tensors into smaller lists
    divided_tensors = [list_of_tensors[i:i + inner_list_length] for i in range(0, len(list_of_tensors), inner_list_length)]
    if sample:
        divided_tensors = [list[:-1] for list in divided_tensors]
    else:
        divided_tensors = [list[1:] for list in divided_tensors]
    return divided_tensors



# Example usage:
# incidence_0_1_T = torch.tensor(...)  # Transposed incidence matrix for edges
# incidence_0_2_T = torch.tensor(...)  # Transposed incidence matrix for faces
# cc = generate_cc_from_transposed_incidence(incidence_0_1_T, incidence_0_2_T)
