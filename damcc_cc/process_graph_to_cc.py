import sys
import os
import toponetx as tpx
import networkx as nx
import pickle 
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import argparse
import numpy as np_array

def ensure_nonzero_dim(tensor, axis=0):
    if tensor.shape[axis] == 0:
        shape = list(tensor.shape)
        shape[axis] = 1  # Set the 0-dimension to 1

        if axis == 0:
            # Create a zero row (sparse)
            zero_tensor = torch.zeros(1, shape[1], dtype=tensor.dtype).to_sparse()
        else:
            # Create a zero column (sparse)
            zero_tensor = torch.zeros(shape[0], 1, dtype=tensor.dtype).to_sparse()

        tensor = torch.cat([tensor, zero_tensor], dim=axis)
    
    return tensor


class CCDataset(Dataset):
    """Class for the SHREC 2016 dataset.

    Parameters
    ----------
    data : npz file
        npz file containing the SHREC 2016 data.
    """
     
    def __init__(self, data, test=False) -> None:
        self.data = data
        self.test = test
        self.complexes = self._graph_data_to_cc_data(data)
        self.a01, self.a02, self.coa2, self.b1, self.b2, self.b10, self.b20, self.b10_t, self.b20_t = self._get_neighborhood_matrix(self.complexes)
        self.x_0 = self._extract_x_0(data)
        self.x_1, self.x_2 = self._extract_x_1_x_2(self.complexes)
        # self.down_laplacian_list, self.up_laplacian_list, self.adjacency_0_list = self._extract_laplacians_and_adjacency(self.complexes)

        # Flatten the lists of lists of tensors into a single list of tensors
        self.a01 = self._flatten_list_of_lists(self.a01)

        
        self.a02 = self._flatten_list_of_lists(self.a02)
        self.coa2 = self._flatten_list_of_lists(self.coa2)
        self.b1 = self._flatten_list_of_lists(self.b1)
        self.b2 = self._flatten_list_of_lists(self.b2)
        self.b10 = self._flatten_list_of_lists(self.b10)
        self.b20 = self._flatten_list_of_lists(self.b20)
        self.b10_t = self._flatten_list_of_lists(self.b10_t)
        self.b20_t = self._flatten_list_of_lists(self.b20_t)
        self.x_0 = self._flatten_list_of_lists(self.x_0)
        self.x_1 = self._flatten_list_of_lists(self.x_1)
        self.x_2 = self._flatten_list_of_lists(self.x_2)
        self.complexes = self._flatten_list_of_lists(self.complexes)
        # self.down_laplacian_list = self._flatten_list_of_lists(self.down_laplacian_list)
        # self.up_laplacian_list = self._flatten_list_of_lists(self.up_laplacian_list)
        # self.adjacency_0_list = self._flatten_list_of_lists(self.adjacency_0_list)

        # Specify the number of binary values you want
        num_values = len(self.a01)  # Example: generate 10 binary values
        print(num_values)
        # Generate a random tensor of binary values (0 or 1)    
        random_binary_values = torch.randint(low=0, high=2, size=(num_values,))

        # Convert to a list if needed
        random_binary_list = random_binary_values.tolist()
        self.y = random_binary_values

    def _flatten_list_of_lists(self, list_of_lists):
        # Utility function to flatten a list of lists into a single list
        return [tensor for sublist in list_of_lists for tensor in sublist]

    def _convert_graph_to_cc_via_clique(self, G):

        G = G.to_undirected()

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        cc = tpx.transform.graph_to_simplicial_complex.graph_to_clique_complex(G)
        cc = cc.to_cell_complex()
        for edge in G.edges():
            print(edge)
            cc.add_cell(edge, rank=1)

        # Add nodes (0-cells)
        for node in G.nodes():
            print(node)
            cc.add_node(node)
        cc = cc.to_combinatorial_complex()
        return cc

    def _graph_data_to_cc_data(self, graph_data):
        list_of_graph_seqs = graph_data
        cc_data = []  # Initialize a list to hold the combinatorial complexes
        for graph_seq in list_of_graph_seqs:
            cc_seq = []  # Initialize a list for the current sequence of CCs
            print('Converting graphs to ccs')
            for graph in tqdm(graph_seq):
                cc = self._convert_graph_to_cc_via_clique(graph)  # Convert graph to CC
                cc_seq.append(cc)  # Append the CC to the current sequence
            cc_data.append(cc_seq)  # Append the sequence of CCs to the main list
        return cc_data  # Return the list of CCs with the same structure as graph_data

    def _extract_x_0(self, graph_data):
        list_of_graph_seqs = graph_data
        cc_data = []  # Initialize a list to hold the combinatorial complexes
        for graph_seq in list_of_graph_seqs:
            cc_seq = []  # Initialize a list for the current sequence of CCs
            print('Converting graphs to ccs')
            for graph in tqdm(graph_seq):
                G = graph
                # for 3comm only change for any other
                # # Check if the graph has node features
                if len(G.nodes) > 0 and len(list(G.nodes(data=True))[0][1]) > 0:
                    # If features are present, extract them
                    feature_matrix = torch.Tensor([list(G.nodes[node].values())[0] for node in G.nodes()])
                else:
                    # If no features are present, use an identity matrix as the feature matrix
                    feature_matrix = torch.eye(len(G.nodes))
                # feature_matrix = torch.eye(len(G.nodes))
                cc_seq.append(feature_matrix)  # Append the feature matrix to the current sequence
            if not self.test:
                cc_data.append(cc_seq[:-1])  # Append the sequence of CCs to the main list
            else:        
                cc_data.append(cc_seq)
        return cc_data  # Return the list of CCs with the same structure as graph_data



    def _extract_x_1_x_2(self, complexes):
        x1batch = []  # Initialize a list to hold the combinatorial complexes
        x2batch = []
        for cc_seq in complexes:
            x1_seq = []  # Initialize a list for the current sequence of CCs
            x2_seq = []
            print("extracting features")
            for cc in tqdm(cc_seq):

                B2 = cc.incidence_matrix(1, 2, index=False)
                B2 = B2.todense()

                if B2.shape[1] == 0 and B2.shape[0] != 0:
                    dims = (B2.shape[0], 1)
                elif B2.shape[1] != 0 and B2.shape[0] == 0:
                    dims = (1, B2.shape[1])
                elif B2.shape[0] == 0 and B2.shape[1] == 0:
                    dims = (1,1)
                x1_feature_matrix = torch.ones(dims[0],1)
                x2_feature_matrix = torch.ones(dims[1],1)
                x1_seq.append(x1_feature_matrix)  # Append the CC to the current sequence
                x2_seq.append(x2_feature_matrix)  # Append the CC to the current sequence
            
            if not self.test:
                x1batch.append(x1_seq[:-1])  # Append the sequence of CCs to the main list
                x2batch.append(x2_seq[:-1])  # Append the sequence of CCs to the main list
            else:
                x1batch.append(x1_seq)  # Append the sequence of CCs to the main list
                x2batch.append(x2_seq)  # Append the sequence of CCs to the main list


        return x1batch, x2batch  # Return the list of CCs with the same structure as graph_data

    # def _extract_laplacians_and_adjacency(self, complexes):
    #     down_laplacian_list = []
    #     up_laplacian_list = []
    #     adjacency_0_list = []
    #     for cc_seq in complexes:
    #         down_laplacian_seq = []
    #         up_laplacian_seq = []
    #         adjacency_0_seq = []
    #         for cell_complex in cc_seq:
    #             adjacency_0 = cell_complex.adjacency_matrix(rank=0)
    #             adjacency_0 = torch.from_numpy(adjacency_0.todense()).to_sparse()
    #             adjacency_0_seq.append(adjacency_0)

    #             down_laplacian_t = cell_complex.down_laplacian_matrix(rank=1)
    #             down_laplacian_t = torch.from_numpy(down_laplacian_t.todense()).to_sparse()
    #             down_laplacian_seq.append(down_laplacian_t)

    #             try:
    #                 up_laplacian_t = cell_complex.up_laplacian_matrix(rank=1)
    #                 up_laplacian_t = torch.from_numpy(up_laplacian_t.todense()).to_sparse()
    #             except ValueError:
    #                 up_laplacian_t = torch.zeros((down_laplacian_t.shape[0], down_laplacian_t.shape[0])).to_sparse()

    #             up_laplacian_seq.append(up_laplacian_t)

    #         down_laplacian_list.append(down_laplacian_seq)
    #         up_laplacian_list.append(up_laplacian_seq)
    #         adjacency_0_list.append(adjacency_0_seq)

    #     return down_laplacian_list, up_laplacian_list, adjacency_0_list

    def _get_neighborhood_matrix(self, complexes) -> list[list[torch.sparse.Tensor], ...]:
        """Neighborhood matrices for each combinatorial complex in the dataset.

        Following the Higher Order Attention Model for Mesh Classification message passing scheme, this method computes the necessary neighborhood matrices
        for each combinatorial complex in the dataset. This method computes:

        - Adjacency matrices for each 0-cell in the dataset.
        - Adjacency matrices for each 1-cell in the dataset.
        - Coadjacency matrices for each 2-cell in the dataset.
        - Incidence matrices from 1-cells to 0-cells for each 1-cell in the dataset.
        - Incidence matrices from 2-cells to 1-cells for each 2-cell in the dataset.

        Returns
        -------
        a01 : list of torch.sparse.FloatTensor
            Adjacency matrices for each 0-cell in the dataset.
        a02 : list of torch.sparse.FloatTensor
            Adjacency matrices for each 1-cell in the dataset.
        coa2 : list of torch.sparse.FloatTensor
            Coadjacency matrices for each 2-cell in the dataset.
        b1 : list of torch.sparse.FloatTensor
            Incidence matrices from 1-cells to 0-cells for each 1-cell in the dataset.
        b2 : list of torch.sparse.FloatTensor
            Incidence matrices from 2-cells to 1-cells for each 2-cell in the dataset.
        """

        a01batch = []
        a02batch = []
        coa2batch = []
        b1batch = []
        b2batch = []
        cob01batch = []
        cob02batch = []
        target_cob01batch = []
        target_cob02batch = []
        #this one is just for me to test if it trains
        # y_batch = []
        print(len(complexes))
        for cc_seq in complexes:
            a01seq = []
            a02seq = []
            coa2seq = []
            b1seq = []
            b2seq = []
            cob01seq = []
            cob02seq = []
            target_cob01seq = []
            target_cob02seq = []
            # y_seq = []
            for cc in cc_seq:

                # a01seq.append(torch.from_numpy(cc.adjacency_matrix(0, 1).todense()).to_sparse())


                # a02seq.append(torch.from_numpy(cc.adjacency_matrix(1, 2).todense()).to_sparse())

                # B = cc.incidence_matrix(rank=1, to_rank=2)
                # A = B.T @ B
                # A.setdiag(0)
                # coa2seq.append(torch.from_numpy(A.todense()).to_sparse())

                # b1seq.append(torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to_sparse())
                # b2seq.append(torch.from_numpy(cc.incidence_matrix(1, 2).todense()).to_sparse())

                # cob01seq.append(torch.from_numpy(cc.incidence_matrix(0, 1).todense().T).to_sparse())
                # cob02seq.append(torch.from_numpy(cc.incidence_matrix(0, 2).todense().T).to_sparse())

                # Assuming cc is defined and provides the adjacency and incidence matrices

                a01 = torch.from_numpy(cc.adjacency_matrix(0, 1).todense()).to_sparse()
                a01 = ensure_nonzero_dim(a01, axis=0)  # Ensure rows are non-zero
                a01 = ensure_nonzero_dim(a01, axis=1)  # Ensure columns are non-zero
                a01seq.append(a01)

                a02 = torch.from_numpy(cc.adjacency_matrix(1, 2).todense()).to_sparse()
                a02 = ensure_nonzero_dim(a02, axis=0)  # Ensure rows are non-zero
                a02 = ensure_nonzero_dim(a02, axis=1)  # Ensure columns are non-zero
                a02seq.append(a02)

                B = cc.incidence_matrix(rank=1, to_rank=2)
                A = B.T @ B
                A.setdiag(0)
                coa2 = torch.from_numpy(A.todense()).to_sparse()
                coa2 = ensure_nonzero_dim(coa2, axis=0)  # Ensure rows are non-zero
                coa2 = ensure_nonzero_dim(coa2, axis=1)  # Ensure columns are non-zero
                coa2seq.append(coa2)

                b1 = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to_sparse()
                b1 = ensure_nonzero_dim(b1, axis=0)
                b1 = ensure_nonzero_dim(b1, axis=1)
                b1seq.append(b1)

                b2 = torch.from_numpy(cc.incidence_matrix(1, 2).todense()).to_sparse()
                b2 = ensure_nonzero_dim(b2, axis=0)
                b2 = ensure_nonzero_dim(b2, axis=1)
                b2seq.append(b2)

                cob01 = torch.from_numpy(cc.incidence_matrix(0, 1).todense().T).to_sparse()
                cob01 = ensure_nonzero_dim(cob01, axis=0)
                cob01 = ensure_nonzero_dim(cob01, axis=1)
                cob01seq.append(cob01)

                cob02 = torch.from_numpy(cc.incidence_matrix(0, 2).todense().T).to_sparse()
                cob02 = ensure_nonzero_dim(cob02, axis=0)
                cob02 = ensure_nonzero_dim(cob02, axis=1)
                cob02seq.append(cob02)


            # Conditionally remove first and last elements if not in test mode
            if not self.test:
                # Remove the first element in the target sequence
                target_cob01seq = cob01seq[1:]
                target_cob02seq = cob02seq[1:]

                # Remove the last element in the sequence to predict from
                a01seq = a01seq[:-1]
                a02seq = a02seq[:-1]
                coa2seq = coa2seq[:-1]
                b1seq = b1seq[:-1]
                b2seq = b2seq[:-1]
                cob01seq = cob01seq[:-1]
                cob02seq = cob02seq[:-1]
            else:
                #not used in test anyways
                target_cob01seq = cob01seq
                target_cob02seq = cob02seq
     
            a01batch.append(a01seq)
            a02batch.append(a02seq)
            coa2batch.append(coa2seq)
            b1batch.append(b1seq)
            b2batch.append(b2seq)
            cob01batch.append(cob01seq)
            cob02batch.append(cob02seq)
            target_cob01batch.append(target_cob01seq)
            target_cob02batch.append(target_cob02seq)
        # y_batch.append(y_seq)

        return a01batch, a02batch, coa2batch, b1batch, b2batch, cob01batch, cob02batch, target_cob01batch, target_cob02batch



    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        Returns
        -------
        int
            Number of classes in the dataset.
        """
        return len(np.unique(self.y))

    def channels_dim(self) -> tuple[int, int, int]:
        """Returns the number of channels for each input signal.

        Returns
        -------
        tuple of int
            Number of channels for each input signal.
        """
        return [self.x_0[0].shape[1], self.x_1[0].shape[1], self.x_2[0].shape[1]]

    def __len__(self) -> int:
        """Returns the number of elements in the dataset.

        Returns
        -------
        int
            Number of elements in the dataset.
        """
        return len(self.complexes)

    def get_via_indices(self, id1x, id2x) -> tuple[torch.Tensor, ...]:
        """Returns the idx-th element in the dataset.

        Parameters
        ----------
        idx : int
            Index of the element to return.

        Returns
        -------
        tuple of torch.Tensor
            Tuple containing the idx-th element in the dataset, including the input signals on nodes, edges and faces, the neighborhood matrices and the label.
        """
        return (
            self.x_0[id1x][id2x],
            self.a01[id1x][id2x],
            self.a02[id1x][id2x],
            self.b10[id1x][id2x],
            self.b20[id1x][id2x],
            self.b10_t[id1x][id2x],
            self.b20_t[id1x][id2x],
            # self.down_laplacian_list[id1x][id2x],
            # self.up_laplacian_list[id1x][id2x],
            # self.adjacency_0_list[id1x][id2x]
        )

    def __getitem__(self, id1x) -> tuple[torch.Tensor, ...]:
        """Returns the idx-th element in the dataset.

        Parameters
        ----------
        id1x : int
            Index of the element to return.

        Returns
        -------
        tuple of torch.Tensor
            Tuple containing the idx-th element in the dataset, including the input signals on nodes, edges, and faces, the neighborhood matrices, and the label.
        """
        return (
            self.x_0[id1x],
            self.x_1[id1x],
            self.a01[id1x],
            self.a02[id1x],
            self.b10[id1x],
            self.b20[id1x],
            self.b10_t[id1x],
            self.b20_t[id1x]
            # self.down_laplacian_list[id1x],
            # self.up_laplacian_list[id1x],
            # self.adjacency_0_list[id1x]
        )



    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        Returns
        -------
        int
            Number of classes in the dataset.
        """
        return len(np.unique(self.y))

    def channels_dim(self) -> tuple[int, int, int]:
        """Returns the number of channels for each input signal.

        Returns
        -------
        tuple of int
            Number of channels for each input signal.
        """
        return [self.x_0[0].shape[1], self.x_1[0].shape[1], self.x_2[0].shape[1]]

    def __len__(self) -> int:
        """Returns the number of elements in the dataset.

        Returns
        -------
        int
            Number of elements in the dataset.
        """
        return len(self.complexes)

    def get_via_indices(self, id1x, id2x) -> tuple[torch.Tensor, ...]:
        """Returns the idx-th element in the dataset.

        Parameters
        ----------
        idx : int
            Index of the element to return.

        Returns
        -------
        tuple of torch.Tensor
            Tuple containing the idx-th element in the dataset, including the input signals on nodes, edges and faces, the neighborhood matrices and the label.
        """
        return (
            self.x_0[id1x][id2x],
            # self.x_1[id1x][id2x],
            # self.x_2[id1x][id2x],
            self.a01[id1x][id2x],
            self.a02[id1x][id2x],
            # self.coa2[id1x][id2x],
            # self.b1[id1x][id2x],
            # self.b2[id1x][id2x],
            self.b10[id1x][id2x],
            self.b20[id1x][id2x],
            self.b10_t[id1x][id2x],
            self.b20_t[id1x][id2x]
            # self.y[id1x][id2x]
        )

    def __getitem__(self, id1x) -> tuple[torch.Tensor, ...]:
        """Returns the idx-th element in the dataset.

        Parameters
        ----------
        id1x : int
            Index of the element to return.

        Returns
        -------
        tuple of torch.Tensor
            Tuple containing the idx-th element in the dataset, including the input signals on nodes, edges, and faces, the neighborhood matrices, and the label.
        """
        return (
            self.x_0[id1x],
            self.x_1[id1x],
            self.x_2[id1x],
            self.a01[id1x],
            self.a02[id1x],
            self.coa2[id1x],    
            self.b1[id1x],
            self.b2[id1x],
            self.b10[id1x],
            self.b20[id1x],
            self.b10_t[id1x],
            self.b20_t[id1x]
            # self.down_laplacian_list[id1x],
            # self.up_laplacian_list[id1x],
            # self.adjacency_0_list[id1x]
        )

def main(input_file, output_dir, test):
    # Unpickle the file
    with open(input_file, 'rb') as file:
        graph_data = pickle.load(file)

    # Create the dataset, pass the test argument
    cc_data = CCDataset(graph_data, test=test)

    # Extract the base name and construct the output file name
    base_name = os.path.basename(input_file)  # Get the base name of the input file
    name_without_extension = os.path.splitext(base_name)[0]  # Remove the file extension

    # Remove the words "graphs" and "raw" from the base name
    name_without_extension = name_without_extension.replace("graphs", "").replace("raw", "").strip("_")

    output_file_name = f"{name_without_extension}_ccs.pkl"  # Append '_ccs' to the name
    output_file = os.path.join(output_dir, output_file_name)  # Construct the full output file path

    # Save the processed data
    with open(output_file, 'wb') as file:
        pickle.dump(cc_data, file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process graph data and save as CCDataset.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input pickle file containing graph data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed CCDataset.')
    parser.add_argument('--Test', action='store_true', help='If set, bypass removal of first and last elements in sequences')
    args = parser.parse_args()

    main(args.input_file, args.output_dir, test=args.Test)