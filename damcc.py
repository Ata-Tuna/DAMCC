import torch
from cc_embedder import CCEmbedder
from model import AutoregressiveSubsetSampler  # Import from model.py
from loss_functions import row_wise_permutation_invariant_loss  # Import the loss function
import sys

class Damcc(torch.nn.Module):
    def __init__(self, num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells):
        super().__init__()

        self.cc_embedder = CCEmbedder(feature_n_0_cells, feature_n_1_cells, feature_n_2_cells)

        # self.gnn0 = GAT(
        #     in_channels=n_features,
        #     hidden_channels=256,
        #     num_layers=1,
        #     dropout=0,
        #     heads=1,
        # )
        # self.gnn1 = GAT(
        #     in_channels=n_features,
        #     hidden_channels=256,
        #     num_layers=1,
        #     dropout=0,
        #     heads=1,
        # )

        self.decoder_0_cells = AutoregressiveSubsetSampler(
            num_nodes, n_features, size_g, n_features, min_nonzero=2, max_nonzero=2)
        self.decoder_1_cells = AutoregressiveSubsetSampler(
            num_nodes, n_features, size_g, n_features, min_nonzero=3, max_nonzero=5)
                # MLP to combine embeddings
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(512, 256),  # Input dimension is 512 because we concatenate two 256-dim embeddings
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256)
        # )

    # def forward_train(self,             
    #         x_0, a01, a02,
    #         b10, b20, b10_t, b20_t, num_nodes):
    #     gnn_embeds0 = self.gnn0(x_0, a01)
    #     gnn_embeds1 = self.gnn1(x_0, a02)

    #     ll_0_cells, states_0_cells = self.decoder_0_cells.forward_train(gnn_embeds0, num_nodes)
    #     ll_1_cells, states_1_cells = self.decoder_1_cells.forward_train(gnn_embeds1, num_nodes)

    #     return -1 * (ll_0_cells + ll_1_cells) / (2 * num_nodes)
# x_0, x_1, x_2, neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_0_to_1, neighborhood_1_to_2
    def forward(self,             
            x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20,
            num_nodes, 
            get_ll=False, delta_edges=None):

        # gnn_embeds0 = self.gnn0(x_0, a01)
        # # print("gnn_embed0: ", gnn_embeds0.size())
        # gnn_embeds1 = self.gnn1(x_0, a02)
        #         # Combine embeddings by concatenation
        # combined_embedding = torch.cat([gnn_embeds0, gnn_embeds1], dim=-1)  # Shape: [num_nodes, 512]

        # # Pass combined embedding through the MLP
        # final_embedding = self.mlp(combined_embedding)  # Shape: [num_nodes, 256]
        embeddings_1, embeddings_2 = self.cc_embedder(x_0, x_1, x_2, a1, a2, coa2, b1, b2)
        print(embeddings_2.size())


        sampled_1_cells = self.decoder_0_cells(embeddings_1, b10)
        sampled_2_cells = self.decoder_1_cells(embeddings_2, b20)

        return sampled_1_cells, sampled_2_cells
