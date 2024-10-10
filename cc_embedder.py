import torch
import torch.nn as nn
from topomodelx.nn.combinatorial.hmc import HMC

class CCEmbedder(nn.Module):
    def __init__(self, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, embedding_dim=256):
        super().__init__()

        # Define input, intermediate, and final channel dimensions
        in_channels = [feature_n_0_cells, feature_n_1_cells, feature_n_2_cells]  # [7, 11, 13]
        intermediate_channels = [60, 60, 60]  # Custom intermediate channels
        final_channels = [60, 256, 256]  # Final channels with embedding dimension of 256 for x_2

        # Channels per layer configuration
        channels_per_layer = [
            [in_channels, intermediate_channels, final_channels]
        ]

        self.hmc = HMC(channels_per_layer)
        self.fc_0 = nn.Linear(60, embedding_dim)  # Projecting node embeddings to 256 dimensions
        self.fc_1 = nn.Linear(60, embedding_dim)  # Projecting edge embeddings to 256 dimensions

    def to_sparse(self, tensor):
        return tensor.to_sparse()

    def forward(self, x_0, x_1, x_2, neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_0_to_1, neighborhood_1_to_2):
        # Convert neighborhood matrices to sparse format
        neighborhood_0_to_0 = self.to_sparse(neighborhood_0_to_0)
        neighborhood_1_to_1 = self.to_sparse(neighborhood_1_to_1)
        neighborhood_2_to_2 = self.to_sparse(neighborhood_2_to_2)
        neighborhood_0_to_1 = self.to_sparse(neighborhood_0_to_1)
        neighborhood_1_to_2 = self.to_sparse(neighborhood_1_to_2)

        # Apply HMC layers
        x_0, x_1, x_2 = self.hmc(
            x_0, x_1, x_2,
            neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2,
            neighborhood_0_to_1, neighborhood_1_to_2
        )
        
        # # Generate 256-dimensional embeddings for each cell
        # embeddings_0 = self.fc_0(x_0)
        # embeddings_1 = self.fc_1(x_1)

        return x_1, x_2


