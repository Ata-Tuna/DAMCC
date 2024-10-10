import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from sampler import SampleIncidenceRow as SampleRow  # Import SampleRow from sampler.py
import numpy as np

class AutoregressiveSubsetSampler(nn.Module):
    def __init__(self, n, n_features, size_g, m_features, leaf_prob=0.5, new_cell_generation_factor=2, left_prob=0.67, right_prob=0.67, min_nonzero=2, max_nonzero=6):
        super().__init__()
        self.n = n
        self.n_features = n_features
        self.size_g = size_g
        self.m_features = m_features
        self.leaf_prob = leaf_prob
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.min_nonzero = min_nonzero
        self.max_nonzero = max_nonzero
        self.new_cell_generation_factor = new_cell_generation_factor

        # MLP for processing m and g
        self.mlp_mg = nn.Sequential(
            nn.Linear(size_g + size_g, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # SampleRow module
        self.sample_row = SampleRow(256, n, size_g, self.leaf_prob, self.left_prob, self.right_prob, 
                                    min_nonzero=self.min_nonzero, max_nonzero=self.max_nonzero)

        # Transformer for processing g
        self.transformer = nn.TransformerEncoderLayer(d_model=size_g, nhead=8)

    def forward(self, gnn_embeds, b):
        num_nodes = gnn_embeds.size(0)
        # g = torch.randn(self.size_g, device=gnn_embeds.device)    
        g = torch.zeros(self.size_g, device=gnn_embeds.device)
        print("gnn_embeds", gnn_embeds.size())
        print("b", b.size())


        b = b.to_dense()

        incidence_matrix = []

        # print("generating rows first time")
        # for i in tqdm(range(b.size(0))):
        #     m = b[i]
        #     h = self.mlp_mg(torch.cat([m, g]))

        #     valid_sample = False
        #     while not valid_sample:
        #         row_sample, g_new = self.sample_row(h)
        #         nonzero_count = row_sample.nonzero().size(0)
        #         if nonzero_count == 0 or (self.min_nonzero <= nonzero_count <= self.max_nonzero):
        #             valid_sample = True
            
        #     incidence_matrix.append(row_sample)
        #     g = self.transformer(g_new.unsqueeze(0)).squeeze(0)
        
        
        # print("generating rows second time")
        # for i in tqdm(range(int(np.floor(b.size(0)*self.new_cell_generation_factor)))):
        #     random_index = torch.randint(0, b.size(0), (1,)).item()
        #     m = b[random_index]
        #     h = self.mlp_mg(torch.cat([m, g]))

        #     valid_sample = False
        #     while not valid_sample:
        #         row_sample, g_new = self.sample_row(h)
        #         nonzero_count = row_sample.nonzero().size(0)
        #         if nonzero_count == 0 or (self.min_nonzero <= nonzero_count <= self.max_nonzero):
        #             valid_sample = True
            
        #     incidence_matrix.append(row_sample)
        #     g = self.transformer(g_new.unsqueeze(0)).squeeze(0)


        for j in range(self.new_cell_generation_factor):
            print("generating rows")
            for i in tqdm(range(gnn_embeds.size(0))):
                m = gnn_embeds[i]
                print(m.size())
                h = self.mlp_mg(torch.cat([m, g]))


                valid_sample = False
                while not valid_sample:
                    row_sample, g_new = self.sample_row(h)
                    nonzero_count = row_sample.nonzero().size(0)
                    if nonzero_count == 0 or (self.min_nonzero <= nonzero_count <= self.max_nonzero):
                        
                        valid_sample = True
                
                incidence_matrix.append(row_sample)
                g = self.transformer(g_new.unsqueeze(0)).squeeze(0)

        print(incidence_matrix)



        incidence_matrix = torch.stack(incidence_matrix)

        # # Check if incidence_matrix is empty
        # if incidence_matrix:
        #     incidence_matrix = torch.stack(incidence_matrix)
        #     # incidence_matrix = incidence_matrix[~torch.all(incidence_matrix == 0, dim=1)]
        
        # # Ensure that we have at least one row in the incidence matrix
        # if incidence_matrix.size(0) == 0:
        #     # Create a fallback row of zeros with requires_grad=True
        #     fallback_row = torch.zeros(self.n, device=gnn_embeds.device, requires_grad=True)
        #     fallback_indices = torch.randperm(self.n, device=gnn_embeds.device)[:self.min_nonzero]
        #     fallback_row = fallback_row.scatter(0, fallback_indices, 1.0).unsqueeze(0)
        #     incidence_matrix = fallback_row


        print("RETURNING INCIDENCE MATRIX")
        print("incidence matrix size was: ", b.size(0))
        print("new size: ", incidence_matrix.size(0))
        return incidence_matrix
