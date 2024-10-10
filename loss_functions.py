import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def compute_bce_pairwise_distances(tensor1, tensor2):
    """
    Compute pairwise Binary Cross-Entropy distances between rows of two tensors.
    
    Args:
    - tensor1: Tensor of shape [c, d]
    - tensor2: Tensor of shape [e, d]

    Returns:
    - pairwise_distances: Tensor of shape [c, e] representing BCE loss between each pair of rows.
    """
    # Ensure tensors are dense and converted to float before computing BCE
    if tensor1.is_sparse:
        tensor1 = tensor1.to_dense()
    if tensor2.is_sparse:
        tensor2 = tensor2.to_dense()

    # Convert to float type to avoid dtype mismatch with binary_cross_entropy
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    c = tensor1.shape[0]
    e = tensor2.shape[0]

    if c == 0 or e == 0:
        return torch.zeros((c, e), device=tensor1.device)  # Return a zero matrix if either tensor is empty

    # Initialize a tensor to store pairwise BCE distances
    pairwise_distances = torch.zeros(c, e, device=tensor1.device)

    # Compute BCE loss for each pair of rows
    for i in range(c):
        for j in range(e):
            row1 = tensor1[i]
            row2 = tensor2[j]
            # Compute BCE between the two rows
            pairwise_distances[i, j] = F.binary_cross_entropy(row1, row2)
            print(F.binary_cross_entropy(row1, row2))


    return pairwise_distances

def pad_with_zeros(tensor, target_rows):
    """
    Pad the given tensor with zero rows to ensure it has 'target_rows' number of rows.
    """
    # If tensor is sparse, convert it to dense
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    
    current_rows = tensor.size(0)
    if current_rows < target_rows:
        # Calculate how many rows to pad
        padding_rows = target_rows - current_rows
        # Create zero rows and concatenate to the original tensor
        zero_rows = torch.zeros((padding_rows, tensor.size(1)), device=tensor.device)
        padded_tensor = torch.cat([tensor, zero_rows], dim=0)
        return padded_tensor
    return tensor


def row_wise_permutation_invariant_loss_batched(tensor1, tensor2, batch_size=100):
    """
    Compute row-wise permutation-invariant loss between two tensors in batches.
    
    Args:
    - tensor1: Tensor of shape [c, d].
    - tensor2: Tensor of shape [c, d].
    - batch_size: The size of the batches for processing.

    Returns:
    - loss: The computed permutation-invariant loss.
    """
    total_loss = 0.0
    n_row1 = tensor1.size(0)
    n_row2 = tensor2.size(0)

    # Handle cases where either tensor is empty
    if n_row1 == 0 or n_row2 == 0:
        return total_loss  # Return zero loss if either tensor is empty

    # Ensure both tensors are dense before slicing
    if tensor1.is_sparse:
        tensor1 = tensor1.to_dense()
    if tensor2.is_sparse:
        tensor2 = tensor2.to_dense()

    # Convert to float to avoid dtype mismatch with binary_cross_entropy
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    # Determine the number of batches
    num_batches = (n_row1 + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx_tensor1 = min((i + 1) * batch_size, n_row1)
        end_idx_tensor2 = min((i + 1) * batch_size, n_row2)

        # Batch-wise processing
        batch_tensor1 = tensor1[start_idx:end_idx_tensor1]  # Slicing dense tensor
        batch_tensor2 = tensor2[start_idx:end_idx_tensor2]  # Slicing dense tensor
        
        # Ensure both batches have the same size by trimming the larger batch
        min_batch_size = min(batch_tensor1.size(0), batch_tensor2.size(0))
        batch_tensor1 = batch_tensor1[:min_batch_size]
        batch_tensor2 = batch_tensor2[:min_batch_size]
        
        # Compute Binary Cross-Entropy loss for the current batch
        batch_loss = F.binary_cross_entropy(batch_tensor1, batch_tensor2)
        total_loss += batch_loss

    return total_loss / num_batches

def first_row_bce_loss(tensor1, tensor2):
    """
    Compute Binary Cross-Entropy loss for just the first row of two tensors.

    Args:
    - tensor1: Tensor of shape [c, d].
    - tensor2: Tensor of shape [c, d].

    Returns:
    - loss: Binary Cross-Entropy loss for the first row.
    """
    # Handle cases where either tensor is empty
    if tensor1.size(0) == 0 or tensor2.size(0) == 0:
        return 0.0  # Return zero loss if either tensor is empty

    # Ensure both tensors are dense and of type float
    if tensor1.is_sparse:
        tensor1 = tensor1.to_dense()
    if tensor2.is_sparse:
        tensor2 = tensor2.to_dense()

    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    # Select the first row from each tensor
    first_row_tensor1 = tensor1[0]
    first_row_tensor2 = tensor2[0]

    # Compute the Binary Cross-Entropy loss for the first row
    loss = F.binary_cross_entropy(first_row_tensor1, first_row_tensor2)

    return loss


def sinkhorn_cosine_loss(matrix_a, matrix_b, epsilon=0.1, num_iters=50):
    """
    Compute the Sinkhorn Distance between two matrices using Cosine Distance.

    Args:
    - matrix_a: Tensor of shape [N_A, D], positive real entries (many zeros)
    - matrix_b: Tensor of shape [N_B, D], binary entries
    - epsilon: Entropy regularization coefficient
    - num_iters: Number of iterations in the Sinkhorn algorithm

    Returns:
    - loss: The computed Sinkhorn Distance
    """
    N_A, D = matrix_a.size()
    N_B, _ = matrix_b.size()

    # Handle cases where one of the matrices is empty
    if N_A == 0 and N_B > 0:
        matrix_a = torch.zeros(N_B, D, device=matrix_b.device)
        N_A = N_B  # Update N_A to match N_B for proper loss calculation
    elif N_B == 0 and N_A > 0:
        matrix_b = torch.zeros(N_A, D, device=matrix_a.device)
        N_B = N_A  # Update N_B to match N_A for proper loss calculation
    elif N_A == 0 and N_B == 0:
        return 0.0  # Both matrices are empty, return zero loss

    # Ensure tensors are dense and float
    if matrix_a.is_sparse:
        matrix_a = matrix_a.to_dense()
    if matrix_b.is_sparse:
        matrix_b = matrix_b.to_dense()
    
    matrix_a = matrix_a.float()
    matrix_b = matrix_b.float()

    # Normalize rows to unit vectors to compute cosine similarity
    matrix_a_norm = F.normalize(matrix_a, p=2, dim=1)
    matrix_b_norm = F.normalize(matrix_b, p=2, dim=1)

    # Compute cosine similarity matrix (N_A x N_B)
    cosine_similarity = torch.mm(matrix_a_norm, matrix_b_norm.t())

    # Compute cosine distance matrix
    C = 1 - cosine_similarity  # Cosine distance ranges from 0 to 2

    # Ensure cost matrix is non-negative
    C = torch.clamp(C, min=0)

    # Marginal distributions (uniform)
    mu = torch.full((N_A,), 1.0 / N_A, device=matrix_a.device)
    nu = torch.full((N_B,), 1.0 / N_B, device=matrix_b.device)

    # Sinkhorn algorithm
    K = torch.exp(-C / epsilon)  # Kernel matrix
    u = torch.ones_like(mu)  # Scaling vector for matrix_a
    v = torch.ones_like(nu)  # Scaling vector for matrix_b

    for _ in range(num_iters):
        u = mu / (K @ v)
        v = nu / (K.t() @ u)
    
    # Compute transport plan pi
    pi = torch.diag(u) @ K @ torch.diag(v)

    # Compute Sinkhorn distance
    loss = torch.sum(pi * C)

    return loss

import torch
import torch.nn.functional as F

def sinkhorn_row_wise_permutation_invariant_loss(tensor1, tensor2, epsilon=0.1, num_iters=50):
    n_row1 = tensor1.size(0)
    n_row2 = tensor2.size(0)

    # Handle cases where one of the matrices is empty
    if n_row1 == 0 and n_row2 > 0:
        tensor1 = torch.zeros(n_row2, tensor1.size(1), device=tensor2.device)
        n_row1 = n_row2
    elif n_row2 == 0 and n_row1 > 0:
        tensor2 = torch.zeros(n_row1, tensor2.size(1), device=tensor1.device)
        n_row2 = n_row1
    elif n_row1 == 0 and n_row2 == 0:
        return torch.tensor(0.0, device=tensor1.device)  # Both matrices are empty, return zero loss

    # Ensure tensors are dense and float
    if tensor1.is_sparse:
        tensor1 = tensor1.to_dense().float()
    else:
        tensor1 = tensor1.float()
    
    if tensor2.is_sparse:
        tensor2 = tensor2.to_dense().float()
    else:
        tensor2 = tensor2.float()

    # Compute pairwise BCE loss
    tensor1_expanded = tensor1.unsqueeze(1).expand(-1, n_row2, -1)  # Shape: [n_row1, n_row2, d]
    tensor2_expanded = tensor2.unsqueeze(0).expand(n_row1, -1, -1)  # Shape: [n_row1, n_row2, d]
    
    # Vectorized pairwise BCE computation
    pairwise_bce_distances = F.binary_cross_entropy(tensor1_expanded, tensor2_expanded, reduction='none').mean(dim=2)

    # Ensure cost matrix is non-negative
    C = torch.clamp(pairwise_bce_distances, min=0)

    # Marginal distributions (uniform)
    mu = torch.full((n_row1,), 1.0 / n_row1, device=tensor1.device)
    nu = torch.full((n_row2,), 1.0 / n_row2, device=tensor2.device)

    # Sinkhorn algorithm
    K = torch.exp(-C / epsilon)  # Kernel matrix
    u = torch.ones_like(mu)  # Scaling vector for tensor1
    v = torch.ones_like(nu)  # Scaling vector for tensor2

    for _ in range(num_iters):
        u = mu / (K @ v)
        v = nu / (K.t() @ u)
        
        # Avoid division by zero by clamping u and v
        u = torch.clamp(u, min=1e-8)
        v = torch.clamp(v, min=1e-8)

    # Compute transport plan pi
    pi = torch.diag(u) @ K @ torch.diag(v)

    # Compute Sinkhorn distance (BCE-based)
    loss = torch.sum(pi * C)

    # Handle potential nan in loss due to numerical issues
    if torch.isnan(loss):
        loss = torch.tensor(float('inf'), device=loss.device)

    return loss



def row_wise_permutation_invariant_loss(tensor1, tensor2):
    """
    Compute pairwise Binary Cross-Entropy distances between rows of two tensors using the Hungarian algorithm.
    
    Args:
    - tensor1: Tensor of shape [n_row1, d]
    - tensor2: Tensor of shape [n_row2, d]

    Returns:
    - loss: The optimal row-wise permutation-invariant loss using the Hungarian algorithm.
    """
    n_row1 = tensor1.size(0)
    n_row2 = tensor2.size(0)

    # Handle cases where one of the matrices is empty
    if n_row1 == 0 and n_row2 > 0:
        tensor1 = torch.zeros(n_row2, tensor1.size(1), device=tensor2.device)
        n_row1 = n_row2
    elif n_row2 == 0 and n_row1 > 0:
        tensor2 = torch.zeros(n_row1, tensor2.size(1), device=tensor1.device)
        n_row2 = n_row1
    elif n_row1 == 0 and n_row2 == 0:
        return 0.0  # Both matrices are empty, return zero loss

    # Ensure tensors are dense and float
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    # Compute pairwise BCE loss
    tensor1_expanded = tensor1.unsqueeze(1).expand(-1, n_row2, -1)  # Shape: [n_row1, n_row2, d]
    tensor2_expanded = tensor2.unsqueeze(0).expand(n_row1, -1, -1)  # Shape: [n_row1, n_row2, d]
    
    # Vectorized pairwise BCE computation
    pairwise_bce_distances = F.binary_cross_entropy(tensor1_expanded, tensor2_expanded, reduction='none').mean(dim=2)

    # Apply the Hungarian algorithm to find the optimal row matching
    row_idx, col_idx = linear_sum_assignment(pairwise_bce_distances.cpu().detach().numpy())

    # Select the optimal rows according to the Hungarian algorithm
    matched_rows1 = tensor1[row_idx]
    matched_rows2 = tensor2[col_idx]

    # Compute Binary Cross-Entropy loss for the matched rows
    loss = F.binary_cross_entropy(matched_rows1, matched_rows2).mean()

    return loss

def hungarian_cosine_loss(matrix_a, matrix_b):
    """
    Compute the permutation-invariant loss between two matrices using Cosine Distance
    and the Hungarian algorithm.

    Args:
    - matrix_a: Tensor of shape [N_A, D], positive real entries (many zeros)
    - matrix_b: Tensor of shape [N_B, D], binary entries

    Returns:
    - loss: The computed permutation-invariant Cosine Distance
    """
    N_A, D = matrix_a.size()
    N_B, _ = matrix_b.size()

    # Handle cases where one of the matrices is empty
    if N_A == 0 and N_B > 0:
        matrix_a = torch.zeros(N_B, D, device=matrix_b.device)
        N_A = N_B  # Update N_A to match N_B for proper loss calculation
    elif N_B == 0 and N_A > 0:
        matrix_b = torch.zeros(N_A, D, device=matrix_a.device)
        N_B = N_A  # Update N_B to match N_A for proper loss calculation
    elif N_A == 0 and N_B == 0:
        return 0.0  # Both matrices are empty, return zero loss

    # Ensure tensors are dense and float
    if matrix_a.is_sparse:
        matrix_a = matrix_a.to_dense()
    if matrix_b.is_sparse:
        matrix_b = matrix_b.to_dense()
    
    matrix_a = matrix_a.float()
    matrix_b = matrix_b.float()

    # Normalize rows to unit vectors to compute cosine similarity
    matrix_a_norm = F.normalize(matrix_a, p=2, dim=1)
    matrix_b_norm = F.normalize(matrix_b, p=2, dim=1)

    # Compute cosine similarity matrix (N_A x N_B)
    cosine_similarity = torch.mm(matrix_a_norm, matrix_b_norm.t())

    # Compute cosine distance matrix
    C = 1 - cosine_similarity  # Cosine distance ranges from 0 to 2

    # Ensure cost matrix is non-negative
    C = torch.clamp(C, min=0)

    # Apply the Hungarian algorithm to find the optimal row matching
    row_indices, col_indices = linear_sum_assignment(C.cpu().detach().numpy())

    # Select the optimal rows according to the Hungarian algorithm
    matched_rows_a = matrix_a[row_indices]
    matched_rows_b = matrix_b[col_indices]

    # Compute the final loss as the average cosine distance between matched rows
    loss = F.cosine_embedding_loss(matched_rows_a, matched_rows_b, torch.ones(len(matched_rows_a)).to(matrix_a.device))

    return loss