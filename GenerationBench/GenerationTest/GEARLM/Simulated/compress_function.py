from gettext import lngettext
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from scipy.sparse import lil_matrix
import ast
import os

import sys
sys.path.append('/teamspace/studios/this_studio/GEAR-E/GenerationBench/GenerationTest/GEARLM/Simulated')

from expander import generate_bipartite_adj_matrix_directly as generate_expander
import math


def fake_groupwise_token_asymmetric_quantization( ####
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ).float()
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2**quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_channel_asymmetric_quantization_new(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    original_seq_len = seq_len  # Store original length
    dtype = input.dtype
    # group_size = 128

    # Ensure valid group_size
    # group_size = min(group_size, seq_len)

    # # Ensure seq_len is divisible by group_size
    # pad_size = (group_size - seq_len % group_size) % group_size
    # if pad_size > 0:
    #     input = F.pad(input, (0, 0, 0, pad_size), "constant", 0)
    #     seq_len += pad_size

    input = (input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head))
    
    # group_num = seq_len // group_size  # Compute group_num correctly

    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size

    fixed_input = input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
    
    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)

    # Trim the padded part if any
    # if pad_size > 0:
    #     dequantized_input = dequantized_input[:, :, :original_seq_len, :]

    return dequantized_input

def fake_poweriteration_group(input: torch.Tensor, loop, rank, device, p_base, q_base):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    dtype = input.dtype
    batch, dim1, dim2, dim3 = input.shape

    input = input.float()
    if q_base is not None and p_base is not None:
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        p_base = [torch.rand(batch,dim1,dim3, rank).to(input.device)]
        q_base = [torch.rand(batch,dim1,dim2, rank).to(input.device)]
    # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 2, 3) @ q_base[0]
    input = q_base[0] @ torch.transpose(p_base[0], 2, 3)
    input = input.view(batch, dim1, dim2, dim3)

    input = input.type(dtype)

    return input, p_base, q_base

def kronecker_approximation(input, shape_A, shape_B):
    """
    Approximates each residual matrix as A ⊗ B.
    input: tensor of shape [B, H, L, D]
    shape_A: tuple (a1, a2) such that a1 * a2 = L
    shape_B: tuple (b1, b2) such that b1 * b2 = D
    """
    batch_size, n_heads, seq_len, head_dim = input.shape
    a1, a2 = shape_A;  b1, b2 = shape_B
    assert a1 * a2 == seq_len and b1 * b2 == head_dim

    # [B*H, a1, a2, b1, b2]
    E5 = input.reshape(batch_size*n_heads, a1, a2, b1, b2)
    # [B*H, a1*b1, a2*b2]
    E  = E5.permute(0,1,3,2,4).reshape(batch_size*n_heads, a1*b1, a2*b2)

    # batched SVD
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)   # -> U:(B*H,m,k), S:(B*H,k), Vh:(B*H,k,n)
    s0 = torch.sqrt(S[:, :1])                             # (B*H,1)

    # rank-1 Kronecker factors
    A_hat = (U[:, :, 0] * s0).reshape(batch_size*n_heads, a1, b1)        # (B*H,a1,b1)
    B_hat = (Vh[:, 0, :] * s0).reshape(batch_size*n_heads, a2, b2)       # (B*H,a2,b2)

    # batched Kron product
    rows, cols = a1 * a2, b1 * b2
    AB = torch.einsum('bij,bkl->bikjl', A_hat, B_hat).reshape(batch_size*n_heads, rows, cols)
    return AB.view(batch_size, n_heads, seq_len, head_dim)

def fake_groupwise_channel_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size
    fixed_length = int(group_num * group_size)
    fixed_input = input[:,:fixed_length,:]
    residual_input = input[:,fixed_length:,:]
    fixed_input = fixed_input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    scale = (mx - mn) / cluster_num
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head * sep_dim)
    concat_input = torch.cat((dequantized_input,residual_input),dim=1)
    dequantized_input = concat_input.view(batch, seq_len, num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_token_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / cluster_num
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def topk_eigs_bipartite(adjacency: torch.Tensor,
                        k: int,
                        loop: int,
                        device: torch.device):
    """
    adjacency: (B, H, N, N)  — the bipartite A for each batch & head
    k:         how many top eigenvalues to extract
    loop:      number of power-iteration passes
    device:    torch.device for any new tensors

    Returns:
      eigs: Tensor of shape (B, H, k) with the top-k eigenvalue approximations
    """
    B, H, N, _ = adjacency.shape

    # 1) Run the block‐power routine directly on the full (B,H,N,N) tensor.
    #    We assume we've modified fake_poweriteration_group so it returns
    #      (A_approx, p_list, q_list), where
    #      p_list[0].shape == (B, H, N, k)
    #      q_list[0].shape == (B, H, N, k)
    A_approx, p_list, q_list = fake_poweriteration_group(adjacency, loop, k, device, None, None)
    P = p_list[0]   # (B, H, N, k)
    Q = q_list[0]   # (B, H, N, k)

    # 2) Build the small projected matrices B_bh = Q_bh^T @ A_bh @ Q_bh
    #    Shape: (B, H, k, k)
    Bproj = torch.einsum('bhip,bhij,bhjq->bhpq', Q, adjacency, Q)
    if torch.isnan(Bproj).any() or torch.isinf(Bproj).any():
        raise RuntimeError("Bproj contains NaN or Inf!")
    
    # 3) Diagonalize each k×k block exactly (k is small, e.g. 4)
    eigs = torch.linalg.eigvalsh(Bproj)   # (B, H, k), ascending order
    return eigs

def compute_second_largest_eigval_from_mask(mask, m, n, device='cpu'):
    # Create the square bipartite adjacency matrix
    zeros_ll = torch.zeros(1, 1, m, m, device=device, dtype=torch.float32)
    zeros_dd = torch.zeros(1, 1, n, n, device=device, dtype=torch.float32)
    top = torch.cat([zeros_ll, mask], dim=1)
    bottom = torch.cat([mask.transpose(0, 1), zeros_dd], dim=1)
    adjacency = torch.cat([top, bottom], dim=0)
    # A = adjacency[0, 0]
    eigs = torch.linalg.eigvalsh(A)  # shape: (m+n,)

    # Get degree d (should be largest eigenvalue)
    d_est = eigs[-1].item()
    print(f"λ₁ (≈d): {d_est:.6f}, λ_n (≈-d): {eigs[0].item():.6f}")

    # Tolerance to match ±d
    tol = 1e-4
    # Filter out λ₁ ≈ d and λ_n ≈ -d
    eigs_filtered = [abs(val.item()) for val in eigs if not math.isclose(abs(val.item()), abs(d_est), rel_tol=tol)]

    if len(eigs_filtered) == 0:
        second_largest = 0.0
    else:
        second_largest = max(eigs_filtered)

    print(f"Second largest eigenvalue (excluding ±d): {second_largest:.6f}")
    return second_largest


def gearslkivi_channelQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # print("Shape being pruned:", input.shape)
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)

    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    quantized_output = gearlkivi_channelQ(input, quantize_bit, group_size,rank,loop)
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    input.scatter_(-1, smallest_indices, smallest_value)
    input.scatter_(-1, largest_indices, largest_value)

    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    input = input.half()
    quantized_output = quantized_output.half()

    return quantized_output


ramanujan_mask_flag = True

def find_kron_shapes(seq_len, head_dim):
    def factor_pairs(n):
        return [(i, n // i) for i in range(1, int(n**0.5)+1) if n % i == 0]
    seq_factors = factor_pairs(seq_len)
    dim_factors = factor_pairs(head_dim)
    # Select the pair closest to a square
    a1, a2 = min(seq_factors, key=lambda x: abs(x[0] - x[1]))
    b1, b2 = min(dim_factors, key=lambda x: abs(x[0] - x[1]))
    return (a1, a2), (b1, b2)

def gearslkivi_tokenQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    cloned_input = input.clone()
    # output = gears_tokenQ(input, quantize_bit, group_size, sparsity)
    output = gears_tokenQ_mask(ramanujan_mask_flag, input, quantize_bit, group_size, sparsity)

    error = cloned_input - output
    # error_lr, _, _ = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    B, H, L, D = error.shape  # batch, num_heads, seq_len, head_dim
    shape_A, shape_B = find_kron_shapes(L, D)
    error_kron = kronecker_approximation(error, shape_A, shape_B)
    return output + error_kron # error_lr

def gearslkivi_channelQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1): ####
    input = input.float()
    cloned_input = input.clone()
    # output = gears_channelQ(input, quantize_bit, group_size, sparsity)
    output = gears_channelQ_mask(ramanujan_mask_flag, input, quantize_bit, group_size, sparsity)

    error = cloned_input - output
    # error_lr, _, _ = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    B, H, L, D = error.shape  # batch, num_heads, seq_len, head_dim
    shape_A, shape_B = find_kron_shapes(L, D)
    error_kron = kronecker_approximation(error, shape_A, shape_B)
    return output + error_kron # error_lr

def gearslkivi_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()

    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    input = input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # print("Shape being pruned:", input.shape)
    
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3) 
    quantized_output = gearlkivi_tokenQ(input, quantize_bit, group_size,rank,loop)
    # Restore the original values at the smallest and largest k indices
    quantized_output = quantized_output = (
        quantized_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    quantized_output.scatter_(-1, smallest_indices, smallest_value)
    quantized_output.scatter_(-1, largest_indices, largest_value)
    

    quantized_output = quantized_output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    quantized_output = quantized_output.half()
    return quantized_output
     
def gears_channelQ(input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # print("KEY, shape being pruned:", output.shape)

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    
    # mask2d = torch.zeros_like(output, dtype=torch.float32)
    # mask2d.scatter_(-1, smallest_indices, 1.0)
    # mask2d.scatter_(-1, largest_indices,   1.0)d
    # print("Mask 2d shape: ", mask2d.shape) # shape = (3, 1024, 704/64)
    # mask4d = mask2d.reshape(batch, num_head, sep_dim, seq_len).permute(0,1,3,2)
    # zeros_ll = torch.zeros(batch, num_head, seq_len, seq_len, device=mask4d.device, dtype=mask4d.dtype)
    # zeros_dd = torch.zeros(batch, num_head, sep_dim, sep_dim, device=mask4d.device, dtype=mask4d.dtype)
    # top = torch.cat([zeros_ll, mask4d], dim=3)
    # bottom = torch.cat([mask4d.transpose(2, 3), zeros_dd], dim=3)
    # adjacency = torch.cat([top, bottom], dim=2)
    # # print("Adjacency matrix shape: ", adjacency.shape)
    # # adjacency = adjacency.view(batch, num_head, sep_dim+seq_len, sep_dim+seq_len)
    # k=6 # k should be even
    # eigs = topk_eigs_bipartite(adjacency, k, loop=10, device=adjacency.device)[..., k//2:].flip(-1) # Arrange in descending order
    # gaps = eigs[..., :k//2-1] - eigs[..., 1:k//2]
    # # print("Eigen values shape: ", eigs.shape)
    # # print("Eigen values: ", eigs)
    # # print("Gaps: ", gaps)
    # key_gaps = open("spectral_gaps_key.txt", 'a')
    # key_eg = open("eigenvalues_key.txt", 'a')
    # key_eg.write(str(eigs))
    # key_gaps.write(str(gaps))
    # key_gaps.close()
    # key_eg.close()

    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = fake_groupwise_channel_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = output.half()
    return output

def compute_mask(m, n, sparsity, device='cpu'):
    # print(f"Required: m={m}, n={n}")
    mask = torch.zeros((m, n), dtype=torch.bool, device=device)

    flag = 0
    # if n > m: 
    if m != 1024:
        n = n + m
        m = n - m
        n = n - m
        flag = 1
    # m will be sep_fim*num_heads, n will be seq_len 
    flag1 = 0

    try:
        file_name = str(m) + "_" + str(n) + ".txt"
        with open("/teamspace/studios/this_studio/"+file_name) as f:
            edges = ast.literal_eval(f.read())
    except:
        try:
            file_name = str(n) + "_" + str(m) + ".txt"
            with open("/teamspace/studios/this_studio/"+file_name) as f:
                edges = ast.literal_eval(f.read())
            flag1 = 1
        except:
            # sparsity = 0.02
            print(f"Compute mask, sparsity = {sparsity}")
            lcm = abs(m*n) // math.gcd(m, n)
            lcm*=round(m*n * sparsity / lcm)
            print(f"Generating new expander: m = {m}, n = {n}")
            expander = generate_expander(int(m), int(n), int(lcm/m), int(lcm/n), max_tries=500)
            if expander is not None:
                edges, _ = expander
            else:
                print("Error: edges is None.")
                return None
            edges = [(int(u), int(v-m)) for u, v in edges]
            # print("Generated: ", m, n, edges)
            file_name = str(m) + "_" + str(n) + ".txt"
            with open("/teamspace/studios/this_studio/"+file_name, 'w') as f:
                f.write(str(edges))
    # print(file_name)
    for u, v in edges:
        if flag == flag1:
            mask[u, v]=1
        else:
            mask[v, u]=1
    return mask

def compute_ramanujan_mask(m, n, sparsity, device='cpu'):
    # print(f"Required: m={m}, n={n}")
    mask = torch.zeros((m, n), dtype=torch.bool, device=device)
    flag = 0
    # if n > m: 
    if m != 1024:
        n = n + m
        m = n - m
        n = n - m
        flag = 1
    # m will be sep_fim*num_heads, n will be seq_len 
    flag1 = 0

    try:
        file_name = "ramanujan_" + str(m) + "_" + str(n) + ".txt"
        with open("/teamspace/studios/this_studio/"+file_name) as f:
            edges = ast.literal_eval(f.read())
    except:
        try:
            file_name = "ramanujan_" + str(n) + "_" + str(m) + ".txt"
            with open("/teamspace/studios/this_studio/"+file_name) as f:
                edges = ast.literal_eval(f.read())
            flag1 = 1
        except:
            # sparsity = 0.02
            print(f"Compute ramanujan_mask, sparsity = {sparsity}")
            lcm = abs(m*n) // math.gcd(m, n)
            lcm*=round(m*n * sparsity / lcm)
            if lcm == max(m, n) : lcm *= 2
            d1, d2 = lcm//m, lcm//n
            print(f"Generating ramanujan graph: m = {m}, n = {n}, d1 = {d1}, d2 = {d2}")
            bound = math.sqrt(d1 - 1) + math.sqrt(d2 - 1)
            print(f"Target ramanujan bound: {bound:.2f}")
            ramanujan_edges = None

            temp_mask = torch.zeros((m, n), dtype=torch.bool, device=device)
            
            for trial in range(50):
                print(f"Trial {trial+1}")
                result = generate_expander(m, n, d1, d2, max_tries=100)
                if result is None:
                    continue
                edges, _ = result
                edges = [(int(u), int(v - m)) for u, v in edges]

                for u, v in edges:
                    if flag == flag1:
                        temp_mask[u, v]=1
                    else:
                        temp_mask[v, u]=1
                eig2 = compute_second_largest_eigval_from_mask(temp_mask, m, n, device=device)

                print(f"Second largest eigenvalue: {eig2:.5f}")

                if eig2 <= bound:
                    ramanujan_edges = edges
                    print(f"✅ Ramanujan graph found for {file_name} in trial {trial+1} with λ₂ = {eig2:.5f}")
                    break  # early stop  
            
            if ramanujan_edges is not None:
                print(f"Writing Ramanujan graph to {file_name}")
                edges = ramanujan_edges
                with open("/teamspace/studios/this_studio/"+file_name, 'w') as out_file:
                    out_file.write(str(ramanujan_edges))
            else: print(f"Failed to generate {file_name}")
    for u, v in edges:
        if flag == flag1:
            mask[u, v]=1
        else:
            mask[v, u]=1
    return mask

def gears_channelQ_mask(ramanujan_mask_flag, input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    if ramanujan_mask_flag:
        mask = compute_ramanujan_mask(sep_dim * num_head, seq_len, sparsity, device=output.device)
    else:
        mask = compute_mask(sep_dim * num_head, seq_len, sparsity, device=output.device)
    
    # === NEW LOGIC STARTS HERE ===
    if mask is not None and sparsity!=0.0:
        expanded_mask = mask.unsqueeze(0).expand(batch, -1, -1)  # Now shape: (batch, channels, seq_len)
        average = output.mean(dim=-1, keepdim=True)
        expanded_average = average.expand_as(output)
        # Save the original outlier values
        outlier_values = torch.where(expanded_mask.bool(), output, torch.tensor(0.0, device=output.device))
        # It selects values from output where expanded_mask is True, and sets all other values to 0.0. The result is stored in outlier_values.
        # Replace outliers with mean
        output = torch.where(expanded_mask.bool(), expanded_average, output)    
  
    # === NEW LOGIC ENDS HERE ===
    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = fake_groupwise_channel_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    
    # === RESTORE ORIGINAL VALUES USING MASK ===
    if mask is not None and sparsity!=0.0:
        output = torch.where(expanded_mask.bool(), outlier_values, output)

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = output.half()
    return output

def gears_channelQ_mask_mag(ramanujan_mask_flag, input, quantize_bit, group_size=128, sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )

    expander_sparsity = 0.03125
    sparsity-=expander_sparsity

    if ramanujan_mask_flag:
        mask = compute_ramanujan_mask(sep_dim * num_head, seq_len, expander_sparsity, device=output.device)
    else:
        mask = compute_mask(sep_dim * num_head, seq_len, expander_sparsity, device=output.device)

    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    
    # === NEW LOGIC STARTS HERE ===
    if mask is not None and sparsity!=0.0:
        expanded_mask = mask.unsqueeze(0).expand(batch, -1, -1)  # Now shape: (batch, channels, seq_len)
        # Save the original outlier values
        outlier_values = torch.where(expanded_mask.bool(), output, torch.tensor(0.0, device=output.device))
        # It selects values from output where expanded_mask is True, and sets all other values to 0.0. The result is stored in outlier_values.
        # Replace outliers with mean
        output = torch.where(expanded_mask.bool(), expanded_average, output)

    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)

    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
  
    # === NEW LOGIC ENDS HERE ===
    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = fake_groupwise_channel_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    
    # === RESTORE ORIGINAL VALUES USING MASK ===
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)

    if mask is not None and sparsity!=0.0:
        output = torch.where(expanded_mask.bool(), outlier_values, output)
    
    # print(f"Magnitude based pruning, smallest value: {smallest_value} \n\n largest value: {largest_value}")
    # nonzero = outlier_values[outlier_values != 0]
    # print(f"Non-zero outlier values ({nonzero.numel()}):\n{nonzero}")

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = output.half()
    return output

def gears_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()

    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
   
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)

    # mask2d = torch.zeros_like(output, dtype=torch.float32)
    # mask2d.scatter_(-1, smallest_indices, 1.0)
    # mask2d.scatter_(-1, largest_indices,   1.0)
    # print("Mask 2d shape: ", mask2d.shape) # shape = (3, 1024, 704/64)
    # mask4d = mask2d.reshape(batch, num_head, sep_dim, seq_len).permute(0,1,3,2)
    # zeros_ll = torch.zeros(batch, num_head, seq_len, seq_len, device=mask4d.device, dtype=mask4d.dtype)
    # zeros_dd = torch.zeros(batch, num_head, sep_dim, sep_dim, device=mask4d.device, dtype=mask4d.dtype)
    # top = torch.cat([zeros_ll, mask4d], dim=3)
    # bottom = torch.cat([mask4d.transpose(2, 3), zeros_dd], dim=3)
    # adjacency = torch.cat([top, bottom], dim=2)
    # # print("Adjacency matrix shape: ", adjacency.shape)
    # # adjacency = adjacency.view(batch, num_head, sep_dim+seq_len, sep_dim+seq_len)
    # k=6 # k should be even
    # eigs = topk_eigs_bipartite(adjacency, k, loop=10, device=adjacency.device)[..., k//2:].flip(-1) # Arrange in descending order
    # gaps = eigs[..., :k//2-1] - eigs[..., 1:k//2]
    # # print("Eigen values shape: ", eigs.shape)
    # # print("Eigen values: ", eigs)
    # # print("Gaps: ", gaps)
    # value_gaps = open("spectral_gaps_value.txt", 'a')
    # value_eg = open("eigenvalues_value.txt", 'a')
    # value_eg.write(str(eigs))
    # value_gaps.write(str(gaps))
    # value_eg.close()
    # value_gaps.close()

    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = fake_groupwise_token_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
    
    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()
    return output

def gears_tokenQ_mask(ramanujan_mask_flag, input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
       
    if ramanujan_mask_flag:
        mask = compute_ramanujan_mask(seq_len, sep_dim * num_head, sparsity, device=output.device)
    else:
        mask = compute_mask(seq_len, sep_dim * num_head, sparsity, device=output.device)

    # === NEW LOGIC STARTS HERE ===
    if mask is not None and sparsity!=0.0:
        expanded_mask = mask.unsqueeze(0).expand(batch, -1, -1)  # Now shape: (batch, channels, seq_len)
        average = output.mean(dim=-1, keepdim=True)
        expanded_average = average.expand_as(output)
        # Save the original outlier values
        outlier_values = torch.where(expanded_mask.bool(), output, torch.tensor(0.0, device=output.device))
        # Replace outliers with mean
        output = torch.where(expanded_mask.bool(), expanded_average, output)
    # === NEW LOGIC ENDS HERE ===

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = fake_groupwise_token_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    
    # === RESTORE ORIGINAL VALUES USING MASK ===
    if mask is not None and sparsity!=0.0:
        output = torch.where(expanded_mask.bool(), outlier_values, output)

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()
    return output

def gears_tokenQ_mask_mag(ramanujan_mask_flag, input, quantize_bit, group_size=128,sparsity=0.0):
    output = input.float()
    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    expander_sparsity = 0.03125
    sparsity-=expander_sparsity
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
       
    if ramanujan_mask_flag:
        mask = compute_ramanujan_mask(seq_len, sep_dim * num_head, expander_sparsity, device=output.device)
    else:
        mask = compute_mask(seq_len, sep_dim * num_head, expander_sparsity, device=output.device)
   
    # === NEW LOGIC STARTS HERE ===
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)

    if mask is not None and sparsity!=0.0:
        expanded_mask = mask.unsqueeze(0).expand(batch, -1, -1)  # Now shape: (batch, channels, seq_len)
        # Save the original outlier values
        outlier_values = torch.where(expanded_mask.bool(), output, torch.tensor(0.0, device=output.device))
        # Replace outliers with mean
        output = torch.where(expanded_mask.bool(), expanded_average, output)

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)

    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))

    # === NEW LOGIC ENDS HERE ===

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = fake_groupwise_token_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size)
    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    
    # === RESTORE ORIGINAL VALUES USING MASK ===
    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)

    # print(f"Magnitude based pruning, smallest value: {smallest_value} \n\n largest value: {largest_value}")
    # nonzero = outlier_values[outlier_values != 0]
    # print(f"Outlier values shape: {outlier_values.shape}")
    # print(f"Non-zero outlier values array size: {nonzero.shape}")
    # print(f"Non-zero outlier values ({nonzero.numel()}):\n{nonzero}")

    # print(f"Expander based pruning, outlier values: {outlier_values}")
    
    if mask is not None and sparsity!=0.0:
        output = torch.where(expanded_mask.bool(), outlier_values, output)

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()
    return output


def tokenwise_gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = cloned_input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr, _, _ = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,

                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr

def gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr, _, _ = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr
def gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr, _, _ = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr
def tokenwise_gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = cloned_input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr, _, _ = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
 
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr


def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    layer_idx,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    prefill=None,
):
    batch, num_head, seq_len, sep_dim = previous_key.shape

    if compress_config.token_preserving[layer_idx] == True:
        starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)
    
    if compress_config.compress_method[layer_idx] == "KCVT":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            seq_len,
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                int(num_head * sep_dim),
            )

    if compress_config.compress_method[layer_idx] == "KIVI_V2":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
        previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
            previous_value[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )

    if compress_config.compress_method[layer_idx] == "GEAR":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEAR-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEARL":

        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rank_used,
            compress_config.loop[layer_idx],

            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx],
 
        )
    if compress_config.compress_method[layer_idx] == "GEARL-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            rank_used,
            compress_config.loop[layer_idx],
            
            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            rankv_used,
            compress_config.loop[layer_idx],
            
        )

    return previous_key, previous_value




