import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import random

def generate_bipartite_adj_matrix_directly(M, N, m, n, max_tries=100):
    """
    Placeholder for a function that attempts to generate a bi-regular
    bipartite graph and returns its adjacency matrix as a SciPy sparse matrix,
    or None on failure.

    This simplified placeholder uses NetworkX internally for demonstration
    and then converts to a sparse matrix. A more optimized version would
    construct the sparse matrix components (data, indices, indptr) directly.
    """
    if M * m != N * n:
        print("Error: Condition mM = nN must be satisfied for bi-regularity.")
        return None
    if m > N or n > M:
        print("Error: Degrees cannot be greater than the number of nodes in the other partition.")
        return None

    # Node labels for NetworkX (0 to M-1 for U, M to M+N-1 for V)
    u_nodes = list(range(M))
    v_nodes = list(range(M, M + N))

    for attempt in range(max_tries):
        G = nx.Graph()


        # Create stubs for matching
        u_stubs = []
        for u_node_idx in u_nodes:
            u_stubs.extend([u_node_idx] * m)

        v_stubs = []
        for v_node_idx in v_nodes:
            v_stubs.extend([v_node_idx] * n)

        random.shuffle(u_stubs)
        random.shuffle(v_stubs)

        edges_to_add = []
        temp_v_stubs = list(v_stubs) # Work with a copy
        possible_to_form = True

        # Attempt to match stubs
        # This is a simplified matching. A full configuration model is more robust
        # but can produce multi-edges, which may or may not be desired.
        # For a simple graph, more care is needed.
        current_edges_set = set() # To help avoid parallel edges in this simple model

        for u_stub_node in u_stubs:
            matched_this_stub = False
            random.shuffle(temp_v_stubs) # Randomize choice from v_stubs
            for i in range(len(temp_v_stubs) -1, -1, -1): # Iterate backwards for safe removal
                v_stub_node = temp_v_stubs[i]
                # Avoid parallel edges for a simple graph interpretation
                if tuple(sorted((u_stub_node, v_stub_node))) not in current_edges_set:
                    edge = tuple(sorted((u_stub_node, v_stub_node)))
                    edges_to_add.append(edge)
                    current_edges_set.add(edge)
                    temp_v_stubs.pop(i)
                    matched_this_stub = True
                    break
            if not matched_this_stub:
                possible_to_form = False # Could not find a valid match for a u_stub
                break
       
        if not possible_to_form or len(edges_to_add) != M * m:
            # print(f"Attempt {attempt + 1}: Failed to form all edges ({len(edges_to_add)} out of {M*m}). Retrying.")
            continue

        # Create the graph from the successfully matched edges
        # All nodes must be present for to_scipy_sparse_array to use nodelist correctly
        final_graph = nx.Graph()
        all_graph_nodes = u_nodes + v_nodes
        final_graph.add_nodes_from(all_graph_nodes)
        final_graph.add_edges_from(edges_to_add)

        # Validate degrees and connectivity
        degrees_U_ok = all(final_graph.degree(u) == m for u in u_nodes)
        degrees_V_ok = all(final_graph.degree(v) == n for v in v_nodes)
       
        if degrees_U_ok and degrees_V_ok and nx.is_connected(final_graph):
            # Ensure nodes are ordered for consistent matrix: U followed by V
            node_order = u_nodes + v_nodes
            edges = list(final_graph.edges())
            adj_matrix_sparse = nx.to_scipy_sparse_array(final_graph, nodelist=node_order, format='csr')
            print(f"Placeholder: Successfully generated a valid graph structure and its sparse matrix in attempt {attempt + 1}.")
            return edges, adj_matrix_sparse
        # else:
            # print(f"Attempt {attempt+1}: Generated graph not biregular or not connected. Degrees U ok: {degrees_U_ok}, Degrees V ok: {degrees_V_ok}, Connected: {nx.is_connected(final_graph) if degrees_U_ok and degrees_V_ok else 'N/A'}")


    print(f"Placeholder: Failed to generate a valid sparse adjacency matrix after {max_tries} attempts.")
    return None


def get_adjacency_matrix_eigenvalues_from_sparse(adj_matrix_sparse, num_eigenvalues_to_compute=None):
    """
    Calculates eigenvalues of a sparse adjacency matrix.
    If num_eigenvalues_to_compute is None or equals total nodes, computes all eigenvalues.
    Otherwise, computes a subset using sparse methods.
    """
    if adj_matrix_sparse is None or adj_matrix_sparse.shape[0] == 0:
        return np.array([])

    N_total = adj_matrix_sparse.shape[0]

    if num_eigenvalues_to_compute is None or num_eigenvalues_to_compute >= N_total:
        # Compute all eigenvalues by converting to dense
        # Adjacency matrix of an undirected graph is symmetric.
        print(f"Calculating all {N_total} eigenvalues (converting sparse to dense)...")
        try:
            # .A is a shorthand for .toarray()
            eigenvalues = np.linalg.eigvalsh(adj_matrix_sparse.toarray())
        except np.linalg.LinAlgError: # Fallback if not perfectly symmetric
            eigenvalues = np.linalg.eigvals(adj_matrix_sparse.toarray())
        return np.sort(eigenvalues)
    else:
        # Compute a subset of k eigenvalues using sparse eigsh
        # k must be less than N_total-1 for eigsh typically.
        k = min(num_eigenvalues_to_compute, N_total - 2 if N_total > 1 else 0)
        if k <= 0:
             print("Warning: Not enough nodes to compute a subset of eigenvalues with sparse methods, trying dense for all.")
             return np.sort(np.linalg.eigvalsh(adj_matrix_sparse.toarray()))


        print(f"Calculating {k} eigenvalues with largest magnitude using sparse methods...")
        try:
            # 'LM' for largest magnitude. For bipartite, spectrum is symmetric.
            # eigvalsh returns sorted eigenvalues.
            eigenvalues = scipy.sparse.linalg.eigsh(adj_matrix_sparse, k=k, which='LM', return_eigenvectors=False)
            return np.sort(eigenvalues)
        except Exception as e:
            print(f"Sparse eigenvalue calculation failed: {e}. Falling back to dense calculation for all eigenvalues.")
            try:
                eigenvalues_dense = np.linalg.eigvalsh(adj_matrix_sparse.toarray())
            except np.linalg.LinAlgError:
                eigenvalues_dense = np.linalg.eigvals(adj_matrix_sparse.toarray())
            return np.sort(eigenvalues_dense)

# --- Example Usage ---
# M = 1344  # Number of nodes in the first partition
# N = 1024  # Number of nodes in the second partition
# m_deg = 16 # Degree of nodes in the first partition (mM = 50*3 = 150)
# n_deg = 21 # Degree of nodes in the second partition (nN = 75*2 = 150)
# # Condition mM = nN is satisfied.

# edges, bipartite_adj_matrix = generate_bipartite_adj_matrix_directly(M, N, m_deg, n_deg, max_tries=500)

# edge_list = [(int(u), int(v - M)) for u, v in edges]


# # Optional: print some edges
# # print("Sample edges:", edge_list)

# print(edge_list)