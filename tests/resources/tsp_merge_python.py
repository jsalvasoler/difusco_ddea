import numpy as np


def merge_python(coords: np.ndarray, adj_mat: np.ndarray) -> tuple:
    # Number of nodes
    points = coords
    N = len(points)

    # Calculate the pairwise distances between nodes
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    print("Pairwise distances:")
    print(dists)

    # Initialize the adjacency matrix
    A = np.zeros((N, N))

    # Arrays to record the beginning and end of the partial route for each node
    route_begin = np.arange(N, dtype=int)
    route_end = np.arange(N, dtype=int)

    # Sort all the possible edges in decreasing order of adj_mat[i, j] / dists[i, j]
    sorted_edges = np.argsort((-adj_mat / dists).flatten()).astype(int)

    # Helper functions to find the root of the partial route
    def find_begin(route_begin: np.array, i: int) -> int:
        begin_i = route_begin[i]
        if begin_i != i:
            begin_i = find_begin(route_begin, begin_i)
            route_begin[i] = begin_i
        return begin_i

    def find_end(route_end: np.array, i: int) -> int:
        end_i = route_end[i]
        if end_i != i:
            end_i = find_end(route_end, end_i)
            route_end[i] = end_i
        return end_i

    # Merge variables
    merge_iterations = 0
    merge_count = 0

    # Process each edge
    for edge in sorted_edges:
        merge_iterations += 1
        # Calculate the corresponding nodes i and j
        i = edge // N
        j = edge % N

        print(f"Iteration {merge_iterations}: Checking edge ({i}, {j})")

        # Find the roots of the routes for i and j
        begin_i = find_begin(route_begin, i)
        end_i = find_end(route_end, i)
        begin_j = find_begin(route_begin, j)
        end_j = find_end(route_end, j)

        print(
            f"  - Route roots: begin_i={begin_i}, end_i={end_i}, begin_j={begin_j}, end_j={end_j}"
        )

        # If i and j are already connected in the same component, skip this edge
        if begin_i == begin_j:
            print("  - Nodes are already in the same connected component, skipping.")
            continue

        # If i is not at the beginning or end of its route, skip this edge
        if i not in (begin_i, end_i):
            print("  - Node i is not at the beginning or end of its route, skipping.")
            continue

        # If j is not at the beginning or end of its route, skip this edge
        if j not in (begin_j, end_j):
            print("  - Node j is not at the beginning or end of its route, skipping.")
            continue

        # Add the edge to the adjacency matrix
        A[i, j] = 1
        A[j, i] = 1
        merge_count += 1
        print(f"  - Edge ({i}, {j}) added to the adjacency matrix.")

        # Update route beginnings and endings
        if i == begin_i and j == end_j:
            route_begin[begin_i] = begin_j
            route_end[end_j] = end_i
            print(
                f"  - Merging: route_begin[{begin_i}] = {begin_j}, route_end[{end_j}] = {end_i}"
            )

        elif i == end_i and j == begin_j:
            route_begin[begin_j] = begin_i
            route_end[end_i] = end_j
            print(
                f"  - Merging: route_begin[{begin_j}] = {begin_i}, route_end[{end_i}] = {end_j}"
            )

        elif i == begin_i and j == begin_j:
            route_begin[begin_i] = end_j
            route_begin[begin_j] = end_j
            route_end[end_j] = end_i
            route_end[begin_j] = end_i
            print(
                f"  - Merging: updated route beginnings and endings for nodes {i} and {j}"
            )

        elif i == end_i and j == end_j:
            route_end[end_i] = begin_j
            route_begin[begin_j] = begin_i
            route_begin[end_j] = begin_i
            route_end[end_j] = begin_j
            route_end[begin_j] = begin_j
            print(
                f"  - Merging: updated route beginnings and endings for nodes {i} and {j}"
            )

        print(f" * Route beginnings: {route_begin}")
        print(f" * Route endings: {route_end}\n")

        # Check if we have N-1 edges to form a complete tour
        if merge_count == N - 1:
            print(f"Tour complete after {merge_iterations} iterations.")
            break

    # Connect the final two nodes to complete the tour
    final_begin = find_begin(route_begin, 0)
    final_end = find_end(route_end, 0)
    A[final_end, final_begin] = 1
    A[final_begin, final_end] = 1
    print(f"Connecting final nodes: {final_begin} and {final_end}")

    return A, merge_iterations
