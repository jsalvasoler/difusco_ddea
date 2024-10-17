import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdio cimport printf


# To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
# procedure.
# • Initialize extracted tour with an empty graph with N vertices.
# • Sort all the possible edges (i, j) in decreasing order of Aij/||vi − vjk||^2 (i.e., the inverse edge weight,
# multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
# • For each edge (i, j) in the list:
#   – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
#   – If inserting (i, j) results in a graph with cycles (of length < N), continue.
#   – Otherwise, insert (i, j) into the tour.
# • Return the extracted tour.

cpdef merge_cython(double[:,:] coords, double[:,:] adj_mat):
    cdef double[:,:] points = coords
    cdef double[:,:] dists = np.linalg.norm(np.asarray(points)[:, None] - np.asarray(points), axis=-1)

    cdef long N = dists.shape[0]

    # we initialize the real adjacency matrix
    cdef double[:,:] A = np.zeros((N, N))

    # we use an array to record the beginning and end of the partial route of each node

    cdef int[:] route_begin = np.arange(N, dtype='int32')
    cdef int[:] route_end = np.arange(N, dtype='int32')

    # we calculate the dist between each pair of nodes
    # TODO: we should remove this
    dist = np.linalg.norm(np.asarray(points)[:, None] - np.asarray(points), axis=-1)

    # we sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk
    cdef int[:] sorted_edges = np.argsort((-np.asarray(adj_mat) / dist).flatten()).astype('int32')
    cdef int i, j

    # • For each edge (i, j) in the list:
    #   – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    #   – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    #   – Otherwise, insert (i, j) into the tour.

    merge_iterations = 0
    merge_count = 0
    # we first enumerate sorted edges
    for edge in sorted_edges:
        # we calculate the corresponding i and j
        merge_iterations += 1
        i = int(edge // N)
        j = int(edge % N)

        # we check if the edge is already in the graph
        begin_i = find_begin(route_begin, i)
        end_i = find_end(route_end, i)
        begin_j = find_begin(route_begin, j)
        end_j = find_end(route_end, j)

        if begin_i == begin_j:
            continue

        if i != begin_i and i != end_i:
            continue

        if j != begin_j and j != end_j:
            continue


        A[j, i] = 1
        A[i, j] = 1
        merge_count += 1
        if i == begin_i and j == end_j:
            route_begin[begin_i] = begin_j
            route_end[end_j] = end_i

        elif i == end_i and j == begin_j:
            route_begin[begin_j] = begin_i
            route_end[end_i] = end_j

        elif i == begin_i and j == begin_j:
            route_begin[begin_i] = end_j

            route_begin[begin_j] = end_j
            route_begin[end_j] = end_j
            route_end[end_j] = end_i
            route_end[begin_j] = end_i

        elif i == end_i and j == end_j:
            route_end[end_i] = begin_j

            route_begin[begin_j] = begin_i
            route_begin[end_j] = begin_i
            route_end[end_j] = begin_j
            route_end[begin_j] = begin_j

        if merge_count == N - 1:
            break

    final_begin = find_begin(route_begin, 0)
    final_end = find_end(route_end, 0)
    A[final_end, final_begin] = 1
    A[final_begin, final_end] = 1
    return A, merge_iterations


cpdef find_begin(int[:] route_begin, int i):
    cdef int begin_i = route_begin[i]
    if begin_i != i:
        begin_i = find_begin(route_begin, begin_i)
        route_begin[i] = begin_i
    return begin_i


cpdef find_end(int[:] route_end, int i):
    cdef int end_i = route_end[i]
    if end_i != i:
        end_i = find_end(route_end, end_i)
        route_end[i] = end_i
    return end_i


    merge_iterations = 0
    merge_count = 0
    # We first enumerate sorted edges.
    for edge in sorted_edges:
        # Increment the number of merge iterations.
        merge_iterations += 1
        # Calculate the corresponding nodes i and j from the flattened edge index.
        i = int(edge // N)
        j = int(edge % N)

        # Check if the edge is already in the graph.
        # Find the beginning and end of the partial route for nodes i and j.
        begin_i = find_begin(route_begin, i)
        end_i = find_end(route_end, i)
        begin_j = find_begin(route_begin, j)
        end_j = find_end(route_end, j)

        # If i and j belong to the same connected component (cycle), skip this edge.
        if begin_i == begin_j:
            continue

        # If i is not at the beginning or end of its route, skip this edge.
        if i != begin_i and i != end_i:
            continue

        # If j is not at the beginning or end of its route, skip this edge.
        if j != begin_j and j != end_j:
            continue

        # Add the edge (i, j) to the adjacency matrix, making it part of the tour.
        A[j, i] = 1
        A[i, j] = 1
        # Increment the merge count.
        merge_count += 1

        # Update the route beginnings and endings depending on the positions of i and j.
        # If i is at the beginning and j is at the end, update route to merge them.
        if i == begin_i and j == end_j:
            route_begin[begin_i] = begin_j
            route_end[end_j] = end_i

        # If i is at the end and j is at the beginning, update route to merge them.
        elif i == end_i and j == begin_j:
            route_begin[begin_j] = begin_i
            route_end[end_i] = end_j

        # If both i and j are at the beginning of their routes, merge them.
        elif i == begin_i and j == begin_j:
            route_begin[begin_i] = end_j

            # Update all the route beginning and ending indices to reflect the merge.
            route_begin[begin_j] = end_j
            route_begin[end_j] = end_j
            route_end[end_j] = end_i
            route_end[begin_j] = end_i

        # If both i and j are at the end of their routes, merge them.
        elif i == end_i and j == end_j:
            route_end[end_i] = begin_j

            # Update all the route beginning and ending indices to reflect the merge.
            route_begin[begin_j] = begin_i
            route_begin[end_j] = begin_i
            route_end[end_j] = begin_j
            route_end[begin_j] = begin_j

        # If the number of edges added reaches N-1, the tour is complete.
        if merge_count == N - 1:
            break

    # After the loop, connect the final two nodes to complete the tour.
    final_begin = find_begin(route_begin, 0)
    final_end = find_end(route_end, 0)
    A[final_end, final_begin] = 1
    A[final_begin, final_end] = 1

    # Return the completed adjacency matrix and the number of merge iterations.
    return A, merge_iterations

# Helper function to find the root of the partial route for node i, with path compression.
cpdef find_begin(int[:] route_begin, int i):
    cdef int begin_i = route_begin[i]
    if begin_i != i:
        # Recursively find the root and update the current node's parent for path compression.
        begin_i = find_begin(route_begin, begin_i)
        route_begin[i] = begin_i
    return begin_i

# Helper function to find the root of the partial route for node i, with path compression.
cpdef find_end(int[:] route_end, int i):
    cdef int end_i = route_end[i]
    if end_i != i:
        # Recursively find the root and update the current node's parent for path compression.
        end_i = find_end(route_end, end_i)
        route_end[i] = end_i
    return end_i
