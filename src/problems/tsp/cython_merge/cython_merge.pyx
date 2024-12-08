#cython: language_level=3

import numpy as np
np.import_array()
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple[np.ndarray, int] _merge_core(double[:,:] coords, double[:,:] adj_mat):
    cdef double[:,:] points = coords
    cdef double[:,:] dists = np.linalg.norm(np.asarray(points)[:, None] - np.asarray(points), axis=-1)
    cdef long N = dists.shape[0]
    
    # we initialize the real adjacency matrix
    cdef double[:,:] A = np.zeros((N, N))
    cdef int[:] route_begin = np.arange(N, dtype='int32')
    cdef int[:] route_end = np.arange(N, dtype='int32')
    
    # we calculate the dist between each pair of nodes
    dist = np.linalg.norm(np.asarray(points)[:, None] - np.asarray(points), axis=-1)
    
    # we sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk
    cdef int[:] sorted_edges = np.argsort((-np.asarray(adj_mat) / dist).flatten()).astype('int32')
    cdef int i, j, begin_i, end_i, begin_j, end_j
    
    merge_iterations = 0
    merge_count = 0
    
    for edge in sorted_edges:
        merge_iterations += 1
        i = int(edge // N)
        j = int(edge % N)
        
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
            
    cdef int final_begin = find_begin(route_begin, 0)
    cdef int final_end = find_end(route_end, 0)
    A[final_end, final_begin] = 1
    A[final_begin, final_end] = 1
    
    return np.asarray(A), merge_iterations

cpdef merge_cython(double[:,:] coords, double[:,:] adj_mat):
    return _merge_core(coords, adj_mat)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple[list[int], int] merge_cython_get_tour(double[:,:] coords, double[:,:] adj_mat):
    A, merge_iterations = _merge_core(coords, adj_mat)
    
    # Convert adjacency matrix to tour
    cdef list[int] tour = [0]
    cdef int current_node = 0, prev_node = -1
    cdef np.ndarray[np.int64_t, ndim=1] nonzero_nodes
    
    while len(tour) < A.shape[0] + 1:
        nonzero_nodes = np.nonzero(A[current_node])[0]
        if prev_node >= 0:
            nonzero_nodes = nonzero_nodes[nonzero_nodes != prev_node]
        next_node = nonzero_nodes.max()
        tour.append(next_node)
        prev_node = current_node
        current_node = next_node
    
    return np.asarray(tour), merge_iterations

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
