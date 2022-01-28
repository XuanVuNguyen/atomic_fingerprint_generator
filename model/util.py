import torch
import numpy as np
from typing import Optional, Union, List

def a2a_to_adjacency(batch_a2a: torch.Tensor, 
                     atom_scope: Union[torch.Tensor, np.ndarray, List[List[int]]]):
    if isinstance(atom_scope, torch.Tensor):
        atom_scope = atom_scope.numpy().tolist()
    elif isinstance(atom_scope, np.ndarray):
        atom_scope = atom_scope.tolist()
        
    adjacencies = []
    for start_id, size in atom_scope:
        adjacency = np.zeros((size, size), dtype=np.int32)
        
        a2a = batch_a2a.narrow(0, start_id, size).numpy()
        a2a = a2a - (start_id-1)*(a2a > 0)
        
        one_ids = []
        for i, row in enumerate(a2a):
            one_ids.extend([(i, j-1) for j in row if j>0])
        
        if one_ids: 
        # there are cases that a graph having only one node, or no connection between nodes. In such cases, one_ids is empty
            one_i, one_j = zip(*one_ids)       
            adjacency[one_i, one_j] = 1
        
        adjacencies.append(adjacency)

    return adjacencies

def adjacency_to_distance(adjacency: np.ndarray, capped_distance: Optional[int]=None):
    '''
    Based on:
    https://github.com/coleygroup/Graph2SMILES/blob/fb510518b773a977d08d65da6d45c29b6e2cb1e0/utils/data_utils.py
    '''
    distance = adjacency.copy()
    shortest_paths = adjacency.copy()
    path_length = 2
    stop_counter = 0
    non_zeros = 0

    while 0 in distance:
        shortest_paths = np.matmul(shortest_paths, adjacency)
        shortest_paths = path_length * (shortest_paths > 0)
        new_distance = distance + (distance == 0) * shortest_paths

        # if np.count_nonzero(new_distance) == np.count_nonzero(distance):
        if np.count_nonzero(new_distance) <= non_zeros:
            stop_counter += 1
        else:
            non_zeros = np.count_nonzero(new_distance)
            stop_counter = 0

        if stop_counter == 3:
            break

        distance = new_distance
        path_length += 1
    
    if capped_distance is not None:
        distance[distance==0] = capped_distance
        distance[distance>capped_distance] = capped_distance
    else:
        distance[distance==0] = -1 # in experiment
        
    np.fill_diagonal(distance, 0) 
    return distance

def a2a_to_distance(batch_a2a: torch.Tensor, 
                    atom_scope: Union[torch.Tensor, np.ndarray, List[List[int]]],
                    capped_distance: Optional[int]=None) -> np.ndarray:
    adjacencies = a2a_to_adjacency(batch_a2a, atom_scope)
    distances = [adjacency_to_distance(adjacency, capped_distance) for adjacency in adjacencies]
    return distances