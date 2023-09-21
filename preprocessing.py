import torch
import torch_geometric
from torch_geometric.utils import add_remaining_self_loops
from constants import *


def reset_graph(subset_indices, data):
    mappings = {}
    new_id = 0
    for index in subset_indices.tolist():
        mappings[index] = new_id
        new_id += 1

    new_edge_index = torch.tensor([[mappings[i] for i in edge.tolist()] for edge in data.edge_index])
    return torch_geometric.data.Data(x=data.x, edge_index=new_edge_index, y=data.y)


def graph_subsample(data, size=100):
    """
    Create smaller version of graph in order to be able to visualize / do faster operations
    """
    subset_indices = torch.unique(data.edge_index.transpose(1, 0).flatten()[:size]).type(torch.LongTensor)
    subset_x = data.x[subset_indices]
    subset_y = data.y[subset_indices]
    mask = torch.isin(data.edge_index[0], subset_indices) & torch.isin(data.edge_index[1], subset_indices)
    subset_edge_index = data.edge_index[:, mask]
    data = torch_geometric.data.Data(x=subset_x, edge_index=subset_edge_index, y=subset_y)
    return reset_graph(subset_indices, data)


def prepare_data(self_loops=False):
    dataset = BENCHMARK(root=ROOT, is_undirected=False)
    if self_loops:
        edges_with_self_loops = add_remaining_self_loops(dataset.edge_index)[0]
        dataset.edge_index = edges_with_self_loops
    return dataset
