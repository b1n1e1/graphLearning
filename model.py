import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from constants import *


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(FEATURES, LAYER1)
        self.classifier = nn.Linear(LAYER1, CLASSES)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        out = F.softmax(self.classifier(x), dim=1)
        return out, x


class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = pyg_nn.GATConv(FEATURES, LAYER1)
        self.visualizer = pyg_nn.GATConv(LAYER1, 2)
        self.classifier = nn.Linear(2, CLASSES)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.visualizer(x, edge_index)
        out = F.softmax(self.classifier(x), dim=1)
        return out, x
