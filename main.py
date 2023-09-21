import torch
import torch.nn as nn
import visualization
from torch_geometric.utils import add_remaining_self_loops, to_networkx
import preprocessing
from model import *
from constants import *
import numpy as np


data = preprocessing.prepare_data(True)
model = GAT()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
criterion = nn.CrossEntropyLoss()


subsample = preprocessing.graph_subsample(data)
G = to_networkx(subsample)
visualization.visualize_graph(G, subsample.y)


def train():
    train_mask = data.train_mask[:, 0]
    model.train()
    optimizer.zero_grad()
    out, x = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss, x


def test():
    model.eval()
    out, _ = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


best_acc = 0
loss_epochs = []
acc_epochs = []
for epoch in range(EPOCHS):
    loss_calc, x = train()
    loss_epochs.append(loss_calc)
    acc = test()
    acc_epochs.append(acc)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss_calc}')
        print('Test accuracy:', acc)
        visualization.visualize_embedding(x[:200], data.y[:200], epoch, loss_calc)

    if acc > best_acc:
        torch.save(model.state_dict(), 'models/GAT.pt')
        best_acc = acc

visualization.plot_loss_accuracy(loss_epochs, acc_epochs)

