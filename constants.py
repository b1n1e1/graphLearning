from torch_geometric.datasets import WikiCS

BENCHMARK = WikiCS
ROOT = 'data'
_DATASET = BENCHMARK(root=ROOT)  # Do not use this value

FEATURES = _DATASET.num_features
CLASSES = _DATASET.num_classes

LAYER1 = 100
LAYER2 = 50
DROPOUT = 0.4

LR = 1e-2
DECAY = 5e-4
EPOCHS = 501