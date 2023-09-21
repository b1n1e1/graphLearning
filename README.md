# graphLearning
First project utilizing GNN's for prediction task. 

In this case, we are using the Wiki CS dataset, a dataset comprising of 11,000 wikipedia articles in the field of Computer Science. Each article has a subcategory (10 subcategories) and the edges in the graph correspond to articles containing a link to other articles.

The task: predict the subcategory of each article.

Loss used: Cross entropy loss

Optimizer: Adam

Check constants to see exact values used for learning rate, decay, etc.

The models being compared were GCN (Graph Convolutional Network) and GAT (Graph Attention Network). The GAT had better results overall, leading to around 66% test accuracy. I did not play around too much with the hidden layer sizes, instead I made a second hidden layer that was 2 dimensional just so I could visualize the embeddings of the nodes.
