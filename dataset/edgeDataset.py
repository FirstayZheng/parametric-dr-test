import torch.utils.data
import numpy as np


def _get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    # CSR -> COO
    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates() # in place operation
    # number of vertices in dataset

    n_vertices = graph.shape[1]
    
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    # 主要是为了方便之后的采样策略，对于权重非常小的边，可以看作两点之间没有连通性，消除即可
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    # 删除0值的键值对
    graph.eliminate_zeros()
    
    # get epochs per sample based upon edge probability
    # 每个epoch中每条边被采样的次数，根据权重求出来的，权重越大采样次数越多
    epochs_per_sample = n_epochs * graph.data

    head = graph.row #graph.row -> COO COO format column index array of the matrix
    tail = graph.col #graph.col -> COO format row index array of the matrix
    weight = graph.data #graph.data -> COO format data array of the matrix

    return graph, epochs_per_sample, head, tail, weight, n_vertices


class edgeDataset(torch.utils.data.Dataset):
    def __init__(self, data, graph_, n_epochs=200):
        graph, epochs_per_sample, head, tail, weight, n_vertices = _get_graph_elements(graph_, n_epochs)
        
        self.edges_to_exp, self.edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        print("self.edges_to_exp.shape", self.edges_to_exp.shape)
        self.data = torch.Tensor(data)
        # exit()
        
    def __len__(self):
        return int( self.edges_from_exp.shape[0])
    
    def __getitem__(self, index):
        # print("index:", index)
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)

