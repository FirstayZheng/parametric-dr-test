import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
# from umap.umap_ import find_ab_params
from scipy.optimize import curve_fit
from dataset.edgeDataset import edgeDataset
import torch.utils.data
from utils.knn import compute_knn_graph


def find_ab_params(spread, min_dist):
    """
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

# def _make_epochs_per_sample(weights, n_epochs):
#     """Given a set of weights and number of epochs generate the number of
#     epochs per sample for each weight.

#     Parameters
#     ----------
#     weights: array of shape (n_1_simplices)
#         The weights of how much we wish to sample each 1-simplex.

#     n_epochs: int
#         The total number of epochs we want to train for.

#     Returns
#     -------
#     An array of number of epochs per sample, one for each 1-simplex.
#     """
#     # weights = np.ones_like(weights) / 2
#     result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
#     n_samples = n_epochs * (weights / weights.max())
#     result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
#     return result


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


def _construct_edge_dataset(
        X,
        graph_,
        n_epochs,
        config,
):
    """
    Construct a tf.data.Dataset of edges, sampled by edge weight.
        Parameters
    ----------
    X: origin data
    
    graph_: sparse matrix 
        umap graph
    
    n_epochs: int
        The total number of epochs we want to train for.
    """

    # get data from graph
    # 这里的graph是去除了重复点对关系之后和一些权重非常小的边之后的权重图
    # head和tail就分别表示了各数据点对的开始和结尾

    graph, epochs_per_sample, head, tail, weight, n_vertices = _get_graph_elements(
        graph_, n_epochs
    )

    # 根据每个点在一个epoch中的采样次数进行重复，此时的数据集就是根据权重采样得到的了
    # 权重越大表示越可能存在边，因此被采样的次数也越多
    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

       # shuffle edges 随机打乱
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
    print(f'edges_from_exp shape before {edges_from_exp.shape}')
    
    edges_to_exp = np.concatenate((edges_to_exp, edges_to_exp), axis=0)
    edges_from_exp = np.concatenate((edges_from_exp, edges_from_exp), axis=0)
    print(f'temporary edges_from_exp shape: {edges_from_exp.shape}')
    # create edge iterator
    # gather X
    X_to, X_from = X[edges_to_exp], X[edges_from_exp]
    edge_dataset = edgeDataset((X_to, X_from))

    edge_DataLoader = torch.utils.data.DataLoader(
       dataset = edge_dataset,
       batch_size = config.batch_size,
       shuffle = True,
    )
    
    return edge_DataLoader, len(edges_to_exp), weight

    #X_to, X_from = X[edges_to_exp], X[edges_from_exp]
    #edge_dataset = edgeDataset(X_to, X_from)
    #edge_DataLoader = torch.utils.data.DataLoader(
    #    dataset = edge_dataset,
    #    batch_size = batch_size,
    #    shuffle = True,
    #    drop_last = True,
    #)
    config.batch_size, config.epochs = batch_size, epochs
    return DataLoader_list, len(edges_to_exp), weight


def generate_graph(X, config):

    knn_indices, knn_dists = compute_knn_graph(
        all_data=X, 
        neighbors_cache_path=config.neighbors_cache_path,
        k=config.n_neighbors,
        pairwise_cache_path=config.pairwise_cache_path,
        metric="euclidean", 
        max_candidates=60, 
        accelerate=False,
        local=config.use_local_cache,
        knn_cutting=config.knn_cutting,
        config=config,
    )

    if config.save_knn:
        np.save(f'{config.output_dir}/knn_indices.npy', knn_indices)

    random_state = check_random_state(None)

    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X = X,
        n_neighbors = config.n_neighbors,
        metric = "euclidean",
        random_state = random_state,
        knn_indices = knn_indices,
        knn_dists = knn_dists,
    )

    return umap_graph
