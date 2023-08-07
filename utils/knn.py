import numpy as np
import os
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from pynndescent import NNDescent
import logging
import sys
import math

__INF = np.inf

def get_pairwise_distance(flattened_data,
                          metric,
                          pairwise_distance_cache_path=None,
                          local=False,
                          preload=False):
    # to check whether there is cache first
    print(f'local:{local}')
    if local and pairwise_distance_cache_path is not None and preload and os.path.exists(
            pairwise_distance_cache_path):
        pairwise_distance = np.load(pairwise_distance_cache_path)
        logging.info("directly load pairwise distance from {}".format(
            pairwise_distance_cache_path))
    else:
        # logging.info("computing pairwise distance...")
        print(f'flattened_data shape: {flattened_data.shape}')
        pairwise_distance = pairwise_distances(flattened_data,
                                               metric=metric,
                                               squared=False)
        # pairwise_distance = pdist(flattened_data, metric="sqeuclidean")
        pairwise_distance[pairwise_distance < 1e-12] = 0.0
        # pairwise_distance = np.divide(pairwise_distance, flattened_data.shape[1])
        if preload and pairwise_distance_cache_path is not None:
            np.save(pairwise_distance_cache_path, pairwise_distance)
            logging.info(
                "successfully compute pairwise distance and save to {}".format(
                    pairwise_distance_cache_path))
    return pairwise_distance


# Calculate the accurate KNN graph
def compute_accurate_knn(flattened_data,
                         k,
                         config,
                         neighbors_cache_path=None,
                         pairwise_cache_path=None,
                         metric="euclidean",
                         local=False,
                         ):

    cur_path = None
    if neighbors_cache_path is not None:
        cur_path = neighbors_cache_path.replace(".npy", "_ac.npy")

        # cur_path = neighbors_cache_path.replace(".npy", "_opt.npy")
        # cur_path = neighbors_cache_path.replace(".npy", "_cls.npy")

    if local and cur_path is not None and os.path.exists(cur_path):
        knn_indices, knn_distances = np.load(cur_path)
        logging.info(
            "directly load accurate neighbor_graph from {}".format(cur_path))
    else:
        preload = flattened_data.shape[0] <= 30000

        pairwise_distance = get_pairwise_distance(flattened_data,
                                                  metric,
                                                  pairwise_cache_path,
                                                  local=local,
                                                  preload=preload)

        sorted_indices = np.argsort(pairwise_distance, axis=1)
        knn_indices = sorted_indices[:, 1:k + 1]
        knn_distances = []
        for i in range(knn_indices.shape[0]):
            knn_distances.append(pairwise_distance[i, knn_indices[i]])
        knn_distances = np.array(knn_distances)
        if cur_path is not None:
            np.save(cur_path, [knn_indices, knn_distances])
            logging.info(
                "successfully compute accurate neighbor_graph and save to {}".
                format(cur_path))
    #np.save("knn_indices.npy",knn_indices)
    #np.save("knn_distances.npy", knn_distances)
    return knn_indices, knn_distances

def purge_connection(dists, A, B):
    _len_0 = dists.shape[0]
    for i in range(_len_0):
        _len_1 = dists[i].shape[0]
        if i in A:
            for j in range(_len_1):
                if j in B:
                    dists[i][j] = np.inf
        elif i in B:
            for j in range(_len_1):
                if j in A:
                    dists[i][j] = np.inf
    return dists

def _norm(dists, A, B, all):
    print(dists.shape)
    AB = np.union1d(A, B)
    #print(AB)
    rest = np.setdiff1d(all, AB)
    rest_data = dists[rest]
    #val_rest = np.linalg.norm(rest_data)
    val_rest = np.average(rest_data)
    print(f'val rest:{val_rest}')
    print(rest_data)
    mag_rest = order_of_magnitude(val_rest)

    AB_data = dists[AB]
    #val_AB = np.linalg.norm(AB_data)
    val_AB = np.average(AB_data)
    print(f'val AB:{val_AB}')
    mag_AB = order_of_magnitude(val_AB)

    
    dists[AB] /= (10 ** mag_AB)
    dists[AB] *= (10 ** mag_rest)
    return dists

def order_of_magnitude(number):
    return math.floor(math.log(number, 10))
    

def special_knn_for_face(
        flattened_data,
        neighbors_cache_path,
        k,
        pairwise_cache_path,
        config,
        metric="euclidean",
        max_candidates=60,
        accelerate=False,
        local=False,
        knn_cutting=False,
    )->np.ndarray:
    cur_path = None
    #print(f'flattened_data shape: {flattened_data.shape}')
    index_l, index_r = load_face_index(config.dataset_path)
    preload = flattened_data.shape[0] <= 30000

    pairwise_distance = get_pairwise_distance(
            flattened_data,
            metric,
            pairwise_cache_path,
            local,
            preload,
        )
    print(f'pairwise_distance shape:{pairwise_distance.shape}')
    pairwise_distance = purge_connection(pairwise_distance, index_l, index_r)
    print(f'pairwise_distance shape:{pairwise_distance.shape}')
    sorted_indices = np.argsort(pairwise_distance, axis=1)
    knn_indices = sorted_indices[:, 1:k + 1]
    knn_distances = []
    for i in range(knn_indices.shape[0]):
        knn_distances.append(pairwise_distance[i, knn_indices[i]])
    knn_distances = np.array(knn_distances)
    print(f'knn_dists shape:{knn_distances.shape}')
    if cur_path is not None:
        np.save(cur_path, [knn_indices, knn_distances])
        logging.info("successfully compute accurate neighbor_graph and save to {}".format(cur_path))

    all_idx = np.arange(knn_distances.shape[0])
    print(knn_distances)
    #knn_distances = _norm(knn_distances, index_l, index_r, all_idx)
    return knn_indices, knn_distances



def compute_knn_graph(
        all_data,
        neighbors_cache_path,
        k,
        pairwise_cache_path,
        config,
        metric="euclidean",
        max_candidates=60,
        accelerate=False,
        local=False,
        knn_cutting=False,
    ):
    _computing_KNN = lambda data: _compute_knn_graph(
        data,
        neighbors_cache_path,
        k,
        pairwise_cache_path,
        config,
        metric,
        max_candidates,
        accelerate,
        local,
    )
    if not knn_cutting:
        return _computing_KNN(all_data)
    if config.special and config.dataset == 'face':
        #special knn cutting for face dataset
        print('using special knn cutting for face')
        return special_knn_for_face(
                    all_data,
                    neighbors_cache_path,
                    k,
                    pairwise_cache_path,
                    config,
                    metric="euclidean",
                    max_candidates=60,
                    accelerate=False,
                    local=local,
                    knn_cutting=False,
                )
    print('Do KNN cutting!')
    left, right = load_index(config.dataset_path)
    left_data = all_data[left]
    right_data = all_data[right]
    l_knn_indices, l_knn_dists = _computing_KNN(left_data)
    r_knn_indices, r_knn_dists = _computing_KNN(right_data)
    convert_l = np.vectorize(lambda i: left[i])
    convert_r = np.vectorize(lambda i: right[i])
    l_knn_indices = convert_l(l_knn_indices).reshape(l_knn_indices.shape)
    r_knn_indices = convert_r(r_knn_indices).reshape(r_knn_indices.shape)
    knn_indices = np.concatenate([l_knn_indices, r_knn_indices], axis=0)
    index = np.argsort(np.concatenate([left, right]))
    knn_indices = knn_indices[index]
    knn_dists = np.concatenate([l_knn_dists, r_knn_dists], axis=0)
    knn_dists = knn_dists[index]
    return knn_indices, knn_dists

def load_index(path:str)->np.ndarray:
    path = '/'.join(path.split('/'))
    index_0 = np.load(f'{path}/left.npy')
    index_1 = np.load(f'{path}/right.npy')
    return index_0, index_1

def load_face_index(path:str)->np.ndarray:
    path = '/'.join(path.split('/'))
    index_0 = np.load(f'{path}/l.npy')
    index_1 = np.load(f'{path}/r.npy')
    return index_0, index_1

# Calculate the approximate KNN graph
def _compute_knn_graph(all_data,
                      neighbors_cache_path,
                      k,
                      pairwise_cache_path,
                      config,
                      metric="euclidean",
                      max_candidates=60,
                      accelerate=False,
                      local=False,
    ):

    """_summary_

    Args:
        all_data (_type_): dataset
        neighbors_cache_path (str): the path of neighbors cache file
        k (int): the number of nearest neighbors
        pairwise_cache_path (str): the path of pairewise cache file
        metric (str, optional): _description_. Defaults to "euclidean".
        max_candidates (int, optional): _description_. Defaults to 60.
        accelerate (bool, optional): _description_. Defaults to False.
        local (bool, optional): whether to use local cache or not Defaults to False.

    Returns:
        _type_: _description_
    """
    flattened_data = all_data.reshape(
        (len(all_data), np.product(all_data.shape[1:])))
    # 精确的KNN，比较慢
    if not accelerate:
        knn_indices, knn_distances = compute_accurate_knn(
            flattened_data=flattened_data,
            k=k,
            neighbors_cache_path=neighbors_cache_path,
            pairwise_cache_path=pairwise_cache_path,
            local=local,
            config=config,
        )
        return knn_indices, knn_distances

    # 近似的KNN，比较快
    if local and neighbors_cache_path is not None and os.path.exists(
            neighbors_cache_path):
        neighbor_graph = np.load(neighbors_cache_path)
        knn_indices, knn_distances = neighbor_graph
        logging.info("directly load approximate neighbor_graph from {}".format(
            neighbors_cache_path))
    else:
        # number of trees in random projection forest
        n_trees = 5 + int(round((all_data.shape[0])**0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(all_data.shape[0]))))

        nnd = NNDescent(flattened_data,
                        n_neighbors=k + 1,
                        metric=metric,
                        n_trees=n_trees,
                        n_iters=n_iters,
                        max_candidates=max_candidates,
                        verbose=False)
        # 获取近邻点下标和距离
        knn_indices, knn_distances = nnd.neighbor_graph
        knn_indices = knn_indices[:, 1:]
        knn_distances = knn_distances[:, 1:]
        # 缓存邻近图
        if neighbors_cache_path is not None:
            np.save(neighbors_cache_path, [knn_indices, knn_distances])
        logging.info(
            "successfully compute approximate neighbor_graph and save to {}".
            format(neighbors_cache_path))
    return knn_indices, knn_distances

# knn cutting


def _purge_(source, target, index):
    length = source.shape[0]
    for i in range(length):
        if i in target:
            t_len = source[i].shape[0]
            for j in range(t_len):
                if j in index:
                    source[i][j] = __INF
    return source








# Calculate the geodesic distance matrix for ISOMAP
def get_geodesic_pairwise_distance(data):
    #TODO Write more elegantly
    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    isomap.fit(X=data)
    geo_distance_matrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample

    # if geodesic_cache_path is not None:
    #     np.save(geodesic_cache_path, geo_distance_matrix)
        # logging.info("successfully compute geo_distance_metrix and save to {}".format(geodesic_cache_path))

    # if geodesic_cache_path is not None and os.path.exists(geodesic_cache_path):
    #     geo_distance_matrix = np.load(geodesic_cache_path)
    #     logging.info("directly load geo_distance_metrix from {}".format(
    #         geodesic_cache_path))
    # else:
    #     isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    #     isomap.fit(X=data)
    #     geo_distance_matrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample

    #     if geodesic_cache_path is not None:
    #         np.save(geodesic_cache_path, geo_distance_matrix)
    #     # logging.info("successfully compute geo_distance_metrix and save to {}".format(geodesic_cache_path))

    return geo_distance_matrix
    # return geo_distance_matrix


