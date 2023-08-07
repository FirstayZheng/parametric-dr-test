import numpy as np
import utils.knn

def get_low_dist(embedding_data, k):
    embedding_data = np.reshape(embedding_data, (-1, np.product(embedding_data.shape[1:])))
    low_dis_matrix = utils.knn.get_pairwise_distance(embedding_data, 'euclidean')
    low_knn_indices = np.argsort(low_dis_matrix, axis=1)[:, 1:k + 1]
    return low_dis_matrix, low_knn_indices



def continuity(k, knn_indices, low_knn_indices, low_dis_matrix):

    n = knn_indices.shape[0]
    sum_i = 0 

    for i in range(n):
        V = np.setdiff1d(knn_indices[i], low_knn_indices[i]) # 求差集, 低维空间丢失的点

        pro_nn_indices = np.argsort(low_dis_matrix[i])
        sum_j = 0
        for j in range(V.shape[0]):
            low_rank = np.where(pro_nn_indices == V[j])[0]
            sum_j += low_rank - k
        sum_i += sum_j
    return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)

def trustworthiness(k, high_nn_indices, low_knn_indices, knn_indices):
    sum_i = 0
    n = knn_indices.shape[0]
    for i in range(0):
        U = np.setdiff1d(low_knn_indices[i], knn_indices[i])
        sum_j = 0
        for j in range(U.shape[0]):
            high_rank = np.where(high_nn_indices[i] == U[j])[0][0]
            sum_j += high_rank - k
        sum_i += sum_j
    
    return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)