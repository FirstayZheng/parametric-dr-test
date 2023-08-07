import numpy as np
import utils.knn
import sys
import os
import re
from utils.visualization import plot_curve

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
    for i in range(n):
        U = np.setdiff1d(low_knn_indices[i], knn_indices[i])
        sum_j = 0
        for j in range(U.shape[0]):
            high_rank = np.where(high_nn_indices[i] == U[j])[0][0]
            sum_j += high_rank - k
        sum_i += sum_j
    
    return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)

def main():
    if len(sys.argv) < 3:
        print('Error! python evaluation.py k the/folder/of/data')
        return
    folder = sys.argv[2]
    folder = '/'.join(folder.split('/'))
    save_path = folder
    k = int(sys.argv[1])
    
    high_nn_indices = np.load(f'{folder}/high_nn_indices.npy')
    knn_indices = np.load(f'{folder}/knn_indices.npy')
    filenameList = []
    for filename in os.listdir(folder):
        if re.match(r'procedure_[0-9]*\.npy', filename):
            filenameList.append(filename)
    filenameList.sort()
    continuity_list = []
    trustworthiness_list = []
    for data in filenameList:
        data = '/'.join([folder, data])
        print(data)
        data = np.load(data)
        low_dis_matrix, low_knn_indices = get_low_dist(data, k)
        continuity_val = continuity(k, knn_indices, low_knn_indices, low_dis_matrix)
        trustworthiness_val = trustworthiness(k, high_nn_indices, low_knn_indices, knn_indices)
        continuity_list.append(continuity_val)
        trustworthiness_list.append(trustworthiness_val)
    plot_curve(continuity_list, save_path, 'continuity')
    plot_curve(trustworthiness_list, save_path, 'trustworthiness')


if __name__ == '__main__':
    main()