import numpy as np
import utils.knn
import sys
import os
import re
from utils.visualization import plot_curve

def transition_error(embedding_data_list:list) -> float:
    total_error = 0
    num_trainsitions = len(embedding_data_list) - 1
    num_points = len(embedding_data_list[0])
    for i in range(num_transitions):
        embedding_data_1 = embedding_data_list[i]
        embedding_data_2 = embedding_data_list[i+1]
        transition = embedding_data_2 - embedding_data_1
        for j in len(num_points):
            total_error += np.linalg.norm(transition[j])

    return total_error / (num_transitions - 1) / num_points


def main():
    dir_path = sys.argv[1]
    filenameList = []
    for filename in os.listdir(dir_path):
        if re.match(r'procedure_[0-9]*\.npy', filename):
            filenameList.append(filename)
    filenameList.sort()
    embedding_data_list = []
    for file in filenameList:
        _data = np.load(file)
        embedding_data_list.append(_data)
    TE_val = transition_error(embedding_data_list)


if __name__ == '__main__':
    main()
