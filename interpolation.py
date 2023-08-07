import numpy as np
import sys


def interpolation(start_embedding:np.array, end_embedding:np.array,
                  num_transitions:int) -> list:
    embedding_list = []
    sum_matrix = start_embedding + end_embedding
    piece_matrix = sum_matrix / num_transitions
    embedding_list.append(start_embedding)
    for i in range(num_transitions - 1):
        inter_matrix = embedding_list[-1] + piece_matrix
        embedding_list.append(inter_matrix)

    embedding_list.append(end_embedding)
    return embedding_list

def main():
    startFile = sys.argv[1]
    endFile = sys.argv[2]
    num_transitions = int(sys.argv[3])
    start_embedding = np.load(startFile)
    end_embedding = np.load(endFile)
    embedding_list = interpolation(start_embedding, end_embedding,
                                   num_transitions)
    for p, i in enumerate(embedding_list):
        np.save(f'procedure_{p:03}.npy', i)
    

if __name__ == '__main__':
    main()
