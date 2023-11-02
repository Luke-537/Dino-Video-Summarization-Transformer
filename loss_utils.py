import numpy as np


def create_correllation_matrix(loss_list):

    correlation = np.correlate(loss_list, loss_list, mode='full')
    normalized_correlation = correlation / correlation[len(loss_list) - 1]


    return normalized_correlation

if __name__ == '__main__':

    print("lol")