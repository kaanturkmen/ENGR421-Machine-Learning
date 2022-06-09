import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy import linalg


def main():
    """
    Main method for Spectral Clustering.
    """

    dataset = read_dataset()
    N, K, R, delta = dataset.shape[0], 9, 5, 2.0

    initial_centroid_row = [242, 528, 570, 590, 648, 667, 774, 891, 955]

    B = construct_euclidean_distance_matrix(N, dataset, delta)

    visualize_connectivity_matrix(B, N, dataset)

    L_symmetric = create_laplacian_matrix(B, N)

    Z = construct_first_R_smallest_eigenvector_matrix(L_symmetric, R)

    centroids, memberships = iterate_algorithm(initial_centroid_row, Z, K)

    plot_current_state(memberships, dataset, K)


def construct_first_R_smallest_eigenvector_matrix(L_symmetric, R):
    """
    Calculates eigenvalues and eigenvectors, then selects first R smallest eigenvectors.
    :param L_symmetric: Laplacian matrix.
    :param R: Number of smallest vectors to be chosen.
    :return: Array that contains R smallest eigenvectors.
    """

    eigenvalues, eigenvectors = np.linalg.eig(L_symmetric)
    selected_smallest_vectors = np.argsort(eigenvalues)[1:(R + 1)]
    Z = eigenvectors[:, selected_smallest_vectors]
    return Z


def create_laplacian_matrix(B, N):
    """
    Creates laplacian matrix.
    :param B: Euclidean distance matrix.
    :param N: Number of data points.
    :return: Laplacian matrix.
    """

    D = np.diag(np.sum(B, axis=0))
    inverse_D = linalg.cho_solve(linalg.cho_factor(D), np.eye(N))
    sqrt_inverse_D = np.sqrt(inverse_D)
    L_symmetric = np.eye(N) - np.matmul(sqrt_inverse_D, np.matmul(B, sqrt_inverse_D))
    return L_symmetric


def visualize_connectivity_matrix(B, N, dataset):
    """
    Plots euclidean distance matrix on the graph.
    :param B: Euclidean distance matrix.
    :param N: Number of data points.
    :param dataset: Given dataset.
    """

    plt.figure(figsize=(6, 6))
    for i in range(N):
        for j in range(N - i):
            if B[i][j + i] == 1:
                x_values = [dataset[i, 0], dataset[j + i, 0]]
                y_values = [dataset[i, 1], dataset[j + i, 1]]
                plt.plot(x_values, y_values, "-", linewidth=0.5, color="grey")

    plt.plot(dataset[:, 0], dataset[:, 1], ".", markersize=10, color="black")
    plt.ylim([-8, 8])
    plt.xlim([-8, 8])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()


def construct_euclidean_distance_matrix(N, dataset, delta):
    """
    Creates euclidean distance matrix.
    :param N: Number of data points.
    :param dataset: Given dataset.
    :param delta: Given delta constant.
    :return: Euclidean distance matrix.
    """

    D = spa.distance_matrix(dataset, dataset)
    B = np.zeros((N, N)).astype(int)
    B[D < delta] = 1

    for i in range(N):
        B[i, i] = 0

    return B


def plot_current_state(memberships, X, K):
    """
    Plots current state of the centroids and memberships.
    :param memberships: Memberships of the centroids.
    :param X: Given data set.
    :param K: Number of data class.
    """

    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

    plt.figure(figsize=(6, 6))

    means = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])

    for c in range(K):
        plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                 color=cluster_colors[c])
        plt.plot(means[c, 0], means[c, 1], "s", markersize=12, markerfacecolor=cluster_colors[c],
                 markeredgecolor="black")

    plt.ylim([-8, 8])
    plt.xlim([-8, 8])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()


def iterate_algorithm(initial_centroid_row, Z, K):
    """
    Runs iteration algorithm.
    :param initial_centroid_row: Rows for initial centroids.
    :param Z: First R eigenvectors.
    :param K: Number of data class.
    :return: Updated centroids and their memberships.
    """

    centroids = Z[initial_centroid_row]
    memberships = update_memberships(centroids, Z)
    iteration = 1

    while True:
        old_centroids = centroids
        centroids = update_centroids(memberships, Z, K)
        if np.alltrue(centroids == old_centroids):
            break

        old_memberships = memberships
        memberships = update_memberships(centroids, Z)
        if np.alltrue(memberships == old_memberships):
            break

        iteration = iteration + 1

    return centroids, memberships


def update_centroids(memberships, Z, K):
    """
    Updates centroids.
    :param memberships: Memberships of the centroids.
    :param Z: First R eigenvectors.
    :param K: Number of data class.
    :return: Updated centroids.
    """

    return np.vstack([np.mean(Z[memberships == k, :], axis=0) for k in range(K)])


def update_memberships(centroids, Z):
    """
    Updates memberships.
    :param centroids: Centroids.
    :param Z: First R eigenvectors
    :return: Memberships of the centroids.
    """

    D = spa.distance_matrix(centroids, Z)
    memberships = np.argmin(D, axis=0)
    return memberships


def read_dataset():
    """
    Reads the csv file.
    :return: Train arrays of the data set.
    """

    return np.genfromtxt("hw09_data_set.csv", delimiter=",")


if __name__ == "__main__":
    main()
