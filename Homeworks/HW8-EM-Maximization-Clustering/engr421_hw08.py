import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal


def main():
    # TODO I have tested my algorithm on two different computers,
    # TODO Algorithm's average run time: M1 Chip MBP -> 3 min. Intel Chip MBP -> 9 min.
    # TODO Please do not terminate the code unless you see "Application is complete" message.

    data_set = read_dataset()
    initial_centroids = read_initial_centroids()

    plot_dataset(data_set)

    N, K, number_of_iterations = 1000, 9, 100

    iteration, centroids, memberships = iterate_algorithm(data_set, K, initial_centroids)

    estimated_covariances = estimate_covariance(K, data_set, memberships)

    estimated_priors = estimate_priors(K, N, centroids, estimated_covariances, data_set, memberships)

    means = run_and_extract_mean(K, N, centroids, data_set,
                                 estimated_covariances, estimated_priors, number_of_iterations)

    print(means)

    plot_contours(K, centroids, data_set, estimated_covariances, memberships)

    print("Application is completed.")


def plot_contours(K, centroids, data_set, estimated_covariances, memberships):
    """
    Plot's contours of the classes.
    :param K: Number of data class.
    :param centroids: Centroids.
    :param data_set: Given data set.
    :param estimated_covariances: Covariances that we have estimated.
    :param memberships: Memberships of the centroids.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_current_state(centroids, memberships, data_set, K)
    x, y = np.mgrid[-8: 8: 0.05, -8: 8: 0.05]
    coord = np.empty(x.shape + (2,))
    coord[:, :, 0], coord[:, :, 1] = x, y
    for each_data_class in range(K):
        predicted_val = multivariate_normal(create_means()[each_data_class], create_covariances()[each_data_class])
        real_val = multivariate_normal(centroids[each_data_class], estimated_covariances[each_data_class])
        plt.contour(x, y, predicted_val.pdf(coord), levels=[0.05], linestyles='dashed')
        plt.contour(x, y, real_val.pdf(coord), levels=[0.05])
    plt.show()


def run_and_extract_mean(K, N, centroids, data_set, estimated_covariances, estimated_priors, number_of_iterations):
    """
    After initialization, running algorithm number_of_iterations times and training EM algorithm.
    :param K: Number of data class.
    :param N: Number of data points.
    :param centroids: Centroids.
    :param data_set: Given data set.
    :param estimated_covariances: Covariances that we have estimated.
    :param estimated_priors: Priors that we have estimated.
    :param number_of_iterations: Number of iterations for the algorithm.
    :return: Centroids.
    """
    for each_iteration in range(number_of_iterations):
        centroids, estimated_covariances, phi = m_step(data_set, centroids, estimated_covariances, estimated_priors, K,
                                                       N)
        estimated_priors = e_step(data_set, centroids, estimated_covariances, phi, N, K)
    return centroids


def estimate_priors(K, N, centroids, estimated_covariances, data_set, memberships):
    """
    Estimate priors with the given parameters.
    :param K: Number of data class.
    :param N: Number of data points.
    :param centroids: Centroids.
    :param estimated_covariances: Covariances that we have estimated.
    :param data_set: Given data sets.
    :param memberships: Memberships of the centroids.
    :return: Priors.
    """
    priors = []

    for each_data in range(N):
        total_data, pri_val = np.empty(0), np.empty(0)

        for each_class in range(K):
            m_varied_data = multivariate_normal(centroids[each_class],
                                                estimated_covariances[each_class]).pdf(data_set[each_data]) \
                            * (data_set[memberships == each_class].shape[0] / N)

            total_data = np.append(total_data, m_varied_data)
        total_n = np.sum(total_data)

        for each_class in range(K):
            m_varied_data = multivariate_normal(centroids[each_class],
                                                estimated_covariances[each_class]).pdf(data_set[each_data]) * \
                            (data_set[memberships == each_class].shape[0] / N)

            pri_val = np.append(pri_val, m_varied_data / total_n)
        priors.append(pri_val)
    return priors


def estimate_covariance(K, data_set, memberships):
    """
    Estimate covariances with the given parameters.
    :param K: Number of data class.
    :param data_set: Given data set.
    :param memberships: Memberships of the centroids.
    :return: Covariances.
    """
    return [np.cov(np.transpose(data_set[memberships == c])) for c in range(K)]


def create_means():
    """
    Create means with given values in the PDF.
    :return: List of means.
    """
    return np.array([[+5.0, +5.0],
                     [-5.0, +5.0],
                     [-5.0, -5.0],
                     [+5.0, -5.0],
                     [+5.0, +0.0],
                     [+0.0, +5.0],
                     [-5.0, +0.0],
                     [+0.0, -5.0],
                     [+0.0, +0.0]])


def create_covariances():
    """
    Create covariances with given values in the PDF.
    :return: List of covariances.
    """
    return np.array([[[+0.8, -0.6],
                      [-0.6, +0.8]],
                     [[+0.8, +0.6],
                      [+0.6, +0.8]],
                     [[+0.8, -0.6],
                      [-0.6, +0.8]],
                     [[+0.8, +0.6],
                      [+0.6, +0.8]],
                     [[+0.2, +0.0],
                      [+0.0, +1.2]],
                     [[+1.2, +0.0],
                      [+0.0, +0.2]],
                     [[+0.2, +0.0],
                      [+0.0, +1.2]],
                     [[+1.2, +0.0],
                      [+0.0, +0.2]],
                     [[+1.6, +0.0],
                      [+0.0, +1.6]]])


def create_sizes():
    """
    Create sizes with given values in the PDF.
    :return: List of sizes.
    """
    return np.array([100, 100, 100, 100, 100, 100, 100, 100, 200])


def e_step(dataset, centroids, estimated_covariances, phi, N, K):
    """
    Calculates Step-E of the EM algorithm.
    :param dataset: Given data set.
    :param centroids: Centroids.
    :param estimated_covariances: Covariances that we have estimated.
    :param phi: Phi value that is calculated with step M.
    :param N: Number of data points.
    :param K: Number of data class.
    :return: Priors.
    """
    priors = []
    for each_data_point in range(N):
        total = np.empty(0)
        temp_p = np.empty(0)
        for each_data_class in range(K):
            n = multivariate_normal(centroids[each_data_class], estimated_covariances[each_data_class]) \
                    .pdf(dataset[each_data_point]) * phi[each_data_class]
            total = np.append(total, n)
        total_n = np.sum(total)
        for each_data_class in range(K):
            n = multivariate_normal(centroids[each_data_class], estimated_covariances[each_data_class]) \
                    .pdf(dataset[each_data_point]) * phi[each_data_class]
            temp_p = np.append(temp_p, n / total_n)
        priors.append(temp_p)
    return priors


def m_step(dataset, centroids, estimated_covariances, estimated_priors, K, N):
    """
    Calculates Step-M of the EM algorithm.
    :param dataset: Given data set.
    :param centroids: Centroids.
    :param estimated_covariances: Covariances that we have estimated.
    :param estimated_priors: Priors that we have estimated.
    :param K: Number of data class.
    :param N: Number of data points.
    :return: Centroids, estimated_covariances and phi.
    """
    phi = []
    for each_data_class in range(K):
        multiplied_total = 0
        total = 0
        for each_data_point in range(N):
            multiplied_total += estimated_priors[each_data_point][each_data_class] * dataset[each_data_point]
            total += estimated_priors[each_data_point][each_data_class]
        centroids[each_data_class] = multiplied_total / total

    for each_data_class in range(K):
        multiplied_total = 0
        total = 0
        for each_data_point in range(N):
            multiplied_total += estimated_priors[each_data_point][each_data_class] * (
                np.dot((dataset[each_data_point].reshape(1, 2) - centroids[each_data_class].reshape(1, 2)).T,
                       (dataset[each_data_point].reshape(1, 2) - centroids[each_data_class].reshape(1, 2))))
            total += estimated_priors[each_data_point][each_data_class]
        estimated_covariances[each_data_class] = multiplied_total / total
    for each_data_class in range(K):
        multiplied_total = 0
        for each_data_point in range(N):
            multiplied_total += estimated_priors[each_data_point][each_data_class]
        phi.append(multiplied_total / N)
    return centroids, estimated_covariances, phi


def iterate_algorithm(X, K, initial_centroids):
    """
    Running iteration algorithm.
    :param X: Given data set.
    :param K: Number of data class.
    :param initial_centroids: Initial centroids that is given in the csv file.
    :return: iteration, centroids, memberships
    """
    centroids = None
    memberships = None
    iteration = 1
    while True:
        print("Iteration#{}:".format(iteration))

        old_centroids = centroids
        centroids = update_centroids(memberships, X, K, initial_centroids)
        if np.alltrue(centroids == old_centroids):
            break
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plot_current_state(centroids, memberships, X, K)

        old_memberships = memberships
        memberships = update_memberships(centroids, X)
        if np.alltrue(memberships == old_memberships):
            plt.show()
            break
        else:
            plt.subplot(1, 2, 2)
            plot_current_state(centroids, memberships, X, K)

        iteration = iteration + 1

    return iteration, centroids, memberships


def update_centroids(memberships, X, K, initial_centroids):
    """
    Updates centroids.
    :param memberships: Memberships of the centroids.
    :param X: Given data set.
    :param K: Number of data class.
    :param initial_centroids: Initial centroids that is given in the csv file.
    :return: Centroids.
    """
    if memberships is None:
        centroids = initial_centroids
    else:
        centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
    return centroids


def update_memberships(centroids, X):
    """
    Updates memberships.
    :param centroids: Centroids.
    :param X: Given data set.
    :return: Memberships of the centroids.
    """
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return memberships


def plot_dataset(data_set):
    """
    Plots requested figure in the PDF.
    :param data_set: Given data set.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data_set[:, 0], data_set[:, 1], "k.")


def plot_current_state(centroids, memberships, X, K):
    """
    Plots current state of the centroids and memberships.
    :param centroids: Centroids.
    :param memberships: Memberships of the centroids.
    :param X: Given data set.
    :param K: Number of data class.
    """
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")


def read_dataset():
    """
    Reads the csv file.
    :return: Train arrays of the data set.
    """
    return np.genfromtxt("hw08_data_set.csv", delimiter=",")


def read_initial_centroids():
    """
    Reads the csv file.
    :return: Test arrays of the data set.
    """
    return np.genfromtxt("hw08_initial_centroids.csv", delimiter=",")


if __name__ == "__main__":
    np.random.seed(421)
    main()
