import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as linalg


def main():
    """
    Main method of the ML program.
    """

    # Creating means, covariances and sizes by using data.
    means, covariances, sizes = create_means(), create_covariances(), create_sizes()

    # Creating points by using these values.
    points = create_points(means, covariances, sizes)

    # Combining points together and assigning them to the x.
    x = np.concatenate(points)

    # Creating labels for the points.
    y = create_labes_for_points(sizes)

    # K is number of classes = 4.
    # N is total point count = 500.
    # D is dimension = 2.
    # in our given data set.
    K = np.max(y)
    N = x.shape[0]
    D = len(x[0])

    # Plotting points on the graph.
    plot_graph(points)

    # # Commented out this section thus since we are generating numbers, we do not need to write or read from the file.
    # write_to_the_file(x, y, D)
    #
    # D, x, y, K, N = read_from_file()

    # Since normally we do not know the means, covariances and number of points in real life example,
    # we are trying to estimate or count these information. We will be using this information to train our ML program.
    sample_means = calculate_sample_mean(x, y, K)
    sample_covariances = calculate_sample_covariance(sample_means, x, y, K)
    class_priors = calculate_class_priors(y, K)

    # Printing the sample data.
    print("\nSample Mean: \n" + str([sample_means[i].tolist() for i in range(K)]))
    print("\nSample Covariance: \n" + str([sample_covariances[i].tolist() for i in range(K)]))
    print("\nClass Priors: \n" + str(class_priors))

    # Calculating required Wc, wc and wc0 data using sample data.
    Wc, wc, wc0 = create_w_values(sample_means, sample_covariances, class_priors, K, D)

    # Calculating score functions.
    score_function = create_score_function(Wc, wc, wc0, x, K, N)

    # Obtaining y predicted by picking the maximum between score functions.
    y_predicted = find_maximum_score(score_function)

    # Creating confusion matrix and understanding which data is labeled wrongly or correctly.
    confusion_matrix = create_confusion_matrix(y_predicted, y)

    # Printing confusion matrix.
    print("\nConfusion Matrix: \n" + str(confusion_matrix))

    # Creating decision boundaries.
    create_desicion_boundaries(Wc, wc, wc0, x, y, y_predicted, K)


def create_means():
    """
    Create means using numpy array data structure.
    :return: Numpy array of means.
    """
    return np.array([
        [0.0, 4.5],
        [-4.5, -1.0],
        [4.5, -1.0],
        [0.0, -4.0]
    ])


def create_covariances():
    """
    Create covariances using numpy array data structure.
    :return: Numpy array of covariances.
    """
    return np.array([
        [[3.2, 0.0], [0.0, 1.2]],
        [[1.2, 0.8], [0.8, 1.2]],
        [[1.2, -0.8], [-0.8, 1.2]],
        [[1.2, 0.0], [0.0, 3.2]]
    ])


def create_sizes():
    """
    Create sizes of class using numpy array data structure.
    :return: Numpy array of sizes.
    """
    return np.array([105, 145, 135, 115])


def create_points(means, covariances, sizes):
    """
    Create points using multivariate normal distribution using given mean, covariance and size.
    :param means: Mean of a distribution to be created.
    :param covariances: Covariance of a distribution to be created.
    :param sizes: Size of a distribution to be created.
    :return: List of points which has multivariate normal distribution.
    """
    return (
        np.random.multivariate_normal(means[0], covariances[0], sizes[0]),
        np.random.multivariate_normal(means[1], covariances[1], sizes[1]),
        np.random.multivariate_normal(means[2], covariances[2], sizes[2]),
        np.random.multivariate_normal(means[3], covariances[3], sizes[3]),
    )


def create_labes_for_points(sizes):
    """
    Create list of label for each class  using their sizes.
    :param sizes: Size of each label.
    :return: List of integer labels.
    """
    return np.concatenate((np.repeat(1, sizes[0]),
                           np.repeat(2, sizes[1]),
                           np.repeat(3, sizes[2]),
                           np.repeat(4, sizes[3])))


def write_to_the_file(x, y, D):
    """
    Write data to the file regardless of their dimension.
    :param x: Point data for each labeled class.
    :param y: Labels of the data set.
    :param D: Dimension of the data set.
    """

    # Creating format with .8 precision.
    data_set_format = "%1.8f," * D + "%d"
    data_set = [x[:, i] for i in range(D)]
    data_set.append(y)
    np.savetxt("data_set.csv", np.stack(tuple(data_set), axis=1), fmt=data_set_format)


def read_from_file():
    """
    Read data set from the file.
    :return: List of dimension, data set, labels, number of classes and total point count.
    """
    data_set = np.genfromtxt("data_set.csv", delimiter=",")

    D = data_set.shape[1] - 1
    N = data_set.shape[0]

    x = []

    for i in range(N):
        x.append([])
        for k in range(D):
            x[-1].append(data_set[i][k])

    y = data_set[:, D].astype(int)

    K = np.max(y)
    N = data_set.shape[0]

    return D, np.array(x), np.array(y), K, N


def plot_graph(points):
    """
    Plotting graph using matplotlib.
    :param points: Points to be drawn on the graph.
    """
    plt.figure(figsize=(8, 8))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(points[0][:, 0], points[0][:, 1], "r.", markersize=10)
    plt.plot(points[1][:, 0], points[1][:, 1], "g.", markersize=10)
    plt.plot(points[2][:, 0], points[2][:, 1], "b.", markersize=10)
    plt.plot(points[3][:, 0], points[3][:, 1], "m.", markersize=10)
    plt.show()


def calculate_sample_mean(x, y, K):
    """
    Calculates the sample means of given dataset.
    :param x: Data to be calculated.
    :param y: Labels of the data.
    :param K: Number of classes.
    :return: Mean of the given data.
    """
    return [np.mean(x[y == (c + 1)], axis=0) for c in range(K)]


def calculate_sample_covariance(sample_means, x, y, K):
    """
    Calculates the sample covariance of given dataset.
    :param sample_means: Sample means of the dataset.
    :param x: Data to be calculated.
    :param y: Labels of the data.
    :param K: Number of classes.
    :return: Covariance of the given data.
    """
    return [np.matmul((x[y == (c + 1)] - sample_means[c]).T, (x[y == (c + 1)] - sample_means[c]))
            / np.count_nonzero(y == (c + 1)) for c in range(K)]


def calculate_class_priors(y, K):
    """
    Calculates class priors of the dataset.
    :param y: Labels of the data.
    :param K: Number of classes.
    :return: Class priors of the given data.
    """
    return [np.mean(y == (c + 1)) for c in range(K)]


def create_w_values(sample_means, sample_covariances, class_priors, K, D):
    """
    Calculates the w values to be used while creating score function.
    :param sample_means: Sample means of the dataset.
    :param sample_covariances: Sample covariance of the dataset.
    :param class_priors: Class priors of the dataset.
    :param K: Number of classes.
    :param D: Dimension of dataset.
    :return: List of w values.
    """
    Wc = [-0.5 * linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), np.eye(D)) for c in range(K)]

    wc = [np.matmul(linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), np.eye(D)), sample_means[c]) for c in
          range(K)]

    wc0 = [-0.5 * np.matmul(
        np.matmul(sample_means[c].T, linalg.cho_solve(linalg.cho_factor(sample_covariances[c]), np.eye(D))),
        sample_means[c])
           - 0.5 * D * np.log(2 * math.pi)
           - 0.5 * np.log(np.linalg.det(sample_covariances[c]))
           + np.log(class_priors[c]) for c in range(K)]

    return Wc, wc, wc0


def create_score_function(Wc, wc, wc0, x, K, N):
    """
    Calculating score function using w values.
    :param x: Data to be calculated.
    :param K: Number of classes.
    :param N: Size of dataset.
    :return: List of score calculated functions.
    """
    return [np.stack([np.matmul(np.matmul(x[i], Wc[c]), x[i].T) + np.matmul(wc[c].T, x[i]) + wc0[c]
                      for c in range(K)]) for i in range(N)]


def find_maximum_score(score_function):
    """
    Find the maximum of the score function.
    :param score_function: Score function to be analyzed.
    :return: Maximum of the given score function.
    """
    return np.argmax(score_function, axis=1) + 1


def create_confusion_matrix(y_predicted, y):
    """
    Creates a confusion matrix using pandas.
    :param y_predicted: Predicted y values.
    :param y: Real y values.
    :return: Confusion matrix.
    """
    return pd.crosstab(np.array(y_predicted), y, rownames=['y_pred'], colnames=['y_truth'])


def create_desicion_boundaries(Wc, wc, wc0, x, y, y_pred, K):
    """
    Creates a decision boundaries.
    :param x: Data to be calculated.
    :param y: Real y values.
    :param y_pred: Predicted y values.
    :param K: Number of classes.
    """

    # Creating intervals of -8 to 8.
    x1_interval = np.linspace(-8, +8, 1601)
    x2_interval = np.linspace(-8, +8, 1601)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

    # Creating discriminant values.
    discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))

    # Filling the discriminant values.
    for c in range(K):
        discriminant_values[:, :, c] = Wc[c][0][0] * x1_grid ** 2 + Wc[c][1][1] * x2_grid ** 2 + \
                                       Wc[c][1][0] * x1_grid * x2_grid + wc[c][0] * x1_grid + wc[c][1] \
                                       * x2_grid + wc0[c]

    # Assigning discriminant values to the variables for further use.
    C1 = discriminant_values[:, :, 0]
    C2 = discriminant_values[:, :, 1]
    C3 = discriminant_values[:, :, 2]
    C4 = discriminant_values[:, :, 3]

    # Removing unnecessary parts of the parabolas.
    C1[(C1 < C2) & (C1 < C3)] = np.nan
    C2[(C2 < C1) & (C2 < C4)] = np.nan
    C3[(C3 < C1) & (C3 < C4)] = np.nan
    C4[(C4 < C1) & (C4 < C2)] = np.nan
    C4[(C4 < C1) & (C4 < C3)] = np.nan

    # Creating figure.
    plt.figure(figsize=(8, 8))

    # Plotting points.
    plt.plot(x[y == 1, 0], x[y == 1, 1], "r.", markersize=10)
    plt.plot(x[y == 2, 0], x[y == 2, 1], "g.", markersize=10)
    plt.plot(x[y == 3, 0], x[y == 3, 1], "b.", markersize=10)
    plt.plot(x[y == 4, 0], x[y == 4, 1], "m.", markersize=10)

    # Selecting wrongly identified points.
    plt.plot(x[y_pred != y, 0], x[y_pred != y, 1], "ko", markersize=12, fillstyle="none")

    # Plotting the regions of classes.
    plt.contour(x1_grid, x2_grid, C1 - C2, levels=0, colors="k")
    plt.contour(x1_grid, x2_grid, C1 - C3, levels=0, colors="k")
    plt.contour(x1_grid, x2_grid, C1 - C4, levels=0, colors="k")
    plt.contour(x1_grid, x2_grid, C2 - C4, levels=0, colors="k")
    plt.contour(x1_grid, x2_grid, C3 - C4, levels=0, colors="k")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()


if __name__ == "__main__":
    # Random seed to be set for creating consistency between each run.
    np.random.seed(521)
    main()
