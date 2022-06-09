import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """
    Main method of the ML program.
    """
    # Reading data set and labels from the file.
    images = read_images_dataset()
    labels = read_labels_dataset()

    images_train, images_test = images[0:2000, :], images[2000:, :]
    labels_train, labels_test = labels[0:2000], labels[2000:]

    # Features, 28 * 28 = 784.
    F = images_train.shape[1]

    # Class count, 10.
    K = np.max(labels_train)

    # Means of the training set.
    mean_of_the_training_set = np.mean(images_train, axis=0)

    SW, SB = find_sw_and_sb(F, K, images_train, labels_train, mean_of_the_training_set)

    print("\n SW -> \n")
    print(SW[0:5, 0:5])

    print("\n SB -> \n")
    print(SB[0:5, 0:5])

    eigenvalues, eigenvectors = find_eigens(SW, SB)

    point_colors = np.array(
        ["#1f78b4",
         "#33a02c",
         "#e31a1c",
         "#ff7f00",
         "#6a3d9a",
         "#a6cee3",
         "#b2df8a",
         "#fb9a99",
         "#fdbf6f",
         "#cab2d6"])

    class_names = ["T-shirt / Top",
                   "Trouser",
                   "Pullover",
                   "Dress",
                   "Coat",
                   "Sandal",
                   "Shirt",
                   "Sneaker",
                   "Bag",
                   "Ankle Boot"]

    plot_figure(K, class_names, eigenvectors, images_train, labels_train, mean_of_the_training_set, point_colors)
    plot_figure(K, class_names, eigenvectors, images_test, labels_test, mean_of_the_training_set, point_colors)

    Z_train = np.matmul((images_train - mean_of_the_training_set), eigenvectors[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]])
    Z_test = np.matmul((images_test - mean_of_the_training_set), eigenvectors[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]])

    knn_train = calculate_knn(Z_train, Z_train, labels_train, 11)
    knn_test = calculate_knn(Z_test, Z_train, labels_train, 11)

    print(create_confusion_matrix(knn_train, labels_train, 'y_hat', 'y_train'))
    print("\n\n")
    print(create_confusion_matrix(knn_test, labels_test, 'y_hat', 'y_test'))


def plot_figure(K, class_names, eigenvectors, images_train, labels_train, mean_of_the_training_set, point_colors):
    """
    Plots data where different colors represents different classes.
    :param K: Number of classes in the dataset.
    :param class_names: Name of each class.
    :param eigenvectors: Eigenvectors of the set.
    :param images_train: Training set of images.
    :param labels_train: Training set of labels.
    :param mean_of_the_training_set: Mean of the training set.
    :param point_colors: Colors of the points.
    """
    Z = np.matmul((images_train - mean_of_the_training_set), eigenvectors[:, [0, 1]])
    plt.figure(figsize=(10, 10))
    for c in range(K):
        plt.plot(Z[labels_train == c + 1, 0], Z[labels_train == c + 1, 1], marker="o", markersize=4, linestyle="none",
                 color=point_colors[c])
    plt.legend(class_names, loc="upper left", markerscale=2)
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.xlabel("Component #1")
    plt.ylabel("Component #2")
    plt.show()


def find_eigens(SW, SB):
    """
    Finding eigenvalues and eigenvectors of the given SB and SW.
    :param SW: Scatter Within array.
    :param SB: Scatter Between array.
    :return: Eigenvalues and eigenvectors sorted in descending order.
    """
    A = np.dot(np.linalg.inv(SW), SB)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Getting rid of the imaginary part.
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    print("\n Largest 9 eigenvalues -> \n")
    print(eigenvalues[0:9])

    return eigenvalues, eigenvectors


def find_sw_and_sb(F, K, images_train, labels_train, mean_of_the_training_set):
    """
    Finding sw and sb values of the training set.
    :param F: Number of features, 28 * 28 = 784 px. in the dataset.
    :param K: Number of classes in the dataset.
    :param images_train: Training set of the images.
    :param labels_train: Labels set of the images.
    :param mean_of_the_training_set: Mean of the training image set.
    :return: SW and SB array.
    """

    # Declaring zero array for the SW and SB for the further use.
    SW = np.zeros((F, F))
    SB = np.zeros((F, F))

    # Starting the indexing from 1.
    for c in (ind + 1 for ind in range(K)):
        # Calculating x_i and m_c for the current class.
        x_i = images_train[labels_train == c]
        m_c = np.mean(x_i, axis=0)

        # Calculating SW from the formula in the PDF.
        SW += np.dot((x_i - m_c).T, (x_i - m_c))

        # Calculating SB from the formula in the PDF.
        n_c = x_i.shape[0]
        mean_diff = (m_c - mean_of_the_training_set).reshape(F, 1)
        SB += n_c * np.dot(mean_diff, mean_diff.T)

    return SW, SB


def calculate_knn(x, X, y, k):
    """
    Calculating knn matrix for each element.
    :param x: Set to be calculated.
    :param X: Set to be compared.
    :param y: Class labels.
    :param k: Knn setting.
    :return: Array that contains knn values.
    """
    result = []
    for elem in x:
        distances = [euclidean_distance(elem, x_train) for x_train in X]
        k_idx = np.argsort(distances)[:k]
        k_neighbor_labels = [y[i] for i in k_idx]
        values, counts = np.unique(k_neighbor_labels, return_counts=True)
        idx = np.argmax(counts)
        result.append(values[idx])
    return result


def euclidean_distance(a, b):
    """
    Calculating euclidean distance between two points.
    :param a: First point.
    :param b: Second point.
    :return: Float value of euclidean distance.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def create_confusion_matrix(y_predicted, y, row_names, col_names):
    """
    Creates a confusion matrix using pandas.
    :param y_predicted: Predicted y values.
    :param y: Real y values.
    :return: Confusion matrix.
    """
    return pd.crosstab(np.array(y_predicted), y, rownames=[row_names], colnames=[col_names])


def read_images_dataset():
    """
    Reads the csv file.
    :return: Train arrays of the data set.
    """
    return np.genfromtxt("hw07_data_set_images.csv", delimiter=",")


def read_labels_dataset():
    """
    Reads the csv file.
    :return: Test arrays of the data set.
    """
    return np.genfromtxt("hw07_data_set_labels.csv", delimiter=",").astype(int)


if __name__ == "__main__":
    main()

# %%
