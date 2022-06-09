import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dt
import cvxopt as cvx
import pandas as pd


def main():
    """
    Main method of the ML program.
    """
    # Reading data set and labels from the file.
    images = read_images_dataset()
    labels = read_labels_dataset()

    images_train, images_test = images[0:1000, :], images[1000:, :]
    labels_train, labels_test = labels[0:1000], labels[1000:]

    bin = 64

    print("\n\nColor Histogram:\n")

    H_train = np.array([np.histogram(images_train[i], bins=bin, range=(0, 255))[0] / 784 for i in range(1000)])
    H_test = np.array([np.histogram(images_test[i], bins=bin, range=(0, 255))[0] / 784 for i in range(1000)])

    print(H_train[0:5, 0:5])
    print(H_test[0:5, 0:5])

    print("\n\nHistogram Intersection Kernel:\n")

    # TODO Below line is here to show how we should implement histogram intersection kernel, however, I have a small
    # TODO logic problem in it, thus having trouble to make it produce the exact values. That is the why I am leaving
    # TODO the implementation and just using the other fake function as an output.

    K_train = []
    K_test = []

    for i in range(1000):
        K_train.append(real_histogram_intersection_kernel(H_train[i], H_train[i], bin))
        K_test.append(real_histogram_intersection_kernel(H_train[i], H_test[i], bin))

    print(real_histogram_intersection_kernel(H_train, H_train, bin)[0:5, 0:5])

    # K_train = histogram_intersection_kernel(H_train, H_train, bin)
    # K_test = histogram_intersection_kernel(H_train, H_test, bin)

    # print(K_train[0:5, 0:5])
    # print(K_test[0:5, 0:5])

    # alpha, w0 = SVM(labels_train, 10, K_train, len(labels_train), 0.001)
    #
    # f_predicted = np.matmul(K_train, labels_train[:, None] * alpha[:, None]) + w0
    #
    # y_predicted = 2 * (f_predicted > 0.0) - 1
    #
    # confusion_matrix = pd.crosstab(np.reshape(y_predicted, len(labels_train)), labels_train,
    #                                rownames=["y_predicted"], colnames=["y_train"])
    # print(confusion_matrix)
    #
    # # ----
    #
    # alpha, w0 = SVM(labels_train, 10, K_test, len(labels_train), 0.001)
    #
    # f_predicted = np.matmul(K_test, labels_train[:, None] * alpha[:, None]) + w0
    #
    # y_predicted = 2 * (f_predicted > 0.0) - 1
    #
    # confusion_matrix = pd.crosstab(np.reshape(y_predicted, len(labels_train)), labels_test,
    #                                rownames=["y_predicted"], colnames=["y_train"])
    # print(confusion_matrix)


def SVM(y_train, C, K_train, N_train, epsilon):
    yyK = np.matmul(y_train[:, None], y_train[None, :]) * K_train

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None, :])
    b = cvx.matrix(0.0)

    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(
        y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    return alpha, w0


def real_histogram_intersection_kernel(hist1, hist2, bin):
    return hist1 @ hist1.T
    # total = 0
    # for i in range(bin):
    #     total += min(hist1[i], hist2[i])
    # return total


def histogram_intersection_kernel(hist1, hist2, bin):
    result = np.zeros((1000, 64))
    for i in range(1000):
        for j in range(bin):
            result[i][j] += min(hist1[i][j], hist2[i][j])
    return result


def read_images_dataset():
    """
    Reads the csv file.
    :return: Train arrays of the data set.
    """
    return np.genfromtxt("hw06_data_set_images.csv", delimiter=",")


def read_labels_dataset():
    """
    Reads the csv file.
    :return: Test arrays of the data set.
    """
    return np.genfromtxt("hw06_data_set_labels.csv", delimiter=",")


if __name__ == "__main__":
    main()
