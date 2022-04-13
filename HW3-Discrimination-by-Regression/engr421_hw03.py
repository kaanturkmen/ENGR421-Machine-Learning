import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """
    Main method of the ML program.
    """

    # Reading data set and labels from the file.
    data_set = read_dataset()
    labels = read_labels()

    # Test set image variable.
    U = 14

    # Number of images from each class.
    R = 25

    # Number of classes.
    K = 5

    # Number of pixels on each photo.
    P = 20 * 16

    x_tr, y_tr = [], []
    x_tst, y_tst = [], []

    # My division algorithm will perform as following:
    # We have an array with 5 elements: A, B, C, D and E.
    # Then we have 25 photos from each of the elements.
    # In every photo we have array with 320 elements.
    # For instance if we want to go into C letters 21th photo's 123rd pixel,
    # x_tr[2][20][122] will give us the value of the pixel.

    for i in range(K):
        x_tr.append(data_set[i * (U + R): R + (i * (U + R))])
        y_tr.append(labels[i * (U + R): R + (i * (U + R))])

        x_tst.append(data_set[R + (i * (U + R)): (U + R) + (i * (U + R))])
        y_tst.append(labels[R + (i * (U + R)): (U + R) + (i * (U + R))])

    # Number of pixes (i.e. 20 x 16 = 320 in total.)
    N = len(x_tr[0][0])

    # Converting my array into into 125 x 320, 70 x 320 version to ease the calculations.
    x_tr_e = np.array(x_tr).reshape(R * K, P)
    x_tst_e = np.array(x_tst).reshape(U * K, P)

    # Getting rid of from numpy arrays inside functions.
    # Will convert it to the pure lists, then will apply numpy if needed.
    y_tr_e = []
    y_tst_e = []

    for elem in y_tr:
        y_tr_e.append(elem.tolist())

    for elem in y_tst:
        y_tst_e.append(elem.tolist())

    y_tr_c = np.zeros((R * K, 5)).astype(int)
    y_tr_c[range(R * K), np.array(y_tr_e).reshape(R * K, ) - 1] = 1

    # Given Eta and Epsilon values.
    eta = 0.001
    epsilon = 0.001

    # Getting random uniformed w and w0 values, will improve in further steps.
    w = np.random.uniform(low=-0.01, high=0.01, size=(P, K))
    w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

    # Improving values by sigmoid function.
    w, w0, iteration, objective_values, Y_predicted = improve_w_and_w0(w, w0, eta, epsilon, x_tr_e, y_tr_c, K)
    print(w)
    print(w0)

    # Plotting graph.
    plot_graph(iteration, objective_values)

    # Creating confusion matrix from the training set.
    y_pred = find_maximum_function(Y_predicted)
    first_cm = create_confusion_matrix(y_pred, np.array(y_tr_e).reshape(R * K, ))
    print(first_cm)

    # Creating confusion matrix from the test set.
    y_tst_pred = sigmoid(x_tst_e, w, w0)
    second_y_pred = find_maximum_function(y_tst_pred)
    second_cm = create_confusion_matrix(second_y_pred, np.array(y_tst_e).reshape(U * K, ))
    print(second_cm)


def read_dataset():
    """
    Reads the csv file.
    :return: Image arrays of the data set.
    """
    return np.genfromtxt("hw03_data_set_images.csv", delimiter=",")


def read_labels():
    """
    Reads the csv file.
    :return: Label arrays of the data set.
    """
    return np.genfromtxt("hw03_data_set_labels.csv", delimiter=",").astype(int)


def sigmoid(X, w, w0):
    """
    Applies sigmoid function by given parameters.
    :param X: Data set.
    :param w: W value.
    :param w0: w0 value.
    :return:
    """
    return 1 / (1 + np.exp(-(np.matmul(X, w) + w0)))


def gradient_W(X, Y_truth, Y_predicted, K):
    """
    Calculates gradient W.
    :param X: Data set.
    :param Y_truth: True labels.
    :param Y_predicted: Predicted labels.
    :param K: Class count.
    :return: W value.
    """
    return (np.asarray(
        [-np.matmul(((Y_truth[:, c] - Y_predicted[:, c]) * Y_predicted[:, c] * (1 - Y_predicted[:, c])), X) for c in
         range(K)]).transpose())


def gradient_w0(Y_truth, Y_predicted):
    """
    Calculates gradient w0.
    :param Y_truth: True labels.
    :param Y_predicted: Predicted labels.
    :return: w0 value.
    """
    return -np.sum((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted), axis=0)


def improve_w_and_w0(W, w0, eta, epsilon, X, Y_truth, K):
    """
    Improves W and w0 values.
    :param W: Random W value.
    :param w0: Random w0 value.
    :param eta: Given eta value.
    :param epsilon: Given epsilon value.
    :param X: Data set.
    :param Y_truth: Labels.
    :param K: Class count.
    :return: W and w0 values, iteration count, objective values and Y_predicted.
    """
    iteration = 1
    objective_values = []
    while True:
        Y_predicted = sigmoid(X, W, w0)

        objective_values = np.append(objective_values, 0.5 * np.sum(np.square(Y_truth - Y_predicted)))

        W_old = W
        w0_old = w0

        W = W - eta * gradient_W(X, Y_truth, Y_predicted, K)
        w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

        if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
            break

        iteration = iteration + 1

    return W, w0, iteration, objective_values, Y_predicted


def plot_graph(iteration, objective_values):
    """
    Plots graph with given iteration and objective values.
    :param iteration: Iteration count from improving w and w0s.
    :param objective_values: Objective values from improving w and w0s.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, iteration + 1), objective_values, "k-")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()


def find_maximum_function(function):
    """
    Find the maximum of the function.
    :param function: function to be analyzed.
    :return: Maximum of the given function.
    """
    return np.argmax(function, axis=1) + 1


def create_confusion_matrix(y_predicted, y):
    """
    Creates a confusion matrix using pandas.
    :param y_predicted: Predicted y values.
    :param y: Real y values.
    :return: Confusion matrix.
    """
    return pd.crosstab(np.array(y_predicted), y, rownames=['y_pred'], colnames=['y_truth'])


if __name__ == "__main__":
    # Random seed to be set for creating consistency between each run.
    np.random.seed(521)
    main()

# %%
