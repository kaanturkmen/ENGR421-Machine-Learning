import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Main method of the ML program.
    """

    # Reading data set and labels from the file.
    train = read_train_dataset()
    test = read_test_dataset()

    # Dividing train and test sets.
    x_train, y_train = train[:, 0], train[:, 1]
    x_test, y_test = test[:, 0], test[:, 1]

    # Setting pre pruning parameter to the given value.
    P = 30

    # Getting minimum and maximum value of the training set.
    minimum_value = min(x_train)
    maximum_value = max(x_train)

    print(minimum_value, maximum_value)

    # Learning tree using the pre pruning parameter, x and y of the training data.
    node_splits = learn_dtree(P, x_train, y_train)

    # Calculating p_hat value, as well as right and left borders using training data.
    right_borders, left_borders, p_hat = calculate_p_hat(minimum_value, maximum_value, node_splits, x_train, y_train)

    # Plotting figures.
    plot_figure(right_borders, left_borders, p_hat, x_train, y_train, 'b.', 'training')
    plot_figure(right_borders, left_borders, p_hat, x_test, y_test, 'r.', 'test')

    # Printing RMSE values.
    print(
        "RMSE on training set is " + str(calculateRMSE(right_borders, left_borders, x_train.shape[0], x_train, y_train,
                                                       p_hat)) + " when P is " + str(P) + ".")

    print("RMSE on test set is " + str(calculateRMSE(right_borders, left_borders, x_test.shape[0], x_test, y_test,
                                                     p_hat)) + " when P is " + str(P) + ".")

    # Plotting comparison figure.
    plot_comparison_figure(maximum_value, minimum_value, x_test, x_train, y_test, y_train)


def plot_comparison_figure(maximum_value, minimum_value, x_test, x_train, y_test, y_train):
    """
    Plotting comparison figure.
    :param maximum_value: Maximum value which is generated by the x_train set.
    :param minimum_value: Minimum value which is generated by the x_train set.
    :param x_test: Test set of X.
    :param x_train: Train set of X.
    :param y_test: Test set of Y.
    :param y_train: Train set of Y.
    """
    training_p, test_p, train_rmse, rmse = [], [], [], []

    training_set_size = len(x_train)

    for current_p_parameter in range(5, 55, 5):
        node_indices, is_terminal, need_split = {}, {}, {}

        node_indices[1] = np.array(range(training_set_size))
        is_terminal[1] = False
        need_split[1] = True
        node_splits = learn_dtree(current_p_parameter, x_train, y_train)

        left_borders = np.append(minimum_value, np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])))
        right_borders = np.append(np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])), maximum_value)

        p_hat = calculate_p_hat(minimum_value, maximum_value, node_splits, x_train, y_train)[2]

        rmse_regress_train = calculateRMSE(right_borders, left_borders, x_train.shape[0], x_train, y_train,
                                           p_hat)
        training_p.append(current_p_parameter)
        train_rmse.append(rmse_regress_train)

        rmse_val = calculateRMSE(right_borders, left_borders, x_test.shape[0], x_test, y_test,
                                 p_hat)
        test_p.append(current_p_parameter)
        rmse.append(rmse_val)
    plt.figure(figsize=(10, 8))
    plt.plot(training_p, train_rmse, marker=".", markersize=10, linestyle="-", color="b", label='train')
    plt.plot(test_p, rmse, marker=".", markersize=10, linestyle="-", color="r", label='test')
    plt.xlabel("Pre-pruning size (P)")
    plt.ylabel("RMSE")
    plt.legend(['training', 'test'])
    plt.show()


def calculateRMSE(right_borders, left_borders, x_train_size, x, y, p_hat):
    """
    Calculating RMSE values with given parameters.
    :param right_borders: Right borders generated by the maximum values.
    :param left_borders: Left borders generated by the minimum values.
    :param x_train_size: Size of x_train.
    :param x: X set.
    :param y: Y set.
    :param p_hat: P_hat values generated by train set of X.
    :return RMSE result.
    """
    summed_total = 0
    for point_index in range(x_train_size):
        for each_lb in range(len(left_borders)):
            if (left_borders[each_lb] < x[point_index]) and (x[point_index] <= right_borders[each_lb]):
                summed_total += (y[point_index] - p_hat[each_lb]) ** 2
    rmse_val = np.sqrt(summed_total / x_train_size)
    return rmse_val


def calculate_p_hat(minimum_value, maximum_value, node_splits, x, y):
    """
    Calculates p_hat values for the
    :param maximum_value: Maximum value which is generated by the x_train set.
    :param minimum_value: Minimum value which is generated by the x_train set.
    :param node_splits: Node list which is to be splitted.
    :param x: X set.
    :param y: Y set.
    :return: Right borders, left borders and the p_hat value.
    """
    left_borders = np.append(minimum_value, np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])))
    right_borders = np.append(np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])), maximum_value)

    p_hat = np.asarray([np.sum(((left_borders[b] < x) & (x <= right_borders[b])) * y) for b in
                        range(len(left_borders))]) / np.asarray(
        [np.sum((left_borders[b] < x) & (x <= right_borders[b])) for b in range(len(left_borders))])

    return right_borders, left_borders, p_hat


def plot_figure(right_borders, left_borders, p_hat, x, y, color, label):
    """
    Plotting figure of the given sets.
    :param right_borders: Right borders generated by the maximum values.
    :param left_borders: Left borders generated by the minimum values.
    :param p_hat: P_hat values generated by train set of X.
    :param x: X set.
    :param y: Y set.
    :param color: Color of the data points.
    :param label: Label of the legend.
    """
    plt.figure(figsize=(10, 8))
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.plot(x, y, color, label=label, markersize=10)
    plt.ylim([-1, 2])
    plt.legend(loc="upper right")
    for b in range(len(left_borders)):
        plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
    for b in range(len(left_borders) - 1):
        plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")
    plt.show()


def learn_dtree(P, x_train, y_train):
    """
    Learning decision tree.
    :param P: Pre pruning parameter.
    :param x_train: Train set of X.
    :param y_train: Train set of Y.
    :return:
    """

    node_indices, is_terminal, need_split, node_splits = {}, {}, {}, {}

    node_indices[1], is_terminal[1], need_split[1] = np.array(range(len(x_train))), False, True

    while True:
        split_nodes = [key for key, value in need_split.items() if value]

        if len(split_nodes) == 0:
            break

        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if data_indices.shape[0] <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):
                    left = np.sum((x_train[data_indices] < split_positions[s]) * y_train[data_indices]) / np.sum(
                        (x_train[data_indices] < split_positions[s]))
                    right = np.sum((x_train[data_indices] >= split_positions[s]) * y_train[data_indices]) / np.sum(
                        (x_train[data_indices] >= split_positions[s]))
                    split_scores[s] = 1 / len(data_indices) * (np.sum(
                        (y_train[data_indices] - np.repeat(left, data_indices.shape[0], axis=0)) ** 2 * (
                                x_train[data_indices] < split_positions[s])) + np.sum(
                        (y_train[data_indices] - np.repeat(right, data_indices.shape[0], axis=0)) ** 2 * (
                                x_train[data_indices] >= split_positions[s])))
                best_splits = split_positions[np.argmin(split_scores)]

                node_splits[split_node] = best_splits

                left_indices = data_indices[x_train[data_indices] < best_splits]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                right_indices = data_indices[x_train[data_indices] >= best_splits]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    return node_splits


def safelog2(x):
    """
    Safe log mechanism to be used in ML applications.
    :param x: Value to be safe logged of 2.
    :return: Result of the logarithm.
    """
    if x == 0:
        return 0
    else:
        return np.log2(x)


def read_train_dataset():
    """
    Reads the csv file.
    :return: Train arrays of the data set.
    """
    return np.genfromtxt("hw05_data_set_train.csv", delimiter=",")


def read_test_dataset():
    """
    Reads the csv file.
    :return: Test arrays of the data set.
    """
    return np.genfromtxt("hw05_data_set_test.csv", delimiter=",")


if __name__ == "__main__":
    main()

# %%
