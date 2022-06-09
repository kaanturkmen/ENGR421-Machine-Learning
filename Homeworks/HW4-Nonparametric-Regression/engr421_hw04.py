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

    # Defining given parameters in the PDF.
    bin_width = 0.1
    origin = 0.0
    bin_width_kernel = 0.02

    # Obtaining left_borders, p_hat and right borders for plotting the figure.
    left_borders, p_hat, right_borders = calculate_p_hat(bin_width, max(x_train), origin, x_train, y_train)

    # Draw regressogram for the train and test.
    plot_regressogram_figure(x_train, y_train, left_borders, p_hat, right_borders, 'b.', 'training')
    plot_regressogram_figure(x_test, y_test, left_borders, p_hat, right_borders, 'r.', 'test')

    # Print RMSE value for regressogram.
    calculate_regressogram_rmse(bin_width, left_borders, p_hat, right_borders, x_test, y_test)

    # Creating data interval.
    data_interval = np.linspace(origin, max(x_train), 1201)

    # Calculating RMS values.
    rms_vals = rms_values(data_interval, bin_width, x_train, y_train)

    # Plotting RMS figures.
    plot_rms_figure(data_interval, rms_vals, x_train, y_train, 'b.', 'training')
    plot_rms_figure(data_interval, rms_vals, x_test, y_test, 'r.', 'test')

    # Calculating mean smoother's RMSE value.
    calculate_ms_rmse(bin_width, data_interval, rms_vals, x_test, y_test)

    # Calculating kernel values.
    kernel_vals = calculate_ks(bin_width_kernel, data_interval, x_train, y_train)

    # Plotting kernel smoother figures.
    plot_ks(data_interval, kernel_vals, x_train, y_train, 'b.', 'training')
    plot_ks(data_interval, kernel_vals, x_test, y_test, 'r.', 'test')

    # Calculating kernel smoother's RMSE value.
    calculate_rmse_ks(bin_width_kernel, data_interval, kernel_vals, x_test, y_test)


def calculate_rmse_ks(bin_width_kernel, data_interval, kernel_vals, x_test, y_test):
    """
    Calculating kernel smoother's RMSE value.
    :param bin_width_kernel: Given specific bin width for kernel.
    :param data_interval: Created data interval.
    :param kernel_vals: Kernel values.
    :param x_test: Test version of x.
    :param y_test: Test version of y.
    """
    summed_vals = 0
    for each_kernel in range(len(kernel_vals) - 1):
        for each_data in range(len(x_test)):
            if data_interval[each_kernel] < x_test[each_data]:
                if x_test[each_data] <= data_interval[each_kernel + 1]:
                    summed_vals += np.square(y_test[each_data] - kernel_vals[each_kernel])
    result = np.sqrt(summed_vals / len(x_test))
    print("Kernel Smoother => RMSE is " + str(result) + " when h is " + str(bin_width_kernel))


def plot_ks(data_interval, kernel_vals, x_train, y_train, color, label):
    """
    Plotting the figure KS is RMS applied to it.
    :param data_interval: Created data interval.
    :param kernel_vals: Kernel values.
    :param x_train: Train version of x.
    :param y_train: Train version of y.
    :param color: Color of the data points.
    :param label: Label of the legend.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Time (sec)")
    plt.ylabel("Singal (millivolt)")
    plt.plot(x_train, y_train, color, label=label, markersize=3)
    plt.plot(data_interval, kernel_vals, "k-", markersize=3)
    plt.xlabel("Time (sec)")
    plt.ylabel("Singal (millivolt)")
    plt.legend(loc="upper right")
    plt.ylim([-1, 2])
    plt.show()


def calculate_ks(bin_width_kernel, data_interval, x_train, y_train):
    """
    Calculates kernel smoother and returns array of calculated values.
    :param bin_width_kernel: Given kernel specific bin width.
    :param data_interval: Created data interval.
    :param x_train: Train version of x.
    :param y_train: Train version of y.
    :return: Array of kernel values.
    """
    kernel_values = []
    for each_interval in data_interval:
        summed_vals = 0
        total_sum = 0
        for j in range(len(x_train)):
            summed_vals += k_func((each_interval - x_train[j]) / bin_width_kernel) * y_train[j]
            total_sum += k_func((each_interval - x_train[j]) / bin_width_kernel)
        kernel_values.append(summed_vals / total_sum)
    return kernel_values


def k_func(u):
    """
    General k function.
    :param u: Parameter to be applied K function to.
    :return: Result of the k function applied to the u parameter.
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-np.square(u) / 2)


def calculate_ms_rmse(bin_width, data_interval, rms_vals, x_test, y_test):
    """
    Calculating mean smoother's RMSE value.
    :param bin_width: Given Bin Width.
    :param data_interval: Created data interval.
    :param rms_vals: Calculated RMS values.
    :param x_test: Test version of x.
    :param y_test: Test version of y.
    """
    summed_val = 0
    for each_rms in range(len(rms_vals) - 1):
        for each_data in range(len(x_test)):
            if data_interval[each_rms] < x_test[each_data]:
                if x_test[each_data] <= data_interval[each_rms + 1]:
                    summed_val += np.square(y_test[each_data] - rms_vals[each_rms])
    result = np.sqrt(summed_val / len(x_test))
    print("Running Mean Smoother => RMSE is " + str(result) + " when h is " + str(bin_width))


def plot_rms_figure(data_interval, rms_vals, x_train, y_train, color, label):
    """
    Plotting the figure which is RMS applied to it.
    :param data_interval: Created data interval.
    :param rms_vals: Calculated RMS values.
    :param x_train: Train version of x.
    :param y_train: Train version of y.
    :param color: Color of the data points.
    :param label: Label of the legend.
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Time (sec)")
    plt.ylabel("Singal (millivolt)")
    plt.plot(x_train, y_train, color, label=label, markersize=3)
    plt.plot(data_interval, rms_vals, "k-", markersize=3)
    plt.ylim([-1, 2])
    plt.legend(loc="upper right")
    plt.show()


def rms_values(data_interval, bin_width, x_train, y_train):
    """
    Calculating RMS values.
    :param data_interval: Created data interval.
    :param bin_width: Given Bin Width.
    :param x_train: Train version of x.
    :param y_train: Train version of y.
    :return: Array containing RMS values.
    """
    rms_value_array = []
    for each_interval in data_interval:
        summed_val = 0
        count = 0

        for each_data in range(len(x_train)):
            if each_interval - bin_width / 2 < x_train[each_data]:
                if x_train[each_data] <= each_interval + bin_width / 2:
                    summed_val += y_train[each_data]
                    count += 1

        # Checking mechanism for division by zero.
        if count != 0:
            rms_value_array.append(summed_val / count)
            continue

        rms_value_array.append(0)
    return rms_value_array


def calculate_regressogram_rmse(bin_width, left_borders, p_hat, right_borders, x_test, y_test):
    """
    Printing RMSE value for regressogram calculation.
    :param bin_width: Given Bin Width.
    :param left_borders: Created left borders.
    :param p_hat: Calculated p_hat value.
    :param right_borders: Created right borders.
    :param x_test: Test version of x.
    :param y_test: Test version of y.
    """
    main_sum = 0
    for border_index in range(len(left_borders)):
        for test_index in range(len(x_test)):
            if left_borders[border_index] < x_test[test_index]:
                if x_test[test_index] <= right_borders[border_index]:
                    main_sum += np.square(y_test[test_index] - p_hat[border_index])

    result = np.sqrt(main_sum / len(x_test))
    print("Regressogram => RMSE is " + str(result) + " when h is " + str(bin_width))


def plot_regressogram_figure(x, y, left_borders, p_hat, right_borders, color, label):
    """
    Plotting regressogram figure.
    :param x: Array which contains x values.
    :param y: Array which contains y values.
    :param left_borders: Created left borders.
    :param p_hat: Calculated p_hat values
    :param right_borders: Created right borders.
    :param color: Color of the data points.
    :param label: Label of the legend.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color, label=label, markersize=10)
    for i in range(len(left_borders)):
        plt.plot([left_borders[i], right_borders[i]], [p_hat[i], p_hat[i]], "k-")
    for i in range(len(right_borders) - 1):
        plt.plot([right_borders[i], right_borders[i]], [p_hat[i], p_hat[i + 1]], "k-")
    plt.xlabel("Time (sec)")
    plt.ylabel("Singal (millivolt)")
    plt.legend(loc="upper right")
    plt.ylim([-1, 2])
    plt.show()
    return p_hat


def calculate_p_hat(bin_width, maximum_value, origin, x, y):
    """
    Calculating p_hat values.
    :param bin_width: Given bin width.
    :param maximum_value: Maximum value of the set.
    :param origin: Given origin parameter.
    :param x: X set.
    :param y: Y set.
    :return: Left_borders, p_hat and right borders variables.
    """
    left_borders, right_borders = np.arange(origin, maximum_value, bin_width), \
                                  np.arange(origin + bin_width, maximum_value + bin_width, bin_width)
    p_hat = []
    for i in range(len(left_borders)):
        p_hat.append(np.sum(((x > left_borders[i]) & (x <= right_borders[i])) * y) / np.sum(
            ((x > left_borders[i]) & (x <= right_borders[i]))))
    return left_borders, np.array(p_hat), right_borders


def read_train_dataset():
    """
    Reads the csv file.
    :return: Train arrays of the data set.
    """
    return np.genfromtxt("hw04_data_set_train.csv", delimiter=",")


def read_test_dataset():
    """
    Reads the csv file.
    :return: Test arrays of the data set.
    """
    return np.genfromtxt("hw04_data_set_test.csv", delimiter=",")


if __name__ == "__main__":
    main()
