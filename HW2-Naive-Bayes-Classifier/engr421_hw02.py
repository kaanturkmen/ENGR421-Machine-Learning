import numpy as np
import pandas as pd


def safelog(x):
    """
    Takes a safe logarithm, and help user not to deal with algebra related errors.
    :param x: Number which will take logarithm of.
    :return: Logarithm of the given number.
    """

    return np.log(x + 1e-100)


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

    # Calculate pcd values.
    pcd = calculate_pcd(x_tr, K, N)
    print("\nPcd:\n" + str(pcd))

    # Calculate class priors.
    class_priors = calculate_class_priors(labels, K)
    print("\nClass Priors:\n" + str(class_priors))

    # Calculate score functions of training data.
    score_training = calculate_score_function(x_tr, class_priors, K, P, R, pcd)

    # Taking maximum of the given score functions of training data.
    y_prediction_training = np.argmax(score_training, axis=1) + 1

    # Creating confusion matrix from training data.
    y_prediction_training_confusion_matrix = pd.crosstab(y_prediction_training, np.array(y_tr).reshape(R * K),
                                                         rownames=['y_pred'],
                                                         colnames=['y_truth'])

    # Printing the training data confusion matrix.
    print()
    print(y_prediction_training_confusion_matrix)

    # Calculate score functions of test data.
    score_test = calculate_score_function(x_tst, class_priors, K, P, U, pcd)

    # Taking maximum of the given score functions of test data.
    y_prediction_test = np.argmax(score_test, axis=1) + 1

    # Creating confusion matrix from test data.
    y_prediction_test_confusion_matrix = pd.crosstab(y_prediction_test, np.array(y_tst).reshape(U * K),
                                                     rownames=['y_pred'], colnames=['y_truth'])

    # Printing confusion matrix from test data.
    print()
    print(y_prediction_test_confusion_matrix)


def read_dataset():
    """
    Reads the csv file.
    :return: Image arrays of the data set.
    """
    return np.genfromtxt("hw02_data_set_images.csv", delimiter=",")


def read_labels():
    """
    Reads the csv file.
    :return: Label arrays of the data set.
    """
    return np.genfromtxt("hw02_data_set_labels.csv", delimiter=",").astype(int)


def calculate_pcd(x, K, N):
    """
    Calculates pcd values of the dataset.
    :param x: Dataset.
    :param K: Number of classes.
    :param N: Number of pixels.
    :return: List containing pcd values.
    """
    res = []
    for i in range(K):
        sub_array = []
        for j in range(N):
            sub_array.append(np.mean(x[i][:, j]))
        res.append(sub_array)

    return res


def calculate_score_function(x, class_priors, K, P, R, pcd):
    """
    Calculates the score functions.
    :param x: Data set.
    :param class_priors: Class priors.
    :param K: Number of classes.
    :param P: Total pixel count.
    :param R: Number of images.
    :param pcd: Pcd list.
    :return: List of score functions.
    """
    score = []

    # Four for loops are being used to calculate score function.
    for i in range(K):
        for c in range(K):
            for j in range(R):
                a = 0
                for k in range(P):
                    a += x[c][j][k] * safelog(pcd[i][k]) + (1 - x[c][j][k]) * safelog(1 - pcd[i][k])

                a += safelog(class_priors[c])
                score.append(a)

    # After I am calculating scores, I am converting it to a better form and it is being easier to deal with
    # the other kind of approaches. Then, I will use this to create confusion matrix.

    a = np.array(score).reshape(K, R * K)
    res = []

    for i in range(R * K):
        for j in range(K):
            res.append(a[j][i])

    return np.array(res).reshape(R * K, K)


def calculate_class_priors(labels, K):
    """
    Calculates class priors using the given labels.
    :param labels: Labels of the classes.
    :param K: Number of classes.
    :return: List of class priors.
    """
    return np.array([np.mean(labels == (c + 1)) for c in range(K)])


if __name__ == '__main__':
    main()

# %%
