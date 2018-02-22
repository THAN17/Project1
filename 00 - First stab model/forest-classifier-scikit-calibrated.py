# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import numpy as np
# File Paths
INPUT_PATH = "data/breast-cancer-wisconsin.data"
OUTPUT_PATH = "data/breast-cancer-wisconsin.csv"

# Headers
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]


def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    return data


def get_headers(dataset):
    """
    dataset headers
    :param dataset:
    :return:
    """
    return dataset.columns.values


def add_headers(dataset, headers):
    """
    Add the headers to the dataset
    :param dataset:
    :param headers:
    :return:
    """
    dataset.columns = headers
    return dataset


def data_file_to_csv():
    """

    :return:
    """

    # Headers
    headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
               "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
               "CancerType"]
    # Load the dataset into Pandas data frame
    dataset = read_data(INPUT_PATH)
    # Add the headers to the loaded dataset
    dataset = add_headers(dataset, headers)
    # Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File saved ...!")


def split_dataset(dataset, train_percentage, valid_percentage,  feature_headers, target_header):
    """
    Split the dataset with train_percentage and valid_percentage
    :param dataset:
    :param train_percentage:
    :param valid_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, valid_x, test_x, train_y, valid_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage + valid_percentage,
                                                        test_size=1-(train_percentage + valid_percentage))

    valid_x = train_x[int(np.ceil(train_percentage * len(dataset))):]
    valid_y = train_y[int(np.ceil(train_percentage * len(dataset))):]

    return train_x, valid_x, test_x, train_y, valid_y, test_y


def handle_missing_values(dataset, missing_values_header, missing_label):
    """
    Filter missing values from the dataset
    :param dataset:
    :param missing_values_header:
    :param missing_label:
    :return:
    """

    return dataset[dataset[missing_values_header] != missing_label]


def random_forest_classifier(train_x, train_y, valid_x, valid_y):
    """
    To train the random forest classifier with features and target data
    :param train_x:
    :param train_y:
    :param valid_x:
    :param valid_y:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    sig_clf.fit(valid_x, valid_y)
    return clf, sig_clf


def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print(dataset.describe())


def main():
    """
    Main function
    :return:
    """
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    # Filter missing values
    dataset = handle_missing_values(dataset, HEADERS[6], '?')
    train_x, valid_x, test_x, train_y, valid_y, test_y = split_dataset(dataset, 0.5, 0.25, HEADERS[1:-1], HEADERS[-1])

    # Train and Test dataset size details
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    # Create random forest classifier instance
    original_model, calibrated_model = random_forest_classifier(train_x, train_y, valid_x, valid_y)
    print("Trained model :: ", calibrated_model)

    predictions = calibrated_model.predict(test_x)

    print("Train Accuracy :: ", accuracy_score(train_y, calibrated_model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Confusion matrix \n", confusion_matrix(test_y, predictions))

    clf_probs = original_model.predict_proba(test_x)
    score = log_loss(test_y, clf_probs)
    sig_clf_probs = calibrated_model.predict_proba(test_x)
    sig_score = log_loss(test_y, sig_clf_probs)

    print()
    print("Log-loss of")
    print(" * uncalibrated classifier trained on 50%% datapoints: %.3f "
          % score)
    print(" * classifier trained on 50%% datapoints and calibrated on "
          "25%% datapoint: %.3f" % sig_score)
    print()
    for i in range(0, 5):
        print("Actual outcome :: {} and Predicted outcome :: {} and Predicted probability :: {}".
              format(list(test_y)[i], predictions[i], sig_clf_probs[i][0]))


if __name__ == "__main__":
    main()
