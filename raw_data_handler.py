import numpy as np
import pandas as pd


def extract_dataframe_from_csv(filename):

    with open(filename, 'r') as file:
        dataframe = pd.read_csv(file, header=None, sep=r"\s+")

    return dataframe


def get_raw_data(training_filename, test_filename, path='./'):

    if path[-1] != '/':
        path = path + '/'

    training_dataframe = extract_dataframe_from_csv(
        f'{path}{training_filename}')
    test_dataframe = extract_dataframe_from_csv(f'{path}{test_filename}')

    return training_dataframe, test_dataframe


# def mean_response(responses):
#     total_response = 0.
#     for i in range(len(responses)):
#         total_response += responses[i]

#     return total_response/len(responses)


# def normalise_responses(training, test):

#     mean_training_response = mean_response(training)

#     normalised_training = []
#     for i in range(len(training)):
#         normalised_training.append(training[i]-mean_training_response)

#     normalised_test = []
#     for i in range(len(test)):
#         normalised_test.append(test[i]-mean_training_response)

#     return np.array(normalised_training), np.array(normalised_test)
