def get_numpy_features_responses(dataframe, no_features):
    features = dataframe.iloc[:, 0:no_features].to_numpy()
    responses = dataframe.iloc[:, no_features].to_numpy()

    return features, responses


def create_datasets(features, responses):
    from numpy import float32
    from torch import tensor
    from torch.utils.data import TensorDataset

    features_T = tensor(features.astype(float32))
    responses_T = tensor(responses.astype(float32))

    return TensorDataset(features_T, responses_T)


def fit_model():
    pass


if __name__ == '__main__':
    from raw_data_handler import get_raw_data
    from sklearn.model_selection import train_test_split

# Get DataFrames for the training and test data.
    raw_dataframe, test_dataframe = get_raw_data('raw.csv',
                                                 'test.csv',
                                                 'lattice-physics-results/')

    no_features = 39
    raw_features, raw_responses = get_numpy_features_responses(raw_dataframe,
                                                               no_features)
    test_features, test_responses = get_numpy_features_responses(test_dataframe,
                                                                 no_features)

# Split raw data into training and validation sets.
    (training_features,
     validation_features,
     training_responses,
     validation_responses) = train_test_split(raw_features,
                                              raw_responses,
                                              test_size=0.1,
                                              random_state=1)

# Create DataSets for training, validation and test.
    training_dataset = create_datasets(training_features,
                                       training_responses)
    validation_dataset = create_datasets(validation_features,
                                         validation_responses)
    test_dataset = create_datasets(test_features,
                                   test_responses)

    fit_model()
