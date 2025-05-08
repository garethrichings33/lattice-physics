def fit_model():
    pass


if __name__ == '__main__':
    from raw_data_handler import get_data

    training_dataframe, test_dataframe = get_data('raw.csv',
                                                  'test.csv',
                                                  'lattice-physics-results/')
    fit_model()
