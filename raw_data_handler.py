def extract_dataframe_from_csv(filename):
    import pandas as pd

    with open(filename, 'r') as file:
        dataframe = pd.read_csv(file)

    return dataframe


def get_data(training_filename, test_filename, path='./'):

    if path[-1] != '/':
        path = path + '/'

    training_dataframe = extract_dataframe_from_csv(
        f'{path}{training_filename}')
    test_dataframe = extract_dataframe_from_csv(f'{path}{test_filename}')

    return training_dataframe, test_dataframe
