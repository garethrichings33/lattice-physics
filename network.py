from torch import nn


class LatticeNetwork(nn.Module):
    def __init__(self):
        super(LatticeNetwork, self).__init__()
        self.activation = nn.Tanh()
        self.linear1 = nn.Linear(39, 50)
        self.linear_out = nn.Linear(50, 1)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear_out(x)
        return x


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


def train_one_epoch(model, training_dataloader, optimiser, loss_fn):
    '''
    Function to train a single epoch
    '''
    running_loss = 0.
    for data in training_dataloader:
        features, responses = data
        optimiser.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions.squeeze(), responses)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    no_samples = training_dataloader.batch_size * len(training_dataloader)

    return running_loss/no_samples


def fit_model(training_dataloader, validation_dataloader):
    import torch

# Create model and define loss function and optimiser.
    model = LatticeNetwork()
    loss_fn = nn.MSELoss(reduction='sum')
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                weight_decay=0.,
                                momentum=0.)

    EPOCHS = 100
    training_loss_tracker = []
    validation_loss_tracker = []
    min_validation_loss = 1.e6

    for epoch in range(EPOCHS):
        model.train()
        training_loss = train_one_epoch(model,
                                        training_dataloader,
                                        optimiser,
                                        loss_fn)
        model.eval()
        print(training_loss)


if __name__ == '__main__':
    from raw_data_handler import get_raw_data
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

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
# Define DataLoaders
    batch_size = 20
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)

    fit_model(training_dataloader, validation_dataloader)
