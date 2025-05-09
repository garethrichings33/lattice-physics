from torch import nn


class LatticeNetwork(nn.Module):
    def __init__(self):
        super(LatticeNetwork, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear1 = nn.Linear(39, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear_out = nn.Linear(100, 1)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
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


def get_validation_loss(model, validation_dataloader, loss_fn):
    import torch

    running_vloss = 0.
    with torch.no_grad():
        for vdata in validation_dataloader:
            v_features, v_responses = vdata
            v_predictions = model(v_features)
            vloss = loss_fn(v_predictions.squeeze(), v_responses)
            running_vloss += vloss.item()

        no_samples = validation_dataloader.batch_size * \
            len(validation_dataloader)
        validation_loss = running_vloss/no_samples

    return validation_loss


def plot_losses(losses):
    '''
    Plot progress of training and validation losses against epoch number.
    '''
    from matplotlib import pyplot as plt

    plt.figure()
    epochs = []
    training_losses = []
    validation_losses = []
    for i in range(len(losses)):
        epoch, training_loss, validation_loss = losses[i]
        epochs.append(epoch)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    ax = plt.axes()
    ax.scatter(epochs, training_losses, marker='o', label='Training')
    ax.scatter(epochs, validation_losses, marker='x', label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend(loc='upper right')
    plt.show()


def fit_model(training_dataloader, validation_dataloader):
    import torch
    from math import log10

# Create model and define loss function and optimiser.
    model = LatticeNetwork()
    loss_fn = nn.MSELoss(reduction='sum')
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                weight_decay=0.,
                                momentum=0.)

    EPOCHS = 1001
    loss_tracker = []
    min_validation_loss = 1.e9

    for epoch in range(EPOCHS):
        model.train()
        training_loss = train_one_epoch(model,
                                        training_dataloader,
                                        optimiser,
                                        loss_fn)
        model.eval()
        validation_loss = get_validation_loss(model,
                                              validation_dataloader,
                                              loss_fn)
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            min_validation_loss_epoch = epoch

        print(f'Epoch: {epoch}, Training Loss: {training_loss:14.12f}, '
              f' Validation Loss: {validation_loss:14.12f}')

        if (epoch) % 50 == 0:
            loss_tracker.append((epoch,
                                 log10(training_loss),
                                 log10(validation_loss)))

    print(f'Minimum validation loss: {min_validation_loss:14.12f} '
          f'at epoch {min_validation_loss_epoch}')

# Plot progress of losses
    plot_losses(loss_tracker)


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
