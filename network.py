from math import log10
from numpy import float32
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
from raw_data_handler import get_raw_data


class LatticeNetwork(nn.Module):
    def __init__(self):
        super(LatticeNetwork, self).__init__()
        self.activation = nn.ReLU()
        self.dropout_in = nn.Dropout(0.1)
        self.linear1 = nn.Linear(39, 200)
        self.dropout1 = nn.Dropout(0.1)
        # nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.1)
        # nn.init.xavier_uniform_(self.linear2.weight)
        # self.dropout3 = nn.Dropout(0.1)
        # self.linear3 = nn.Linear(100, 100)
        self.linear_out = nn.Linear(200, 1)
        # nn.init.xavier_uniform_(self.linear_out.weight)
        # nn.init.uniform_(self.linear_out.bias, a=0.)

    def forward(self, x):
        # x = self.dropout_in(x)
        x = self.activation(self.linear1(x))
        # x = self.dropout1(x)
        x = self.activation(self.linear2(x))
        # x = self.dropout2(x)
        # x = self.activation(self.linear3(x))
        # x = self.dropout3(x)
        # x = self.activation(self.linear_out(x))
        x = self.linear_out(x)
        return x


def get_numpy_features_responses(dataframe, no_features):
    features = dataframe.iloc[:, 0:no_features].to_numpy()
    responses = dataframe.iloc[:, no_features].to_numpy()

    return features, responses


def create_datasets(features, responses):

    features_T = tensor(features.astype(float32))
    responses_T = tensor(responses.astype(float32))

    return TensorDataset(features_T, responses_T)


def train_one_epoch(model, training_dataloader, optimiser, loss_func):
    '''
    Function to train a single epoch
    '''
    model.train()
    running_loss = 0.
    for data in training_dataloader:
        features, responses = data
        optimiser.zero_grad()
        predictions = model(features)
        loss = loss_func(predictions.squeeze(), responses)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    return running_loss/len(training_dataloader.dataset)


def get_validation_loss(model, validation_dataloader, loss_fn):

    model.eval()
    running_vloss = 0.
    with torch.no_grad():
        for vdata in validation_dataloader:
            v_features, v_responses = vdata
            v_predictions = model(v_features)
            vloss = loss_fn(v_predictions.squeeze(), v_responses)
            running_vloss += vloss.item()

        validation_loss = running_vloss/len(validation_dataloader.dataset)

    return validation_loss


def plot_losses(losses, loss_type=''):
    '''
    Plot progress of training and validation losses against epoch number.
    '''

    epochs = []
    training_losses = []
    validation_losses = []
    for i in range(1, len(losses)):
        epoch, training_loss, validation_loss = losses[i]
        epochs.append(epoch)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    plt.figure()
    ax = plt.axes()
    ax.scatter(epochs, training_losses, marker='o', label='Training')
    ax.scatter(epochs, validation_losses, marker='x', label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_type} Loss')
    plt.legend(loc='upper right')
    plt.show()


def plot_responses(model, dataloader):

    data = DataLoader(dataloader.dataset,
                      shuffle=False,
                      batch_size=1)

    with torch.no_grad():
        response_list = []
        prediction_list = []
        index_list = []
        for i, datum in enumerate(data):
            features, response = datum
            prediction = model(features)
            response_list.append(response)
            prediction_list.append(prediction)
            index_list.append(i)
            if i == 100:
                break

        plt.figure()
        ax = plt.axes()
        ax.scatter(index_list, response_list, marker='o', label='Raw')
        ax.scatter(index_list, prediction_list, marker='x', label='Prediction')

        plt.xlabel('Sample')
        plt.ylabel('Response')
        plt.legend(loc='upper right')
        plt.show()


def fit_model(training_dataloader, validation_dataloader):
    # Create model and define loss function and optimiser.
    model = LatticeNetwork()
    loss_fn = nn.MSELoss(reduction='sum')
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=1.e-5,
                                weight_decay=0.,
                                momentum=0.)
    # optimiser = torch.optim.Adam(model.parameters(),
    #                              lr=1.e-5,
    #                              weight_decay=0.)

    EPOCHS = 1_001
    loss_tracker = []
    log_loss_tracker = []
    min_training_loss = 1.e9
    min_validation_loss = 1.e9

    for epoch in range(EPOCHS):
        training_loss = train_one_epoch(model,
                                        training_dataloader,
                                        optimiser,
                                        loss_fn)
        validation_loss = get_validation_loss(model,
                                              validation_dataloader,
                                              loss_fn)
        if training_loss < min_training_loss:
            min_training_loss = training_loss
            min_training_loss_epoch = epoch
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            min_validation_loss_epoch = epoch

        print(f'Epoch: {epoch}, Training Loss: {training_loss:14.12f}, '
              f' Validation Loss: {validation_loss:14.12f}')

        if (epoch) % 50 == 0:
            loss_tracker.append((epoch,
                                 training_loss,
                                 validation_loss))
            log_loss_tracker.append((epoch,
                                     log10(training_loss),
                                     log10(validation_loss)))

    print(f'Minimum training loss: {min_training_loss:14.12f} '
          f'at epoch {min_training_loss_epoch}')
    print(f'Minimum validation loss: {min_validation_loss:14.12f} '
          f'at epoch {min_validation_loss_epoch}')

# Plot progress of losses
    plot_losses(loss_tracker)
    # plot_losses(log_loss_tracker, loss_type='Log')

# Plot raw responses and predicted responses
    plot_responses(model, training_dataloader)


if __name__ == '__main__':

    # Random seed to ensure repeatability when testing.
    seed_everything(0, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)


# Get DataFrames for the training and test data.
    raw_dataframe, test_dataframe = get_raw_data('raw.csv',
                                                 'test.csv',
                                                 'lattice-physics-results/')

    no_features = 39
    raw_features, raw_responses = get_numpy_features_responses(raw_dataframe,
                                                               no_features)
    test_features, test_responses = get_numpy_features_responses(test_dataframe,
                                                                 no_features)

    # norm_raw_responses, norm_test_responses = normalise_responses(raw_responses,
    #                                                               test_responses)

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
    batch_size = 500
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)

    fit_model(training_dataloader, validation_dataloader)
