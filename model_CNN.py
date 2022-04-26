from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from data_process import get_embedLookup
from GUI_designer.Control import  embed_lookup
#embed_lookup = get_embedLookup()

# First checking if GPU is available
# train_on_gpu=torch.cuda.is_available()
#
# if(train_on_gpu):
#     print('Training on GPU.')
# else:
#     print('No GPU available, training on CPU.')



class SentimentCNN(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.
    """

    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                 num_filters=100, kernel_sizes=[3, 4, 5], freeze_embeddings=True, dropout=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentCNN, self).__init__()

        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  # all vectors
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embedding.requires_grad = False

        # 2. convolutional layers
        # even though it's Conv2d, effectively it's 1d thanks to carefully chosen dimensions.
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k - 2, 0))
            for k in kernel_sizes])

        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)

        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x)).squeeze(3)

        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x.long())  # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)

        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]

        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)

        # final logit
        logit = self.fc(x)

        # sigmoid-activated --> a class score
        return self.sig(logit)

def CNN_model(output_size, num_filters, kernel_sizes, dropout):
    embed_lookup = KeyedVectors.load_word2vec_format(r"E:\Debug\vscode\python\CNN\GoogleNews-vectors-negative300-SLIM.bin", binary=True)
    word = 'news'
    vocab_size = len(embed_lookup)
    embedding_dim = len(embed_lookup[word]) # 300-dim vectors
    freeze_embeddings = True
    net = SentimentCNN(embed_lookup, vocab_size, output_size, embedding_dim,
                       num_filters, kernel_sizes, freeze_embeddings, dropout)
    print(net)
    return net
    # loss and optimization functions




# 开始训练
def train(net, epochs, lr, batch_size):
    pttexts = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\pttexts.npy')
    labels = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\labels.npy')
    train_x, rem_x, train_y, rem_y = train_test_split(pttexts, labels, train_size=0.8, random_state=1)
    val_x, test_x, val_y, test_y = train_test_split(rem_x, rem_y, test_size=0.5, random_state=1)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    # create Tensor datasets
    # train_loader = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\train_loader.npy', allow_pickle=True)
    # valid_loader = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\valid_loader.npy', allow_pickle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # move model to GPU, if available
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        net.cuda()

    counter = 0  # for printing
    print_every =  2 * batch_size
    # train for some num    ber of epochs
    net.train()
    for e in range(epochs):

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            # loss stats, f1 score
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                f1_fake_all = []
                y_true = []
                y_pred = []

                net.eval()
                for inputs, labels in valid_loader:
                    cpulabels = labels
                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())
                    pred = torch.round(output.squeeze())

                    y_true = np.append(y_true, cpulabels)
                    y_pred = np.append(y_pred, pred.detach().numpy()
                    if not train_on_gpu
                    else pred.detach().cpu().numpy())

                    f1_fake = f1_score(y_true, y_pred)

                    f1_fake_all.append(f1_fake)
                    # you can also print classification report to get more details
                # report = classification_report(y_true, y_pred, target_names=['true', 'fake'])

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}...".format(np.mean(val_losses)),
                      "F1-score 'fake': {:.6f}...".format(np.mean(f1_fake_all)), )
                # "\nClassification Report:\n{}".format(report))
        torch.save(net, r'E:\office应用\毕业设计\fake-news-CNN\net_model.pth')  # 保存整个网络

# epochs = 2 # this is approx where the validation loss stops decreasing
# print_every = batch_size*2
#train(net, train_loader, epochs, print_every=print_every)

def test(net_path):
    pttexts = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\pttexts.npy')
    labels = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\labels.npy')
    train_x, rem_x, train_y, rem_y = train_test_split(pttexts, labels, train_size=0.8, random_state=1)
    val_x, test_x, val_y, test_y = train_test_split(rem_x, rem_y, test_size=0.5, random_state=1)
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    batch_size = 64
    test_loader = DataLoader(test_data, batch_size=batch_size)

    test_losses = []  # track loss
    num_correct = 0
    y_true = []
    y_pred = []
    #test_loader = np.load(r'E:\office应用\毕业设计\fake-news-CNN\processed_data\test_loader.npy', allow_pickle=True)
    net = torch.load(net_path)
    net.eval()
    criterion = nn.BCELoss()
    train_on_gpu = torch.cuda.is_available()
    for inputs, labels in test_loader:
        y_true = np.append(y_true, labels.numpy())

        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output = net(inputs)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        y_pred = np.append(y_pred, pred.detach().numpy() if not train_on_gpu
        else pred.detach().cpu().numpy())
        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    report = classification_report(y_true, y_pred,
                                   target_names=['true', 'fake'],
                                   digits=3)
    # -- stats! -- ##
    print("Classification report:\n{}".format(report))

    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))