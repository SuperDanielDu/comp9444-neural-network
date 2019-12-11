import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
# from nltk.tokenize.moses import MosesTokenizer

# Class for creating the neural network.
class Network(tnn.Module):
    """
        Implement an LSTM-based network that accepts batched 50-d
        vectorized inputs, with the following structure:
        LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
        Assume batch-first ordering.
        Output should be 1d tensor of shape [batch_size].
        """
    def __init__(self):
        super(Network, self).__init__()
        self.lstm = tnn.LSTM(input_size=50, hidden_size=100, batch_first=True, num_layers=2, bidirectional=True)  # input_size = input_dim

        self.linear = tnn.Linear(100, 64)

        self.relu = tnn.ReLU()
        self.linear2 = tnn.Linear(64, 1)



    def forward(self, input, length=None):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """(B, L, 64)
        # print(input.shape)
        l_out, _ = self.lstm(input, None)  # input = (batch_64, L, 50)
        # l_out = (batch, L, num_dir_1 * hidden_size_100)  | h_n, c_n == (最后一个时刻branch hidden state, main hidden state)
        # h_n = (num_layers*num_directions, batch, hidden_size)
        linear_x = self.linear(l_out)  # (batch,L,input)  # (B, L, 100)   ->
        relu_x = self.relu(linear_x)
        return self.linear2(relu_x[:, -1, :]).squeeze()  # (batch, 64)->(batch, 1)->(batch)

class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        # shuffle
        # print(x)
        x = list(np.random.permutation(x))

        # remove punct
        for word in x[::-1]:
            if not word.isalnum():
                x.remove(word)

        # random delete some text
        p=0.2
        len_ = len(x)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            x[i] = ''
        x = list(x)
        return ' '.join(x)

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        # print(batch)
        # print(vocab)
        return batch
    stop_words = {'I','movie','film','some', 'those', 'if', 're', 'further', 'ma', 'below', 'above', 'it', 'are', "shan't", 'who', 'have', 'the', 'because', "couldn't", 'too', 'through', 'his', 'does', 'these', 'hers', 'they', 'before', 'up', 'only', 'any', "isn't", 'wouldn', 'more', 'from', 'been', "wouldn't", 'd', 'their', 'into', 'should', 'didn', "shouldn't", 'why', 'ourselves', 'at', 'has', 'then', 'theirs', 'won', 'until', 'shan', 'mustn', 'to', 'yourselves', 'haven', 'during', 'shouldn', 'don', "weren't", "doesn't", 'an', 'just', "mightn't", 'her', 'having', 'not', 'will', 'hadn', "you've", 'do', 'yours', 'had', 'whom', 'myself', 'out', 'll', 'same', 'him', 'nor', 'each', 'as', 'aren', 'doesn', 'm', 'here', 'over', 'needn', 'hasn', 'she', 'itself', 'did', 'between', 'very', "you'd", 'your', "haven't", 'how', 'is', 'wasn', 's', 'by', 'he', "she's", 'which', "you'll", 'themselves', 'there', 'you', 'about', 'and', 'down', 'my', 'under', 'own', 'few', 'weren', 'yourself', 'couldn', "hasn't", 'all', 'but', 'off', 'o', "you're", 'were', 'now', 'can', "hadn't", 't', 'am', 'herself', 'a', 'in', 'such', 'that', 'ours', "aren't", "needn't", 'again', "wasn't", "that'll", 'what', 'on', 'y', 've', 'was', 'once', 'its', 'when', 'be', 'where', 'this', 'than', "mustn't", 'me', 'i', 'them', 'we', 'or', 'doing', 'no', 'most', "didn't", 'our', 'with', 'after', "should've", 'against', 'for', 'mightn', "won't", 'while', 'of', 'isn', "don't", 'so', 'himself', 'ain', 'being', 'both', "it's", 'other'}
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post,stop_words=stop_words)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    net.load_state_dict(torch.load('./model.pth'))

    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(20):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


# from torchsummary import summary
if __name__ == '__main__':
    # net = Network()
    # print(summary(net, input_size=(2000,50)))
    main()
