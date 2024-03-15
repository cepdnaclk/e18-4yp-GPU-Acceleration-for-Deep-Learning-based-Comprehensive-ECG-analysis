import torch.nn as nn


class SimpleRNNClassification(nn.Module):
    def __init__(self):
        super(SimpleRNNClassification, self).__init__()
        self.rnn1 = nn.RNN(40000, 100, batch_first=True)
        self.relu = nn.ReLU()
        # self.rnn2 = nn.RNN(10, 1, batch_first=True)
        self.MLP = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.relu(out)
        # out, _ = self.rnn2(out)
        out = self.MLP(out)
        return out
