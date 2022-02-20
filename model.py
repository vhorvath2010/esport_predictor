from torch import nn as nn


class Model(nn.Module):
    def __init__(self, n_ins):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(n_ins, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.out(x)
        x = self.sig(x)

        return x
