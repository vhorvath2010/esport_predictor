from torch import nn
import torch


class NN(nn.Module):
    def __init__(self, n_ins):
        super(NN, self).__init__()
        self.h1 = nn.Linear(n_ins, 2)
        self.out = nn.Linear(2, 1)

    def forward(self, ins):
        out = self.h1(ins)
        out = self.out(out)
        return out


# Load model
model = NN(3)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Get inputs
t1_wins = int(input("Enter the first teams form: "))
t2_wins = int(input("Enter the second teams form: "))
h2h = int(input("Enter the head to head: "))
y = model.forward(torch.as_tensor([t1_wins, t2_wins, h2h], dtype=torch.float))
t1_odds = str(round(100 * (1-y.item()), 2)) + "%"
print("Team one has a", t1_odds, "chance of winning")
