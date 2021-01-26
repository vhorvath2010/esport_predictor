import csv
import torch.nn as nn
import torch


class Match:
    def __init__(self, t1_form, t2_form, h2h, rank_diff, res):
        self.t1_form = t1_form
        self.t2_form = t2_form
        self.h2h = h2h
        self.rank_diff = rank_diff
        self.res = res


class NN(nn.Module):
    def __init__(self, n_ins):
        super(NN, self).__init__()
        self.h1 = nn.Linear(n_ins, 2)
        self.out = nn.Linear(2, 1)

    def forward(self, ins):
        out = self.h1(ins)
        out = self.out(out)
        return out


# Load match data
matches = []
with open("csgo_match_data.csv") as match_data_file:
    csv_reader = csv.reader(match_data_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            matches.append(Match(int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])))
        line_count += 1

model = NN(3)
criterion = nn.MSELoss()
optim = torch.optim.Adam(lr=0.000025, params=model.parameters())
for epoch in range(1, 1000):
    optim.zero_grad()
    model.zero_grad()
    criterion.zero_grad()
    c_loss = 0
    for match in matches:
        x = torch.as_tensor([match.t1_form, match.t2_form, match.h2h], dtype=torch.float)
        y = model(x)
        loss = criterion(y, torch.as_tensor(match.res, dtype=torch.float))
        loss.backward()
        optim.step()
        c_loss += loss.item()
    print("loss for", epoch, c_loss/len(matches))

torch.save(model.state_dict(), "model.pt")
