# Script to run a prediction on a single match
import torch
from model import Model

# Load model
model = Model(n_ins=5)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Select device
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Predicting using:", device)
model.to(device)

# Acquire inputs
t1_rank = int(input("Enter the first team's rank: "))
t2_rank = int(input("Enter the second team's rank: "))
h2h = int(input("Enter the h2h ranking: "))
t1_form = int(input("Enter the first team's form: "))
t2_form = int(input("Enter the second team's form: "))
x = torch.tensor([t1_rank, t1_form, h2h, t2_rank, t2_form], device=device).float()

y_hat = model(x).item()
t1_win = round(y_hat) == 1
print("the", "first" if t1_win else "second", "team is predicted to win with",
      y_hat * 100 if t1_win else (1 - y_hat) * 100, "% certainty")
