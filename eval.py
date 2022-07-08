import numpy as np
import torch
from model import Model

# Load model
model = Model(n_ins=5)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Select device
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)
model.to(device)

test = np.load('test.npy', allow_pickle=True)
correct_picks = 0
for match in test:
    x = torch.tensor([match.get_t1_rank(), match.get_t1_form(), match.get_h2h(), match.get_t2_rank(),
                      match.get_t2_form()], device=device).float()
    y = match.res
    y_hat = round(model(x).item())
    if y == y_hat:
        correct_picks += 1

print("Correctly predicted:", correct_picks/len(test) * 100, "percent")
