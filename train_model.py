import numpy as np
import torch

from model import Model
import torch.optim as optim
import torch.nn as nn
from match import Match

# Load data
train = np.load('train.npy', allow_pickle=True)

# Create model and initialize functions
model = Model(n_ins=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)

# Select device
device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)
model.to(device)

# Begin training model
epochs = 100
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for match in train:
        # Zero gradients
        optimizer.zero_grad()

        # Create inputs and outputs
        x = torch.tensor([match.get_t1_rank(), match.get_t1_form(), match.get_h2h(), match.get_t2_rank(),
                          match.get_t2_form()], device=device).float()
        y = torch.tensor([match.res], device=device).float()

        # forward + backward + optimize
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Loss for epoch", epoch, "is", running_loss / len(train))

# Save model
torch.save(model.state_dict(), 'model.pth')
