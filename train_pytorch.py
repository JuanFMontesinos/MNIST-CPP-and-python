# test_libmnist.py
import numpy as np
import libmnist
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# PARAMETERS
# ================================================================================
lr = 0.01
n_epochs = 10
batch_size = 16

# Loading the data
# ================================================================================
test_images = libmnist.DataLoader.load_images('data/t10k-images-idx3-ubyte')
train_images = libmnist.DataLoader.load_images('data/train-images-idx3-ubyte')
train_images = torch.from_numpy(train_images).float()
test_images = torch.from_numpy(test_images).float()

test_labels = libmnist.DataLoader.load_labels('data/t10k-labels-idx1-ubyte')
train_labels = libmnist.DataLoader.load_labels('data/train-labels-idx1-ubyte')
test_labels = torch.from_numpy(test_labels).long()
train_labels = torch.from_numpy(train_labels).long()

N_CLASSES = 10
INPUT_SHAPE = train_images.shape[1]
N_TRAIN_SAMPLES = train_images.shape[0]
N_TEST_SAMPLES = test_images.shape[0]

# Buiding our neural network
# ================================================================================


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_SHAPE, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, N_CLASSES)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training
# ================================================================================
avg = 0
loss_values = []
pbar = tqdm(range(0, N_TRAIN_SAMPLES), total=N_TRAIN_SAMPLES)

c = 1  # Counter for gradient accumulation
for i in pbar:
    sample = train_images[i].unsqueeze(0)  # Add batch dimension
    gt = train_labels[i].unsqueeze(0)  # Add batch dimension

    # Forward pass
    output = model(sample)
    loss_value = criterion(output, gt)
    avg += loss_value.item()
    loss_values.append(loss_value.item())

    # Backward pass
    loss_value.backward()

    # Gradient accumulation and update
    if c == batch_size:
        optimizer.step()
        optimizer.zero_grad()
        c = 1
    else:
        c += 1

    pbar.set_description(f'Loss: {(avg/(i+1)):.4f}')

# Save loss plot
plt.plot(loss_values)
plt.savefig('results/loss.png')

# Testing
avg = 0
correct = 0
wrong = 0
pbar = tqdm(range(0, N_TEST_SAMPLES), total=N_TEST_SAMPLES)
with torch.no_grad():
    for i in pbar:
        sample = test_images[i].unsqueeze(0)  # Add batch dimension
        gt = test_labels[i].unsqueeze(0)  # Add batch dimension

        output = model(sample)
        loss_value = criterion(output, gt)
        avg += loss_value.item()

        # Get prediction
        pred = output.argmax(dim=1).item()
        if pred == gt.item():
            correct += 1
        else:
            wrong += 1

        acc = correct / (correct + wrong)

        if i % 1000 == 0:
            img = test_images[i].view(28, 28).numpy()
            plt.imsave(f'results/img_{i}_{int(pred)}.png', img)

        pbar.set_description(f'Loss: {(avg/(i+1)):.4f} | Acc: {acc:.4f}')
