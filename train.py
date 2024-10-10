# test_libmnist.py
import numpy as np
import libmnist
from tqdm import tqdm
import matplotlib.pyplot as plt


# PARAMETERS
# ================================================================================
lr = 0.01
n_epochs = 10
batch_size = 16

# Loading the data
# ================================================================================
test_images = libmnist.DataLoader.load_images('data/t10k-images-idx3-ubyte')
train_images = libmnist.DataLoader.load_images('data/train-images-idx3-ubyte')

test_labels = libmnist.DataLoader.load_labels('data/t10k-labels-idx1-ubyte')
train_labels = libmnist.DataLoader.load_labels('data/train-labels-idx1-ubyte')

N_CLASSES = 10
INPUT_SHAPE = train_images.shape[1]
N_TRAIN_SAMPLES = train_images.shape[0]
N_TEST_SAMPLES = test_images.shape[0]

# Buiding our neural network
# ================================================================================

neural_network = (
    libmnist.LinearLayer(INPUT_SHAPE, 256),
    libmnist.ReLu(),
    libmnist.LinearLayer(256, 100),
    libmnist.ReLu(),
    libmnist.LinearLayer(100, N_CLASSES)
)
# This computes softmax and cross entropy loss
loss = libmnist.SoftmaxndCrossEntropy(N_CLASSES)

# Training 
avg = 0
loss_values = []
pbar = tqdm(range(0, N_TRAIN_SAMPLES), total=N_TRAIN_SAMPLES)

c = 1 # Counter for gradient accumulation
for i in pbar:
    sample = train_images[i]
    gt = train_labels[i]

    # Iterate through the layers and compute the forward pass
    for layer in neural_network:
        sample = layer.forward(sample)
    
    # Compute the softmax out of logits from last layer and
    # Cross entropy loss
    loss_value = loss.forward(sample, gt.item())
    
    
    loss_values.append(loss_value)
    avg += loss_value
    
    # Compute gradient of the loss w.r.t the output of the last layer
    grad = loss.backward()
    
    # Propagate the gradients through the network
    # Gradients are accumulated as in pytorch
    for layer in reversed(neural_network):
        grad = layer.backward(grad)
    
    # Once we accumulated gradients for BACH_SIZE samples
    # Update the weights and reset gradients
    if c == batch_size:
        for layer in neural_network:
            layer.update(lr)
        c = 1
    c += 1
    pbar.set_description(f'Loss: {(avg/(i+1)):.4f}')

plt.plot(loss_values)
plt.savefig('results/loss.png')
# Testing
avg = 0
good = 0
wrong = 0
pbar = tqdm(range(0, N_TEST_SAMPLES), total=N_TEST_SAMPLES)
for i in pbar:
    sample = test_images[i]
    gt = test_labels[i]

    for layer in neural_network:
        sample = layer.forward(sample)
    loss_value = loss.forward(sample, gt.item())
    avg += loss_value
    pred = np.argmax(sample)
    if pred == gt:
        good += 1
    else:
        wrong += 1
    acc = good / (good + wrong)
    if i % 1000 == 0:
        img = test_images[i].view(28, 28)
        plt.imsave(f'results/img_{i}_{int(pred)}.png', img)
    pbar.set_description(f'Loss: {(avg/(i+1)):.4f} | Acc: {acc:.4f}')
