import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import torch

import os
import math as m
from PIL import Image as I
import numpy as np
import matplotlib.pyplot as plt

#Model Initialization, Layer Replacement, and Gradient Changes
x = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in x.parameters():
    param.requires_grad = False

x.fc = nn.Linear(2048, 6)
x.fc.requires_grad = True

def load_images():
    #Bunch of arrays for storing and moving data
    aux_img = []
    aux_labels = []
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    val_imgs = []
    val_labels = []

    #Cycle through various folders for images
    for folder in range(len(sorted(os.listdir('./Data')))):
        if folder == 0:
            continue
        else:
            aux_img = os.listdir('./Data/'+os.listdir('./Data')[folder])

            for i in range(len(aux_img)):
                #Remove .DS_Store file from data
                if aux_img[i] == '.DS_Store':
                    aux_img.pop(i)
                    break
            
            #Resize Image and Convert Channels
            for i in range(len(aux_img)):
                try:
                    with I.open('./Data/'+os.listdir('./Data')[folder]+'/'+aux_img[i]) as img:
                        img = img.convert('L').resize((32, 32))
                        img = np.array(img)
                        img = np.stack([img]*3, axis=-1)
                        aux_img[i] = img
                except Exception as e:
                    print(f"{i}")

            #Generate Labels
            aux_labels = [(folder-1) for i in range(len(aux_img))]

            #Divide Data and Labels into Training, Testing, and Validation Sets
            train_imgs += aux_img[:int(m.ceil(len(aux_img)*0.6))]
            aux_img = aux_img[int(m.ceil(len(aux_img)*0.6)):]
            val_imgs += aux_img[:int(m.ceil(len(aux_img)*0.75))]
            aux_img = aux_img[int(m.ceil(len(aux_img)*0.75)):]
            test_imgs += aux_img

            train_labels += aux_labels[:int(m.ceil(len(aux_labels)*0.6))]
            aux_labels = aux_labels[int(m.ceil(len(aux_labels)*0.6)):]
            val_labels += aux_labels[:int(m.ceil(len(aux_labels)*0.75))]
            aux_labels = aux_labels[int(m.ceil(len(aux_labels)*0.75)):]
            test_labels += aux_labels

    #Data Normalization RGB -> [0, 1]
    aux_img = train_imgs + test_imgs + val_imgs
    mu = np.mean(aux_img, axis=(0, 1, 2))
    sigma = np.std(aux_img, axis=(0, 1, 2))

    normalize = T.Compose([
        T.Normalize(mu, sigma)
    ])

    #Conversion to tensors and recast data
    train_imgs, test_imgs, val_imgs = torch.tensor(train_imgs).to(torch.float32), torch.tensor(test_imgs).to(torch.float32), torch.tensor(val_imgs).to(torch.float32)
    train_labels, test_labels, val_labels = torch.tensor(train_labels), torch.tensor(test_labels), torch.tensor(val_labels)

    #Shuffle Images Sets
    train_imgs, train_labels = shuffle_imgs(train_imgs, train_labels)
    test_imgs, test_labels = shuffle_imgs(test_imgs, test_labels)
    val_imgs, val_labels = shuffle_imgs(val_imgs, val_labels)

    #Normalize and Reorder Data for Processing
    train_imgs = normalize(train_imgs.permute(0, 3, 1, 2))
    test_imgs = normalize(test_imgs.permute(0, 3, 1, 2))
    val_imgs = normalize(val_imgs.permute(0, 3, 1, 2))

    return train_imgs, train_labels, test_imgs, test_labels, val_imgs, val_labels

#Shuffle Images according to Random Noise Generation
def shuffle_imgs(imgs, labels):
   B = imgs.size(0)
   shuffle = torch.randperm(B)
   return imgs[shuffle], labels[shuffle]

#Training Method with Adam Optimizer
def train(model, data, labels, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_list = []

    for i in range(epochs):
        optimizer.zero_grad()

        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
    return model, loss_list


train_imgs, train_labels, test_imgs, test_labels, val_imgs, val_labels = load_images()

x, loss_list = train(x, train_imgs, train_labels, 50)

test_output = x(test_imgs)
y_hat = -1 + torch.zeros_like(test_labels, dtype=torch.int64)

#Testing Evaluation
y_hat[:len(test_imgs)] = torch.argmax(test_output, axis=1)
print(torch.mean((y_hat == test_labels).float()))

#Training Loss vs. Epochs Plot
plt.plot(np.linspace(1, len(loss_list), len(loss_list)), loss_list)
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()