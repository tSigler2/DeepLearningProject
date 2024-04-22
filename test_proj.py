import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T

import os
import math as m
from PIL import Image as I
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as cm

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch

#Model Initialization, Layer Replacement, and Gradient Changes
x = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in x.parameters():
    param.requires_grad = False

x.fc = nn.Linear(2048, 12)
nn.init.kaiming_uniform_(x.fc.weight, nonlinearity='relu')
x.fc.requires_grad = True

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch, 'has_mps', False):
    device = "mps"
else:
    device = "cpu"

x = x.to(device)

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
    idx = 0

    crop = T.CenterCrop((128, 128))

    #Cycle through various folders for images
    for folder in range(len(sorted(os.listdir('/Users/thomassigler/DeepLearningProject/Data')))):
        if os.listdir('/Users/thomassigler/DeepLearningProject/Data')[folder] == '.DS_Store':
            continue
        else:
            aux_img = os.listdir('/Users/thomassigler/DeepLearningProject/Data/'+os.listdir('/Users/thomassigler/DeepLearningProject/Data')[folder])

            for i in range(len(aux_img)):
                #Remove .DS_Store file from data
                if aux_img[i] == '.DS_Store':
                    aux_img.pop(i)
                    break
            
            #Resize Image and Convert Channels
            for i in range(len(aux_img)):
                try:
                    with I.open('/Users/thomassigler/DeepLearningProject/Data/'+os.listdir('/Users/thomassigler/DeepLearningProject/Data')[folder]+'/'+aux_img[i]) as img:
                        img = crop(img)

                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        img = np.array(img)
                        aux_img[i] = img
                except Exception as e:
                    print(f"{i}")
            aux_labels = []

            #Generate Labels
            for i in range(len(aux_img)):
                aux_labels.append(idx)
            idx += 1

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
    aux_img = []
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

    train_set, test_set, val_set = TensorDataset(train_imgs, train_labels), TensorDataset(test_imgs, test_labels), TensorDataset(val_imgs, val_labels)

    trainloader, testloader, valloader = DataLoader(dataset=train_set, shuffle=True, batch_size=128), DataLoader(dataset=test_set, shuffle=True, batch_size=64), DataLoader(dataset=val_set, shuffle=True, batch_size=64)

    return trainloader, testloader, valloader

#Shuffle Images according to Random Noise Generation
def shuffle_imgs(imgs, labels):
   B = imgs.size(0)
   shuffle = torch.randperm(B)
   return imgs[shuffle], labels[shuffle]

#Training Method with Adam Optimizer
def trainfunc(model, traindata, valdata, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.006349288161821683, eps=6.5131448417080975e-09)
    #optimizer = optim.SGD(model.parameters(), lr=0.006349288161821683)
    lr_optim = optim.lr_scheduler.StepLR(optimizer, step_size=2)
    loss_list = []
    loss_list_aux = []

    for i in range(epochs):
        for imgs, labels in traindata:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss_list_aux.append(loss.item())

            loss.backward()
            optimizer.step()
        loss_list.append(torch.mean(torch.tensor(loss_list_aux)))

        with torch.no_grad():
            div = 0
            tot = 0
            for imgs, labels in valdata:
                imgs = imgs.to(device)
                labels = labels.to(device)
                val_logits = model(imgs)
                val_logits = nn.Softmax()(val_logits)
                pred = torch.argmax(val_logits, axis=1)
                tot += (pred == labels).float().sum().item()
                div += imgs.shape[0]
            print("Validation Accuracy: ", tot/div*100, "%")

    return model, loss_list

#Finetune Function
def finetune(model, data, valdata, epochs):
    model.fc.requires_grad = False
    model.avgpool.requires_grad = True
    model.layer4.requires_grad = True
    model.layer3.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_list = []
    loss_list_aux = []

    for i in range(epochs):
        for imgs, labels in data:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss_list_aux.append(loss.item())

            loss.backward()
            optimizer.step()
        
        loss_list.append(torch.mean(torch.tensor(loss_list_aux)))
        
        div = 0
        tot = 0
        for imgs, labels in valdata:
            imgs = imgs.to(device)
            labels = labels.to(device)
            val_logits = model(imgs)
            val_logits = nn.Softmax()(val_logits)
            pred = torch.argmax(val_logits, axis=1)
            tot += (pred == labels).float().sum().item()
            div += imgs.shape[0]
        print("Validation Accuracy: ", tot/div*100, "%")
    return model, loss_list

def test(model, data):
    div = 0
    total = 0

    for imgs, labels in data:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        logits = nn.Softmax()(logits)
        pred = torch.argmax(logits, axis=1)
        div += imgs.shape[0]
        total += (pred == labels).float().sum().item()
    print("Accuracy: ", total/div*100, "%")

def confusion_matrix(model, data):
    total_pred = []
    total_labels = []

    for imgs, labels in data:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        logits = nn.Softmax()(logits)
        pred = torch.argmax(logits, axis=1)

        total_pred += list(pred.to('cpu'))
        total_labels += list(labels.to('cpu'))

    con_mat = cm(total_labels, total_pred)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(con_mat, annot=True, cmap="inferno")
    plt.title(f'ResNet50 Accuracy')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()
    plt.clf()

def hfinetunetrain(model, data, optimizer):
    for imgs, labels in data:
        imgs, labels = imgs.to(device), labels.to(device)

        pred = model(imgs)
        optimizer.zero_grad()
        loss = F.cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()
    torch.mps.empty_cache()

def hfinetunetest(model, data):
    model.eval()
    correct = 0
    total = 0

    for imgs, labels in data:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        
        pred = torch.argmax(outputs, axis=1)

        total += imgs.shape[0]
        correct += (pred == labels).float().sum().item()
    torch.mps.empty_cache()
    return correct/total

def objective(config):
    train_loader, test_loader, val_loader = load_images()

    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    for param in x.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, config["l1"]),
                                       nn.ReLU(),
                                       nn.Linear(config["l1"], config["l2"]),
                                       nn.ReLU(),
                                       nn.Linear(config["l2"], 12)
                                       ).to(device)
    nn.init.kaiming_uniform_(x.fc.weight, nonlinearity='relu')
    x.fc.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"])


    for epoch in range(10):
        hfinetunetrain(model, train_loader, optimizer)
        acc = hfinetunetest(model, val_loader)
        
        checkpoint = None
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(),
                       "/Users/thomassigler/DeepLearningProject/TuningCheckpoints/model.pth")
            checkpoint = Checkpoint.from_directory("/Users/thomassigler/DeepLearningProject/TuningCheckpoints")
        train.report({"mean_accuracy": acc}, checkpoint=checkpoint)
        torch.mps.empty_cache()

#search_space = {"lr": tune.loguniform(1e-4, 1e-1), "eps": tune.loguniform(1e-10, 1e-6), "l1": tune.randint(50, 2048), "l2": tune.randint(12, 100)}
#tuner = tune.Tuner(objective, tune_config=tune.TuneConfig(metric="mean_accuracy", mode="max", search_alg=OptunaSearch(), num_samples=2), param_space=search_space)

#results = tuner.fit()
#print("Best Config is: ", results.get_best_result().config)

trainloader, testloader, valloader = load_images()
test(x, testloader)

confusion_matrix(x, testloader)

x, loss_list = trainfunc(x, trainloader, valloader, 50)
test(x, testloader)

confusion_matrix(x, testloader)

#Finetuning and testing Finetuning
x, loss_list_val = finetune(x, trainloader, valloader, 50)
test(x, testloader)

#Training Loss vs. Epochs Plot
plt.plot(np.linspace(1, len(loss_list), len(loss_list)), loss_list)
plt.title('Training and Validation Loss over Epochs No Fine Tuning')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
plt.clf()

plt.plot(np.linspace(1, len(loss_list_val), len(loss_list_val)), loss_list_val)
plt.title('Loss over Epochs Fine Tuning')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()