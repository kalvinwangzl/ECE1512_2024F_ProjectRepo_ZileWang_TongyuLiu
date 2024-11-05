import time
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from collections import defaultdict
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
from utils import get_dataset, TensorDataset, get_default_convnet_setting, get_network, get_time, distance_wb, match_loss, get_loops, epoch, evaluate_synset, augment
from utils import get_daparam, get_eval_pool, ParamDiffAug, set_seed_DiffAug, DiffAugment, rand_scale, rand_rotate, rand_flip, rand_brightness, rand_saturation, rand_contrast, rand_crop, rand_cutout
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

data_path = './data' 
num_epochs = 20    
initial_lr = 0.01   
batch_size = 16      


channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset('CIFAR10', data_path) # Could be CIFAR10 here

def sample_per_class(dataset, num_classes, samples_per_class):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
    sampled_indices = [idx for indices in class_indices.values() for idx in indices]
    return Subset(dataset, sampled_indices)


sampled_train_dataset = sample_per_class(dst_train, num_classes=10, samples_per_class=40)
trainloader = DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=True)

'''Uncomment this line if want to train on whole data'''
# trainloader = DataLoader(dst_train, batch_size=batch_size, shuffle=True)

device = 'cuda'

net_width = 128
net_depth = 3
net_act = 'relu'
net_norm = 'instancenorm'
net_pooling = 'avgpooling'
model = ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

   
    scheduler.step()

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100.*correct/total:.2f}%')

end_time = time.time()
training_time = end_time - start_time


print(f'Total Training Time: {training_time:.2f} seconds')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# from ptflops import get_model_complexity_info

# flops, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True, print_per_layer_stat=False)
# print(f'FLOPs: {flops}, Parameters: {params}')

# Test Accuracy: 99.41%
# FLOPs: 49.59 MMac, Parameters: 317.71 k

# Loss: 0.0079, Accuracy: 99.87%
# Test Accuracy: 99.42%
# FLOPs: 49.59 MMac, Parameters: 317.71 k