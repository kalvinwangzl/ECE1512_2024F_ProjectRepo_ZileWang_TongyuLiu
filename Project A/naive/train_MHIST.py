import time
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN
from utils import get_dataset, TensorDataset, get_default_convnet_setting, get_network, get_time, distance_wb, match_loss, get_loops, epoch, evaluate_synset, augment
from utils import get_daparam, get_eval_pool, ParamDiffAug, set_seed_DiffAug, DiffAugment, rand_scale, rand_rotate, rand_flip, rand_brightness, rand_saturation, rand_contrast, rand_crop, rand_cutout
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

device = 'cpu'

class MHISTDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'SSA': 0, 'HP': 1}
        
        self.annotations['Label'] = self.annotations['Majority Vote Label'].map(self.label_map)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.annotations.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return image, label


data_path = './mhist_dataset' 
images_dir = os.path.join(data_path, 'images')
annotations_file = os.path.join(data_path, 'annotations.csv')


transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
])


all_dataset = MHISTDataset(annotations_file, images_dir, transform=transform)


# train_indices = all_dataset.annotations[all_dataset.annotations['Partition'] == 'train'].index.tolist()
# test_indices = all_dataset.annotations[all_dataset.annotations['Partition'] == 'test'].index.tolist()


# train_labels = all_dataset.annotations.loc[train_indices, 'Label'].values


# class_0_indices = [idx for idx, label in zip(train_indices, train_labels) if label == 0]
# class_1_indices = [idx for idx, label in zip(train_indices, train_labels) if label == 1]


# np.random.seed(42)
# class_0_sampled = np.random.choice(class_0_indices, 50, replace=False)
# class_1_sampled = np.random.choice(class_1_indices, 50, replace=False)


# sampled_train_indices = list(class_0_sampled) + list(class_1_sampled)


# train_dataset = torch.utils.data.Subset(all_dataset, sampled_train_indices)
# test_dataset = torch.utils.data.Subset(all_dataset, test_indices)


# trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# print(f'Train dataset size (sampled): {len(trainloader.dataset)}')
# print(f'Test dataset size: {len(testloader.dataset)}')
# exit()


train_indices = all_dataset.annotations[all_dataset.annotations['Partition'] == 'train'].index.tolist()
test_indices = all_dataset.annotations[all_dataset.annotations['Partition'] == 'test'].index.tolist()


train_dataset = torch.utils.data.Subset(all_dataset, train_indices)
test_dataset = torch.utils.data.Subset(all_dataset, test_indices)


trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print(f'Train dataset size: {len(trainloader.dataset)}')
print(f'Test dataset size: {len(testloader.dataset)}')

# num_samples_to_show = 5
# for i in range(num_samples_to_show):
#     image, label = train_dataset[i]
#     image = image.permute(1, 2, 0)
#     plt.imshow(image)  # 显示图像
#     plt.title(f'Label: {label}')
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
# exit()


num_epochs = 20   
initial_lr = 0.01   
batch_size = 16    

net_width = 128
net_depth = 7
net_act = 'relu'
net_norm = 'instancenorm'
net_pooling = 'avgpooling'
channel = 3
num_classes = 2
im_size = (128, 128)
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

# 6. 计算 FLOPs
from ptflops import get_model_complexity_info

flops, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True, print_per_layer_stat=False)
print(f'FLOPs: {flops}, Parameters: {params}')

# Loss: 0.0854, Accuracy: 99.40%
# Test Accuracy: 78.92%
# FLOPs: 881.31 MMac, Parameters: 891.14 k