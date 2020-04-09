import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os

# dataset = r'D:\graduation-project\likely building'
# train_directory = os.path.join(dataset, 'train data')
# valid_directory = os.path.join(dataset, 'test data')
#
# image_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomHorizontalFlip(),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
# }
#
# data = {
#     'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
#     'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
#
# }
#
# train_data_size = len(data['train'])
# valid_data_size = len(data['valid'])
# print(train_data_size)
# print(valid_data_size)

# model = models.resnet50(pretrained=True)
# print('\n', model)
# feature0 = model.fc
# feature = nn.Sequential(*list(model.children())[:-1])
# print('\n', feature0)
# print('\n', feature)
# import matplotlib.pyplot as plt
# history = [[1,2,3,4],[1,2,6,3],[1,2,9,4],[1,2,12,5]]
# history = np.array(history)
# print(history)
# plt.plot(history[:, 0:2])
# plt.legend(['Tr Loss', 'Val Loss'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.ylim(0, 20)
# plt.xlim(0, 20)
# plt.show()
# plt.plot(history[:, 0:2])

resnet50 = models.resnet50(pretrained=True)


for param in resnet50.parameters():
    param.requires_grad = False

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)
f = nn.Sequential(resnet50)
print(f)
