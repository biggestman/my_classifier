import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import time

image_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


imagepath = r'D:\graduation-project\likely building\test data\oracle\oracle(1).png'
image = Image.open(imagepath).convert("RGB") #when the input is not a standard "rgb"
imgblob = image_transforms(image).unsqueeze(0)
imgblob = Variable(imgblob)

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

resnet50 = torch.load('likely building_model_6.pt')

resnet50.eval()
torch.no_grad()

test_start = time.time()

out = resnet50(imgblob)
prediction = F.softmax(out)

test_end = time.time()

print(test_end-test_start)
print(prediction)