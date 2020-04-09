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


imagepath = r'D:\graduation-project\likely building\test data\oracle\22.jpg'
image = Image.open(imagepath) #.convert("RGB") #when the input is not a standard "rgb"
imgblob = image_transforms(image).unsqueeze(0)
imgblob = Variable(imgblob)

resnet34 = models.resnet34(pretrained=True)


for param in resnet34.parameters():
    param.requires_grad = False

fc_inputs = resnet34.fc.in_features
resnet34.fc = nn.Sequential(
    nn.Linear(in_features=fc_inputs, out_features=4096, bias=True),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=4096, out_features=3, bias=True)
)
resnet34 = torch.load('resnet50_4.94.pt')

resnet34.eval()
torch.no_grad()

test_start = time.time()

out = resnet34(imgblob)
prediction = F.softmax(out)

test_end = time.time()

print(test_end-test_start)
print(prediction)