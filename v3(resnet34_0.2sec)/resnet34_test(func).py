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
image = Image.open(imagepath)
print(type(image))
imgblob = image_transforms(image).unsqueeze(0)
imgblob = Variable(imgblob)


class Net(nn.Module):
    def __init__(self, model, fc_inputs):
        super(Net, self).__init__()
        self.main = nn.Sequential(*list(model.children())[:-1])
        self.feature = nn.Sequential(nn.Linear(in_features=fc_inputs, out_features=4096,bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(in_features=4096, out_features=4096,bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(in_features=4096, out_features=3,bias=True)
)


    def forward(self, x):
        out = self.main(x)
        out = self.feature(out)
        return out


resnet34 = models.resnet34(pretrained=True)
fc_inputs = resnet34.fc.in_features

model = Net(resnet34, fc_inputs)

model.eval()
torch.no_grad()

model = torch.load('resnet50_4.94.pt')
test_start = time.time()
out = model(imgblob)
prediction = F.softmax(out)
test_end = time.time()
print(test_end-test_start)
print(prediction)