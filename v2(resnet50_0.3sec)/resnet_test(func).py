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
        self.feature = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        out = self.main(x)
        out = self.feature(out)
        return out


resnet50 = models.resnet50(pretrained=True)
fc_inputs = resnet50.fc.in_features

model = Net(resnet50, fc_inputs)

model.eval()
torch.no_grad()

model = torch.load('likely building_model_6.pt')
test_start = time.time()
out = model(imgblob)
prediction = F.softmax(out)
test_end = time.time()
print(test_end-test_start)
print(prediction)