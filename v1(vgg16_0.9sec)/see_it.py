import time
starttime = time.time()
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch
from PIL import Image
import torch.nn.functional as F

data_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor()])


# imagepath = r'D:\graduation-project\traindata\rose\rose (10).jpg'
imagepath = r'D:\graduation-project\traindata\daisy\daisy (10).jpg'
image = Image.open(imagepath)
imgblob = data_transforms(image).unsqueeze(0)
imgblob = Variable(imgblob)




model = models.vgg16_bn(pretrained=True)
num_fc_ftr = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_fc_ftr, 2)
model.load_state_dict(torch.load('net_params(1).pkl'))
model.eval()
torch.no_grad()
test_start = time.time()
out = model(imgblob)
prediction = F.softmax(out)
endtime = time.time()
print(prediction)
dtime = endtime - starttime
print("程序运行时间：%.8s s" % dtime)
