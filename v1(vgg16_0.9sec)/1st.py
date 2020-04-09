import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch

model = models.vgg16_bn(pretrained=True)
# feature = nn.Sequential(model)
# feature2 = nn.Sequential(*list(model.children())[1:])
print(model.classifier)
num_fc_ftr = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_fc_ftr,2)
print(model.classifier)

for param in model.parameters():
    param.requires_grad = False
for param in model.classifier[6].parameters():
    param.requires_grad = True

for children in model.children():
    print(children)
    for param in children.parameters():
        print(param)

#train data
train_data=torchvision.datasets.ImageFolder(r'D:\graduation-project\traindata',transform=transforms.Compose(
                                                                        [
                                                                            transforms.Resize(256),
                                                                            transforms.CenterCrop(224),
                                                                            transforms.ToTensor()
                                                                        ]))
train_loader = DataLoader(train_data,batch_size=20,shuffle=True)

#test data
test_data = torchvision.datasets.ImageFolder(r'D:\graduation-project\traindata',transform=transforms.Compose(
                                                                        [
                                                                            transforms.Resize(256),
                                                                            transforms.CenterCrop(224),
                                                                            transforms.ToTensor()
                                                                        ]))
test_loader = DataLoader(test_data,batch_size=20,shuffle=True)


criterion = torch.nn.CrossEntropyLoss() #loss
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) #backporpagation

# training
EPOCH = 10
for epoch in range(EPOCH):
    train_loss = 0.
    train_acc = 0.
    for step, data in enumerate(train_loader):
        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()#GPU
        # batch_y not one hot
        # out is the probability of each class
        # such as one sample[-1.1009  0.1411  0.0320],need to calculate the max index
        # out shape is batch_size * class
        out = model(batch_x)
        loss = criterion(out, batch_y)
        train_loss += loss.item()
        # pred is the expect class
        # batch_y is the true label
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()# item 将只有一个数的tensor转化成 python scaler返回
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: ', epoch, 'Step', step,
                  'Train_loss: ', train_loss / ((step + 1) * 20), 'Train acc: ', train_acc / ((step + 1) * 20))#

    print('Epoch: ', epoch, 'Train_loss: ', train_loss / len(train_data), 'Train acc: ', train_acc / len(train_data))

torch.save(model, 'net(1).pkl')  # save entire net
torch.save(model.state_dict(), 'net_params(1).pkl')  # save only the parameters

#test
model.eval()
eval_loss=0
eval_acc=0
for step ,data in enumerate(test_loader):
    batch_x,batch_y=data
    batch_x,batch_y=Variable(batch_x),Variable(batch_y)
    # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
    out = model(batch_x)
    loss = criterion(out, batch_y)
    eval_loss += loss.item()
    # pred is the expect class
    # batch_y is the true label
    pred = torch.max(out, 1)[1]
    test_correct = (pred == batch_y).sum()
    eval_acc += test_correct.item()
print( 'Test_loss: ', eval_loss / len(test_data), 'Test acc: ', eval_acc / len(test_data))
