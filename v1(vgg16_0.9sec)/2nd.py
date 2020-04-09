import torchvision.models as models
import torch

torch.utils.model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='/')
