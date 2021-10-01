import torch.nn
import numpy as np
from model_pytorch import Model
from BoniDL import utils
from Dataset_pytorch import MNISTSourceDataset, MNISTTargetDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt


USE_CUDA = torch.cuda.is_available()

data_source = MNISTSourceDataset(is_test=True)
data_target = MNISTTargetDataset(is_test=True)

weights_path = './checkpoints/mnist_mnistm_model_epoch_75.pth'
model = torch.load(weights_path)
model = model.eval()

if USE_CUDA:
    model = model.cuda()

for i in range(10):
    sample = data_source[i]
    s_image, s_label, s_domain = sample['image'], sample['class'], sample['domain']
    s_label = s_label.long().unsqueeze(0)
    s_domain = s_domain.long().unsqueeze(0)
    s_image = s_image.unsqueeze(0)

    if USE_CUDA:
        s_image = s_image.cuda()
        s_label = s_label.cuda()
        s_domain = s_domain.cuda()

    s_image_v = Variable(s_image)
    s_label_v = Variable(s_label)
    s_domain_v = Variable(s_domain)

    class_output, _ = model(input_data=s_image_v, alpha=0)
    pred = class_output.data.max(1, keepdim=True)[1]

    y_true = s_label_v.cpu().data.numpy().flatten()[0]
    y_pred = pred.cpu().data.numpy().flatten()[0]
    print(f"y_true: {y_true}, y_pred: {y_pred}")
