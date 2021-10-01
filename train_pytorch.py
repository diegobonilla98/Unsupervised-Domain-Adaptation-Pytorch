import torch.nn
import numpy as np
from model_pytorch import Model
from BoniDL import utils
from Dataset_pytorch import MNISTSourceDataset, MNISTTargetDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt


BATCH_SIZE = 128
LEARNING_RATE = 1e-3
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 100

model = Model()
print(model)
utils.count_parameters(model)

data_source_loader = DataLoader(MNISTSourceDataset(), batch_size=BATCH_SIZE, shuffle=True)
data_target_loader = DataLoader(MNISTTargetDataset(), batch_size=BATCH_SIZE, shuffle=True)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if USE_CUDA:
    model = model.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in model.parameters():
    p.requires_grad = True

for epoch in range(N_EPOCHS + 1):
    data_source_iter = iter(data_source_loader)
    data_target_iter = iter(data_target_loader)
    i = 0
    while i < len(data_source_loader):
        p = float(i + epoch * len(data_source_loader)) / N_EPOCHS / len(data_source_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        sample = next(data_source_iter)
        s_image, s_label, s_domain = sample['image'], sample['class'], sample['domain']
        s_label = s_label.long()
        s_domain = s_domain.long()

        model.zero_grad()
        if USE_CUDA:
            s_image = s_image.cuda()
            s_label = s_label.cuda()
            s_domain = s_domain.cuda()

        s_image_v = Variable(s_image)
        s_label_v = Variable(s_label)
        s_domain_v = Variable(s_domain)

        s_class_output, s_domain_output = model(input_data=s_image_v, alpha=alpha)
        err_s_label = loss_class(s_class_output, s_label_v)
        err_s_domain = loss_domain(s_domain_output, s_domain_v)

        sample = next(data_target_iter)
        t_image, t_label, t_domain = sample['image'], sample['class'], sample['domain']
        t_label = t_label.long()
        t_domain = t_domain.long()

        model.zero_grad()
        if USE_CUDA:
            t_image = t_image.cuda()
            t_label = t_label.cuda()
            t_domain = t_domain.cuda()

        t_image_v = Variable(t_image)
        t_label_v = Variable(t_label)
        t_domain_v = Variable(t_domain)

        _, t_domain_output = model(input_data=t_image_v, alpha=alpha)
        err_t_domain = loss_domain(t_domain_output, t_domain_v)

        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

    print(f'[Epoch: {epoch}/{N_EPOCHS}, Iter: {i}/{len(data_source_loader)}], [Err_source_label: {err_s_label.cpu().data.numpy()}, Err_source_domain: {err_s_domain.cpu().data.numpy()}, Err_target_domain: {err_t_domain.cpu().data.numpy()}]')

    if epoch % 25 == 0:
        torch.save(model, f'./checkpoints/mnist_mnistm_model_epoch_{epoch}.pth')
