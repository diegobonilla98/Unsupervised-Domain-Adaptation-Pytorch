import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import gzip
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class MNISTSourceDataset(Dataset):
    def __init__(self, is_test=False):
        self.transform = transforms.Compose([self.ToTensor()])
        if is_test:
            (_, _), (self.x_mnist_train, self.y_train) = mnist.load_data()
        else:
            (self.x_mnist_train, self.y_train), (_, _) = mnist.load_data()
        self.x_mnist_train = np.stack((self.x_mnist_train,) * 3, axis=-1).astype('float32') / 255.

        idx = np.arange(self.x_mnist_train.shape[0])
        np.random.shuffle(idx)

        self.x_mnist_train = self.x_mnist_train[idx]
        self.y_train = self.y_train[idx]

    @staticmethod
    class ToTensor(object):
        def __call__(self, sample):
            image, label, domain = sample['image'], sample['class'], sample['domain']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'class': torch.from_numpy(label),
                    'domain': torch.from_numpy(domain)}

    def __len__(self):
        return self.x_mnist_train.shape[0]

    def __getitem__(self, idx):
        x = self.x_mnist_train[idx]
        domain = np.array(0.)

        y = np.array(self.y_train[idx])
        sample = {'image': x, 'class': y, 'domain': domain}
        sample = self.transform(sample)
        return sample


class MNISTTargetDataset(Dataset):
    def __init__(self, is_test=False):
        self.transform = transforms.Compose([self.ToTensor()])
        with gzip.open('/media/bonilla/My Book/housenumbers/keras_mnistm.pkl.gz', "rb") as f:
            a = pickle.load(f, encoding="latin-1")
            if is_test:
                self.x_color_train = a['test'].astype('float32') / 255.
            else:
                self.x_color_train = a['train'].astype('float32') / 255.

        if is_test:
            (_, _), (_, self.y_train) = mnist.load_data()
        else:
            (_, self.y_train), (_, _) = mnist.load_data()

        idx = np.arange(self.x_color_train.shape[0])
        np.random.shuffle(idx)

        self.x_color_train = self.x_color_train[idx]
        self.y_train = self.y_train[idx]

    @staticmethod
    class ToTensor(object):
        def __call__(self, sample):
            image, label, domain = sample['image'], sample['class'], sample['domain']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'class': torch.from_numpy(label),
                    'domain': torch.from_numpy(domain)}

    def __len__(self):
        return self.x_color_train.shape[0]

    def __getitem__(self, idx):
        x = self.x_color_train[idx]
        domain = np.array(1.)

        y = np.array(self.y_train[idx])
        sample = {'image': x, 'class': y, 'domain': domain}
        sample = self.transform(sample)
        return sample
