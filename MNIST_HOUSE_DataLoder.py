from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets


class MNIST_HOUSE_DataLoader(Dataset):
    def __init__(self):
        self.mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
