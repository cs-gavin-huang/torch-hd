from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch_hd.hdlayers as hd
import torch_hd.utils as utils
import torch.nn as nn

transforms = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST(root='./', download = True, train = True, transform = transforms)
testset = MNIST(root='./', download = True, train = False, transform = transforms)

train_loader = DataLoader(trainset, batch_size = 512, shuffle = False)
test_loader = DataLoader(testset, batch_size = 512, shuffle = False)

encoder = nn.Sequential(
        nn.Flatten(),
        hd.RandomProjectionEncoder(dim_in = 784, D = 10000)
    )
model = hd.HDClassifier(nclasses = 10, D = 10000)

trained_model = utils.train_hd(encoder, model, train_loader, valloader = test_loader, nepochs = 5) 

utils.test_hd(encoder, trained_model, test_loader)

