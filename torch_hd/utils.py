import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from torch.utils.data import TensorDataset, DataLoader
import argparse

def load_data(filepath):

    with open(filepath, 'rb') as f:
        data = np.load(f)
    
    return data

def create_dataloader(data, targets, batch_size, nworkers = 2, shuffle = False):
    dataset = TensorDataset(data, targets)
    data_loader = DataLoader(dataset, shuffle = shuffle, batch_size = batch_size, num_workers = nworkers)

    return data_loader


def hd_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--D', type = int, default = 10000, help = 'HD dimensionality')
    parser.add_argument('-e', '--nepochs', type = int, default = 10, help = '# training epochs')
    parser.add_argument('-d', '--device', type = str, default = 'cuda', help = 'device to run on')
    parser.add_argument('-b', '--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('-w', '--nworkers', type = int, default = 8, help = 'number of workers for the dataloader')
    parser.add_argument('-a', '--algo', type = str, default = 'rp', help = 'HD encoding algorithm')
    parser.add_argument('-q', '--qbins', type = int, default = 8, help = '# of quantizer bins')
    parser.add_argument('-r', '--radius', type = float, default = 2.5, help = 'radius for skc')

    return parser

def train_hd(model, classifier, trainloader, nepochs=10, device='cpu', batch_size = 512):
    model = model.to(device)
    classifier = classifier.to(device)
    t = tqdm(range(len(trainloader)))

    classifier.train()

    epoch_acc = 0
    sparsity = 0.0

    for epoch in range(nepochs):
        overall_acc = 0.0
        overall_sum = 0.0

        for idx, batch in enumerate(trainloader):
            t.set_description('Epoch {} train_acc: {} sparseness: {}'.format(epoch, epoch_acc, sparsity))
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            encoded = model(x)
            overall_sum += encoded.sum().cpu().item()
            scores = classifier(encoded, y)
            _, preds = scores.max(dim=1)

            acc = accuracy(preds, y) 
            overall_acc += acc.cpu().item()
            t.update()
        
        overall_acc /= (idx + 1)
        overall_sum /= ((idx + 1) * batch_size * encoded.shape[1])
        epoch_acc = overall_acc
        sparsity = overall_sum
        t.set_description('Epoch {} train_acc: {} sparseness: {}'.format(epoch, epoch_acc, sparsity))
        t.refresh()
        t.reset()
    
    t.close()

    classifier.class_hvs = nn.Parameter(classifier.class_hvs.clamp(-1, 1), requires_grad = False)

    return classifier

def test_hd(model, classifier, testloader, device = 'cuda'):
    classifier = classifier.to(device)

    classifier.eval()
    nclasses = classifier.class_hvs.shape[0]
    overall_acc = 0.0
    mistakes = {}
    t = tqdm(range(len(trainloader)))

    for i in range(nclasses):
        mistakes[i] = 0.0

    for idx, batch in enumerate(testloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        encoded = model(x)
        scores = classifier(encoded, y)
        _, preds = scores.max(dim = 1)

        acc = accuracy(preds, y)
        overall_acc += acc.cpu().item()

        for i in range(nclasses):
            incorrect = y[torch.bitwise_and(preds != y, y == i)]
            mistakes[i] += len(incorrect)

        t.update()
    t.refresh()
    t.close()
    
    overall_acc /= (idx + 1)

    print("---------------------------------")
    print("Test accuracy: {}".format(overall_acc))
    print("#Mistakes by class:")
    for label in range(nclasses):
        print("class {}: {}".format(label, mistakes[label]))
    print("---------------------------------")