---
layout: default
title: Examples
nav_order: 3
description: "Torch-HD"
permalink: /examples
---

# Examples
{: .no_toc }

This section has a few examples for demonstrating the HD computing functions with Torch-HD.
I will try to add more examples in the future. But for now we will train a simple classifier
using principles of Hyperdimensional computing

1. TOC
{:toc}

# Training a classifier
You must be wondering how to get started with using this package and what sort of tasks
you can perform with them. To start with let's try to build a simple Hyperdimensional 
classifier.

Training a HDC classifier will broadly involve the following steps:
1. First we load the data and create a DataLoader
2. Next we define an HD encoder for encoding the data into hypervectors
3. Instantiate the HD classifier
4. Train the HD model on training data
5. Evaluate the trained model on test data

## Data
{: .no_toc }
For this example we are going to train a classifier for the MNIST image dataset.
The MNIST is one of the most basic datasets for machine learning problems and is a good
example to demonstrate the simplicity of using Torch-HD.

## 1. Load data and create DataLoader
We make use of PyTorch `torchvision` library to load the MNIST dataset. `torchvision`
takes care of the boiler plate code for loading the raw dataset and preprocessing it to
the required format.

```python
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
```

We create a transform to convert the raw data into tensors. Now the `trainset` and `testset`
contains the data in the form of a touple of `(data, labels)` where the data is of shape
`no. of data points x 28 x 28 x 1` and the labels of shape `no. of data points x 1`.
Now we create a DataLoader in-order to batch the data.

```python
train_loader = DataLoader(trainset, batch_size = 512, shuffle = False)
test_loader = DataLoader(testset, batch_size = 512, shuffle = False)
```

## 2. Define encoder and classifier
Now that we have loaded the data, the next step is to define the HD model. The HD model
consists of 2 components, a HD encoder and the classifier. We will be using the
*Random Projection Encoding* to encode the data.

```python
encoder = nn.Sequential(
        nn.Flatten(),
        hd.RandomProjectionEncoder(dim_in = 784, D = 10000, dist = 'bernoulli')
    )
classifier = hd.HDClassifier(nclasses = 10, D = 10000)
```

Just like regular PyTorch models, you can create a sequence of layers using `nn.Sequential`.
In our case, remember that the data was of the shape `npoints x 28 x 28 x 1`. However,
for the random projection encoding we expect the data to be of the form `npoints x 784`
which is basically the image flattened. Hence we use a `Flatten()` layer before the encoder.

**Note:** We don't include the classifier in the `Sequential` model because during training
`hd.HDClassifier` expects both the encoded hypervectors and the labels. However, the other layers `nn.Flatten(),
hd.RandomProjectionEncoder()` takes as input only the features.

## 3. Training and testing the model
This is it. We have loaded the data and defined the model. We will now train the model then
evaluate it on the test data.

```python
trained_model = utils.train_hd(encoder, model, train_loader, valloader = test_loader, nepochs = 5)
utils.test_hd(encoder, trained_model, test_loader)
```

### Output
```bash
---------------------------------
Test accuracy: 96.57

---------------------------------
```

## 3. Full code
Here's the full code

```python
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
        hd.RandomProjectionEncoder(dim_in = 784, D = 10000, dist = 'bernoulli')
    )
model = hd.HDClassifier(nclasses = 10, D = 10000)

trained_model = utils.train_hd(encoder, model, train_loader, valloader = test_loader, nepochs = 5)
utils.test_hd(encoder, trained_model, test_loader)
```




