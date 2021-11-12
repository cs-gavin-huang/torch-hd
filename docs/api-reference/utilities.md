---
layout: default
title: Utilities
parent: API Reference
nav_order: 2
---

# Utilities
{: .no_toc }

This section provides the API reference for various utility functions that are part of
the package
{: .fs-6 .fw-300 }

1. TOC
{:toc}

---

## CREATE DATALOADER

<div class="code-example" markdown=1>


<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } &nbsp;&nbsp; `torch_hd.utils.create_dataloader(data, targets, batch_size, `<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` nworkers = 2, shuffle = False)`
{: .no_toc .fs-5 .text-blue-300 }
</div>

This function creates a pytorch dataloader from numpy arrays. 

### Parameters
{: .no_toc }
- **data**{: .text-blue-100 } (*`numpy array`*{: .fs-5 .text-purple-200 }) - The input features of the form `N x M` where `N` is the number
of data points and `M` is the number of features
- **targets**{: .text-blue-100 } (*`numpy array`*{: .fs-5 .text-purple-200 }) - The labels for the data of the form `N x 1` where each label
is a number indicating the class `0, 1, ... nclasses - 1`
- **batch_size**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - The batch_size for the dataloader
- **nworkers**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - The number of workers to use for the dataloader. Default: `2`
- **shuffle**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - This flag controls if the data is shuffled or not. Default: `False`

### Returns
{: .no_toc }

- PyTorch DataLoader

</div>

--- 

## HD TRAINER

<div class="code-example" markdown=1>


<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } &nbsp;&nbsp; `train_hd(encoder, classifier, trainloader, process_batch = None, `<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`nepochs=10, device='cpu', mode='norm', valloader=None):`
{: .no_toc .fs-5 .text-blue-300 }
</div>
This method implements a simple PyTorch training loop which iterates over the data in batches and
trains the model. If a validation dataloader is passed, the function also performs checking over validation data and prints val accuracy.
The function returns the trained hdmodel.

### Parameters
{: .no_toc }
- **encoder**{: .text-blue-100 } (*`torch_hd.hdlayers.<encoder-type>Encoder`*{: .fs-5 .text-purple-200 }) - The encoder layer. See [Encoders](encoders)
- **classifier**{: .text-blue-100 } (*`torch_hd.hdlayers.hd_classifier`*{: .fs-5 .text-purple-200 }) - The `hd_classifier` layer. See [Encoders](encoders)
- **trainloader**{: .text-blue-100 } (*`torch.utils.data.DataLoader`*{: .fs-5 .text-purple-200 }) - The pytorch train dataloader. Use [`create_dataloader`](#CREATE DATALOADER)
to generate dataloader
- **process_batch**{: .text-blue-100 } (*`python function`*{: .fs-5 .text-purple-200 }) - any preprocessing to be done to the data before 
feeding it to the encoder. Pass a method. This will be called just before HD encoding.
- **nepochs**{: .text-blue-100 } (*`python function`*{: .fs-5 .text-purple-200 }) - Number of epochs to train for. Default: 10
- **device**{: .text-blue-100 } (*`string`*{: .fs-5 .text-purple-200 }) - device to train on. One of `'cpu', 'cuda'`. You can also specify specific GPUs by passing `cuda:0`. Default: `'cpu'`
- **norm**{: .text-blue-100 } (*`string`*{: .fs-5 .text-purple-200 }) - One of `'norm', 'clip', 'None'`. Default: `'norm'`
	- `norm` normalizes the class hypervectors at the end of training (most commonly used in literature and the default setting
	- `'clip'` the class hypervectors are clipped to `[-1, 1]` at the end of training
	- `'None'` the class hypervectors are left as integers
- **valloader**{: .text-blue-100 } (*`torch.utils.data.DataLoader`*{: .fs-5 .text-purple-200 }) - (*optional*) The pytorch validation dataloader 


### Returns
{: .no_toc }

- trained hd_classifier

</div>

--- 

## TEST HD (INFERENCE)

<div class="code-example" markdown=1>


<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } &nbsp;&nbsp; `test_hd(encoder, classifier, testloader,process_batch = None, `<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`device = 'cpu', show_mistakes = False, cm = False)`
{: .no_toc .fs-5 .text-blue-300 }
</div>

This function implements the testing pipeline for a trained `hd_classifier` model. The function takes as input a dataloader
and prints the overall test accuracy.

### Parameters
{: .no_toc }
- **encoder**{: .text-blue-100 } (*`torch_hd.hdlayers.<encoder-type>Encoder`*{: .fs-5 .text-purple-200 }) - The encoder layer. This must be the same encoder used for training
- **classifier**{: .text-blue-100 } (*`torch_hd.hdlayers.hd_classifier`*{: .fs-5 .text-purple-200 }) - The trained `hd_classifier` layer returned by `train_hd`. See [Encoders](encoders)
- **testloader**{: .text-blue-100 } (*`torch.utils.data.DataLoader`*{: .fs-5 .text-purple-200 }) - The pytorch test dataloader
- **process_batch**{: .text-blue-100 } (*`python function`*{: .fs-5 .text-purple-200 }) - any preprocessing to be done to the data before 
feeding it to the encoder. Pass a method. This will be called just before HD encoding.
- **device**{: .text-blue-100 } (*`string`*{: .fs-5 .text-purple-200 }) - device to train on. One of `'cpu', 'cuda'`. You can also specify specific GPUs by passing `cuda:0`. Default: `'cpu'`
- **show_mistakes**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - Setting this flag to `True` will show the number of mistakes for each class
- **cm**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - Setting this flag to `True` will generate a confusion matrix using seaborn and save the figure to the current directory

</div>

--- 


