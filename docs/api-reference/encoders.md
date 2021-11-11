---
layout: default
title: Encoders
parent: API Reference
nav_order: 1
---

# HD Encoding functions
{: .no_toc }

This section provides the API reference for the various HDC encoding methods.
{: .fs-6 .fw-300 }

1. TOC
{:toc}

---

## RANDOM PROJECTION ENCODER

<div class="code-example" markdown=1>


<div class="code-example" markdown=1>
### *CLASS*{: .text-blue-300 } &nbsp;&nbsp; `torch_hd.hdlayers.RandomProjectionEncoder(dim_in, D = 5000, p = 0.5, `<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` dist = 'normal', mean = 0.0, std = 1.0, quantize = True)`
{: .no_toc .fs-5 .text-blue-300 }
</div>

Applies the Random Projection Encoding method. The function creates a projection matrix
sampled from either a bernoulli or normal distribution. The input data passed is flattened
and then projected into hypervector space using the projection matrix. Quantization to
ternary can be applied optionally.
For more details about this method refer [BRIC: Locality-based Encoding for Energy-Efficient Brain-Inspired Hyperdimensional Computing](https://acsweb.ucsd.edu/~j1morris/documents/DAC2019_JusitnMorris_Final.pdf)

The method expects an input in the form of `N x M` where `N` is the number of data points
and `M` is the number of features in each data point.


### Parameters
{: .no_toc }
- **dim_in**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - the number of input features (`N`)
- **D**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - The dimensionality of the hypervector. Default: 5000
- **p**{: .text-blue-100 } (*`float`*{: .fs-5 .text-purple-200 }) - If `dist` is `bernoulli` this defines the probability. If `dist` is `normal` then this is the threshold for quantization if `quantize` is set to `True`. This parameter is ignored for non-quantized calls.
- **dist**{: .text-blue-100 } (*`string`*{: .fs-5 .text-purple-200 }) - `'normal', 'bernoulli'`. This sets the distribution from which to sample the projection matrix from. Default: `'normal'`
- **mean**{: .text-blue-100 } (*`float`*{: .fs-5 .text-purple-200 }) - The mean to be used for the normal distribution. Ignored if `dist` is not `normal`. Default: 0.0
- **std**{: .text-blue-100 } (*`float`*{: .fs-5 .text-purple-200 }) - The standard deviation for the normal distribution. Ignored if `dist` is not `normal`. Default: 0.0
- **quantize**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - Whether to quantize the projection matrix and the output or not. If set to `True`, all values are quantized to `{1, -1}`. This flag also decides whether the output hypervector is quantized or not. If set to `False` then all values will be `float`.


### Shape
{: .no_toc }

- **input**{: .text-blue-100 }: `N x M` where `N` is the number of data points and `M` 
is the number of features in each data point
- **output**{: .text-blue-100 }: `N x D` `D` is the dimensionality of the hypervector

</div>

--- 


## ID-LEVEL ENCODER

<div class="code-example" markdown=1>

<div class="code-example" markdown=1>
### *CLASS*{: .text-blue-300 } &nbsp;&nbsp; `torch_hd.hdlayers.IDLevelEncoder(dim_in, D, qbins = 16, max_val = None, `<br/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` min_val = None, sparsity = None, quantize=True)`
{: .no_toc .text-blue-300 }
</div>

This method implements the ID-level encoding scheme. Each feature is assigned a random
hypervector called the ID hypervector, and for mapping the continuous values that these features
can take, continuous item memory is used. The values are quantized into bins and each bin
is assigned  hypervector (Level hypervector) that's corelated with it's neighbours based on difference in magnitude.
For each data point, the binding operation is used to form a conjunction of features(ID hypervectors) 
and their corresponding values (Level hypervectors) all of which are bundled together.


For more details about this method refer [Hyperdimensional Biosignal Processing: A Case Study for EMG-based Hand Gesture Recognition](https://iis-people.ee.ethz.ch/~arahimi/papers/ICRC16.pdf)

### Parameters
{: .no_toc }
- **dim_in**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - the number of input features (`N`)
- **D**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - The dimensionality of the hypervector. Default: 5000
- **qbins**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - Number of quantization bins. This decides the level of quantization for each feature value
- **max_val**{: .text-blue-100 } (*`float`*{: .fs-5 text-purple-200 }) - The maximum of all feature values. This is used for setting quantization limits
- **min_val**{: .text-blue-100 } (*`float`*{: .fs-5 text-purple-200 }) - The minimum of all feature values. This is used for setting quantization limits
- **sparsity**{: .text-blue-100 } (*`float`*{: .fs-5 .text-purple-200 }) - When the `sparsify` flag is set, it maintains the sparsity of hypervectors to this value. Defaul: 0.5
- **quantize**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - Whether to quantize the hypervector or not. If set, the hypervectors are clipped to `{-1, 1}`. 
If not set, the hypervectors will be integers. Default: True


### Shape
{: .no_toc }

- **input**{: .text-blue-100 }: `N x M` where `N` is the number of data points and `M` 
is the number of features in each data point
- **output**{: .text-blue-100 }: `N x D` `D` is the dimensionality of the hypervector

</div>

--- 

## HD CLASSIFIER

<div class="code-example" markdown=1>

<div class="code-example" markdown=1>
### *CLASS*{: .text-blue-300 } `nclasses, D, alpha = 1.0, clip = False, cdt = False, k = 10, sparsity=0.5`
{: .no_toc .text-blue-300 }
</div>

This layer implements the perceptron based classifier that's typically used for HD learning.
It takes as input the encoded data points and optionally the labels (during training). If the labels
are provided the layer operates in training mode and if not it performs inference. 

The encoded hypervectors are bundled together by class initially to perform one-shot learning
and generate the initial class hypervectors on the first batch passed. After that, during each round of training
if a data point is misclassified, the corresponding hypervector is subtracted from the wrong class
and added to the correct class. A learning rate is used during the updates of the class hypervectors to
control the rate of learning.

For more details about this method refer [VoiceHD: Hyperdimensional Computing for Efficient Speech Recognition
](https://iis-people.ee.ethz.ch/~arahimi/papers/ICRC16.pd://ieeexplore.ieee.org/document/8123650)

### Parameters
{: .no_toc }
- **nclasses**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - the number of classes
- **D**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - hypervector dimensionality
- **alpha**{: .text-blue-100 } (*`float`*{: .fs-5 .text-purple-200 }) - learning rate. Default: 1.0
- **clip**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - If set to `True` the class hypervectors are quantized to `{1, -1}`. Default: False
- **cdt**{: .text-blue-100 } (*`bool`*{: .fs-5 .text-purple-200 }) - If set to `True` Context Dependent Thinning (CDT) is applied to ensure that
the sparsity of the class hypervectors are maintained. This avoids saturation of hypervectors when large number of vectors are bundled together. Default: False
- **k**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - Number of rounds of CDT to be applied. This appplies only when `cdt = True`.	
- **sparsity**{: .text-blue-100 } (*`float`*{: .fs-5 .text-purple-200 }) - Desired sparsity of the class hypervectors. This appplies only when `cdt = True`.	


### Shape
{: .no_toc }

- **input**{: .text-blue-100 }: Takes as input encoded hypervectors to be classified and optionally
the labels during training
	- *encoded*: `N x D` where `N` is the number of data points and `D` is the hypervector dimensionality.
	- *targets*: `N x 1` these are the labels which need to be passed during training. The labels should be of the form `0, 1 .. nclasses-1`.
**Note**: Labels are required only during training. If the labels are not passed the classifier operates in inference mode and outputs the similarity scores.

- **output**{: .text-blue-100 }: `N x nclasses` outputs the similarity score for each class hypervector

</div>

--- 



