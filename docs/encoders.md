---
layout: default
title: Encoders
nav_order: 2
---

# HD Encoding functions
{: .no_toc }

This section provides the API reference for the various HDC encoding methods.
{: .fs-6 .fw-300 }

1. TOC
{:toc}

---

## Random Projection Encoder

<div class="code-example" markdown=1>

<h3>
```python
torch_hd.hdlayers.RandomProjectionEncoder(dim_in, D = 5000, p = 0.5, dist = 'normal',
	mean = 0.0, std = 1.0, quantize = True)
```
</h3>


Applies the Random Projection Encoding method. The function creates a projection matrix
sampled from either a bernoulli or normal distribution. The input data passed is flattened
and then projected into hypervector space using the projection matrix. Quantization to
ternary can be applied optionally.
For more details about this method refer [BRIC: Locality-based Encoding for Energy-Efficient Brain-Inspired Hyperdimensional Computing](https://acsweb.ucsd.edu/~j1morris/documents/DAC2019_JusitnMorris_Final.pdf)

The method expects an input in the form of `M x N` where `M` is the number of data points
and `N` is the number of features in each data point.

---

### Parameters
{: .no_toc }
- **dim_in**{: .text-blue-100 } (*int*{: .text-purple-200 }) - the number of input features (`N`)
- **D**{: .text-blue-100 } (*int*{: .text-purple-200 }) - The dimensionality of the hypervector. Default: 5000
- **p**{: .text-blue-100 } (*float*{: .text-purple-200 }) - If `dist` is `bernoulli` this defines the probability. If `dist` is `normal` then this is the threshold for quantization if `quantize` is set to `True`. This parameter is ignored for non-quantized calls.
- **dist**{: .text-blue-100 } (*string*{: .text-purple-200 }) - `'normal', 'bernoulli'`. This sets the distribution from which to sample the projection matrix from. Default: `'normal'`
- **mean**{: .text-blue-100 } (*float*{: .text-purple-200 }) - The mean to be used for the normal distribution. Ignored if `dist` is not `normal`. Default: 0.0
- **std**{: .text-blue-100 } (*float*{: .text-purple-200 }) - The standard deviation for the normal distribution. Ignored if `dist` is not `normal`. Default: 0.0
- **quantize**{: .text-blue-100 } (*bool*{: .text-purple-200 }) - Whether to quantize the projection matrix and the output or not. If set to `True`, all values are quantized to `{1, -1}`



