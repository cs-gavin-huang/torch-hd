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

```python
torch_hd.hdlayers.RandomProjectionEncoder(dim_in, D = 5000, p = 0.5, dist = 'normal',
	mean = 0.0, std = 1.0, quantize = True)
```

Applies the Random Projection Encoding method. The function creates a projection matrix
sampled from either a bernoulli or normal distribution. The input data passed is flattened
and then projected into hypervector space using the projection matrix. Quantization to
ternary can be applied optionally.
For more details about this method refer [BRIC: Locality-based Encoding for Energy-Efficient Brain-Inspired Hyperdimensional Computing](https://acsweb.ucsd.edu/~j1morris/documents/DAC2019_JusitnMorris_Final.pdf)

The method expects an input in the form of `M x N` where `M` is the number of data points
and `N` is the number of features in each data point.

### Parameters
- **dim_in** (int{: .text-purple-200 }) - the number of input features (`N`)
- **D** (int{: .text-purple-200 }) - The dimensionality of the hypervector. Default: 5000
- **p** (float{: .text-purple-200 }) - If `dist` is `bernoulli` this defines the probability. If `dist` is `normal` then this is the threshold for quantization if `quantize` is set to `True`. This parameter is ignored for non-quantized calls.
- **dist** (string{: .text-purple-200 }) - `'normal', 'bernoulli'`. This sets the distribution from which to sample the projection matrix from. Default: `'normal'`
- **mean** (float{: .text-purple-200 }) - The mean to be used for the normal distribution. Ignored if `dist` is not `normal`. Default: 0.0
- **std** (float{: .text-purple-200 }) - The standard deviation for the normal distribution. Ignored if `dist` is not `normal`. Default: 0.0
- **quantize** (bool{: .text-purple-200 }) - Whether to quantize the projection matrix and the output or not. If set to `True`, all values are quantized to `{1, -1}`


---


