---
layout: default
title: hdfunctional
parent: V1.0.4
grand_parent: API Reference
nav_order: 2
---

# Hyperdimensional Computing operations
{: .no_toc }

This section provides the API reference for various HD operations for different types of
Vector Symbolic Architectures (VSA). A survey and explanation of different types of VSAs
can be found in this [paper](https://arxiv.org/abs/2001.11797). Right now only the binary
(-1 ,1) VSA is supported. This VSA type is referred to in the paper as MAP-I.

NOTE: All functions support both CPU and GPU.

{: .fs-6 .fw-300 }

1. TOC
{:toc}

---

## generate_hypervectors

<div class="code-example" markdown=1>


<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } &nbsp;&nbsp; `torch_hd.hdfunctional.generate(n, D, vsa_type, device)`
{: .no_toc .fs-5 .text-blue-300 }
</div>

Generates hypervectors according to the VSA type chosen. Right now only binary VSA is chosen.
Note that this type is equivalent to the MAP-I type in 
[A comparison of Vector Symbolic Architectures](https://arxiv.org/abs/2001.11797)


### Parameters
{: .no_toc }
- **n**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - the number of hypervectors to generate.
- **D**{: .text-blue-100 } (*`int`*{: .fs-5 .text-purple-200 }) - The dimensionality of the hypervector.
- **vsa_type**{: .text-blue-100 } (*`str`*{: .fs-5 .text-purple-200 }) - The type of VSA to use. Default: `binary`
- **device**{: .text-blue-100 } (*`str`*{: .fs-5 .text-purple-200 }) - Device to create vectors on. `cpu` or `cuda` Default: `cpu`


### Shape
{: .no_toc }

**output**{: .text-blue-100 }: `n x D`

</div>

--- 


## bind

<div class="code-example" markdown=1>

<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } &nbsp;&nbsp; `torch_hd.hdfunctional.bind(vecs_a, vecs_b, vsa_type)`
{: .no_toc .text-blue-300 }
</div>

Implements the binding operation. The function takes as input `vecs_a` and `vecs_b` and
binds or associates them. Either both `vecs_a` and `vecs_b` should have the same dimensions
or one of them can be of shape `1 x D` where `D` is the dimensionality of the hypervector.


Refer to [A comparison of Vector Symbolic Architectures](https://arxiv.org/abs/2001.11797)
for details on the bind operator for specific VSA types.


### Parameters
{: .no_toc }
- **vecs_a**{: .text-blue-100 } (*`tensor`*{: .fs-5 .text-purple-200 }) - First set of hypervectors
- **vecs_b**{: .text-blue-100 } (*`tensor`*{: .fs-5 .text-purple-200 }) - Second set of hypervectors
- **vsa_type**{: .text-blue-100 } (*`str`*{: .fs-5 .text-purple-200 }) - The type of VSA to use. Default: `binary`


### Shape
{: .no_toc }

- **vecs_a**{: .text-blue-100 }: `N x D` where `N` is the number of hypervectors or `1 x D`
- **vecs_b**{: .text-blue-100 }: `N x D` where `N` is the number of hypervectors or `1 x D`
- **output**{: .text-blue-100 }: `N x D` 


</div>

--- 

## bundle

<div class="code-example" markdown=1>

<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } `torch_hd.hdfunctional.bundle(vecs, vsa_type)`
{: .no_toc .text-blue-300 }
</div>

The bundle function implements the bundling operator.

Refer to [A comparison of Vector Symbolic Architectures](https://arxiv.org/abs/2001.11797)
for details on the bundling operator for specific VSA types.

### Parameters
{: .no_toc }
- **vecs**{: .text-blue-100 } (*`tensor`*{: .fs-5 .text-purple-200 }) - the hypervectors to bundle
- **vsa_type**{: .text-blue-100 } (*`str`*{: .fs-5 .text-purple-200 }) - The type of VSA to use. Default: `binary`


### Shape
{: .no_toc }

- **input**{: .text-blue-100 }: `N x D` where `N` is the number of hypervectors
- **output**{: .text-blue-100 }: `1 x D` 

</div>

--- 

## similarity

<div class="code-example" markdown=1>

<div class="code-example" markdown=1>
### *FUNCTION*{: .text-blue-300 } `torch_hd.hdfunctional.similarity(vecs_a, vecs_b, vsa_type)`
{: .no_toc .text-blue-300 }
</div>

This function implements the similarity check for the specific VSA type used.
Either both `vecs_a` and `vecs_b` should have the same dimensions
or one of them can be of shape `1 x D` where `D` is the dimensionality of the hypervector.

Refer to [A comparison of Vector Symbolic Architectures](https://arxiv.org/abs/2001.11797)
for details on the similarity method for specific VSA types.

### Parameters
{: .no_toc }
- **vecs_a**{: .text-blue-100 } (*`tensor`*{: .fs-5 .text-purple-200 }) - First set of hypervectors
- **vecs_b**{: .text-blue-100 } (*`tensor`*{: .fs-5 .text-purple-200 }) - Second set of hypervectors
- **vsa_type**{: .text-blue-100 } (*`str`*{: .fs-5 .text-purple-200 }) - The type of VSA to use. Default: `binary`


### Shape
{: .no_toc }

- **vecs_a**{: .text-blue-100 }: `N x D` where `N` is the number of hypervectors or `1 x D`
- **vecs_b**{: .text-blue-100 }: `N x D` where `N` is the number of hypervectors or `1 x D`
- **output**{: .text-blue-100 }: `N,` 

</div>

--- 



