import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryHDC:
    def __unit__(self):
        return

    def generate(self, n, D, device = 'cpu'):
        probs = torch.ones((n, D)) * 0.5
        hypervecs = (2 * torch.bernoulli(probs) - 1).to(device)

        return hypervecs

    def bind(self, vecs_a, vecs_b):
        bound = vecs_a * vecs_b

        return bound

    def bundle(self, vecs):
        bundled = torch.sum(vecs, dim = 0, keepdim = True)

        return bundled

    def similarity(self, vecs_a, vecs_b):
        scores = F.cosine_similarity(vecs_a, vecs_b)

        return scores


def create_vsa(vsa_type):
    if vsa_type == 'binary':
        vsa = BinaryHDC()
    else:
        raise ValueError("invalid vsa type")

    return vsa

def generate_hypervecs(n, D, vsa_type = 'binary', device = 'cpu'):
    vsa = create_vsa(vsa_type)
    hypervecs = vsa.generate(n, D, device)

    return hypervecs

def bind(vecs_a, vecs_b, vsa_type = 'binary'):
    vsa = create_vsa(vsa_type)
    bound = vsa.bind(vecs_a, vecs_b)

    return bound

def bundle(vecs, vsa_type = 'binary'):
    vsa = create_vsa(vsa_type)
    bundled = vsa.bundle(vecs)

    return bundled

def similarity(vecs_a, vecs_b, vsa_type = 'binary'):
    vsa = create_vsa(vsa_type)
    scores = vsa.similarity(vecs_a, vecs_b)

    return scores


