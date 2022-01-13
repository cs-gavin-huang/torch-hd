import torch_hd.hdfunctional as F
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dims = np.arange(32, 0, -2) ** 2
ks = np.arange(2, 51)
output_array = np.zeros((len(dims), len(ks)))

for expts in range(10):
    for d_idx, d in enumerate(dims):
        hypervecs = F.generate_hypervecs(1000, d)

        for k_idx, k in enumerate(ks):
            k_hypervecs = hypervecs[:k]
            bundled_vecs = F.bundle(k_hypervecs)
            scores = F.similarity(bundled_vecs, hypervecs)
            _, retrieved = torch.topk(scores, k)
            retrieved, _  = torch.sort(retrieved)
            k_indices = np.arange(0, k)

            acc = np.sum(k_indices == retrieved.numpy()) / k
            output_array[d_idx, k_idx] += acc

output_array /= 10
sns.heatmap(output_array, yticklabels = dims)
plt.ylabel("# dimensions")
plt.xlabel("# of bundled vecs")
plt.show()
