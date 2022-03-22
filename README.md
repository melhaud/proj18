# Anomaly detection using self-supervised point clouds
 ---

* Agafonova Ekaterina (<ekaterina.agafonova@skoltech.ru>)
* Volkov Dmitry (<dmitry.volkov@skoltech.ru>)
* Sidnov Kirill (<kirill.sidnov@skoltech.ru>)
* Dembitskiy Artem (<artem.dembitsky@skoltech.ru>)

**TA: Nikita Balabin**
---
## Description
### Powered by
1. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
2. [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230)
3. [Manifold Topology Divergence: a Framework for Comparing Data Manifolds](https://arxiv.org/pdf/2106.04024.pdf)
4. [Lightly](https://github.com/lightly-ai/lightly)

### Models
**SimCLR.** In a Simple framework for Contrastive Learning of visual Representations (“SimCLR”) two separate data augmentation operators are sampled from the same family of augmentations and applied to each data example to obtain two correlated views.

**Barlow Twins.** The  objective function in “Barlow Twins” measures the cross-correlation matrix between the embeddings of two identical networks fed with distorted versions of a batch of samples, and tries to make this matrix close to the identity.

### Metrics
**[Hausdorff distance.](https://doi.org/10.1109/tpami.2015.2408351)** The Hausdorff distance (HD) between two point sets is a commonly used dissimilarity measure for comparing point sets and image segmentations. 

**[MTopDiv.](https://doi.org/10.48550/arXiv.2106.04024)** Manifold Topology Divergence is a framework for comparing data manifolds, aimed, in particular, towards the evaluation of deep generative models. 
