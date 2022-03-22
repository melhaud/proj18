# Anomaly detection using self-supervised point clouds

* Agafonova Ekaterina (<ekaterina.agafonova@skoltech.ru>)
* Volkov Dmitry (<dmitry.volkov@skoltech.ru>)
* Sidnov Kirill (<kirill.sidnov@skoltech.ru>)
* Dembitskiy Artem (<artem.dembitsky@skoltech.ru>)

TA: Nikita Balabin

## Powered by
1. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
2. [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230)
3. [Manifold Topology Divergence: a Framework for Comparing Data Manifolds](https://arxiv.org/pdf/2106.04024.pdf)
4. [Lightly](https://github.com/lightly-ai/lightly)
<<<<<<< HEAD

## Description

The work explores the problem of point clouds similarity estimation in an SSL framework. We have compared the performance of  Hausdorff and MTopDiv metrics on CIFAR10 dataset using embeddings extracted from linear layer of BarlowTwins and SimCLR models (implemented in Lightly). To extract the embeddings, we used a single-class learning strategy. We claim that the metrics have failed to distinguish embeddings of the augmented classes due to low robustness to non-rigid augmentations.

## Augmentations

We use set of augmentation suggested by the authors of lightly for training of the model and implemented in their ImageCollateFunction (we use default parameters and input size of the CIFAR10 images - 32, the full list of augmentation can be found in lightly documentation).

## Models
**SimCLR.** In a Simple framework for Contrastive Learning of visual Representations (“SimCLR”) two separate data augmentation operators are sampled from the same family of augmentations and applied to each data example to obtain two correlated views.

**Barlow Twins.** The  objective function in “Barlow Twins” measures the cross-correlation matrix between the embeddings of two identical networks fed with distorted versions of a batch of samples, and tries to make this matrix close to the identity.

## Metrics
We have picked and evaluated metrics using several criteria:
1. Value of metric will allow measuring how our generated by SSL embedding representation stable to data argumentation. In other word from the view point of metric SLL capable to produce close enough embeddings to augmented images of the same class. 
2. Numerical value of the metrics will allow distinguishing images of the different classes, i.e. images which we consider as an anomaly. 
3. Metrics evaluation should not be computationally demanding.

**[Hausdorff distance.](https://doi.org/10.1109/tpami.2015.2408351)** The Hausdorff distance (HD) between two point sets is a commonly used dissimilarity measure for comparing point sets and image segmentations. 

**[MTopDiv.](https://doi.org/10.48550/arXiv.2106.04024)** Manifold Topology Divergence is a framework for comparing data manifolds, aimed, in particular, towards the evaluation of deep generative models. 

## Environment Requirements

All the tests were performed using Google Colab GPU's Tesla P100.

## Dependencies

The requred packages can be installed from ``requirements.txt``:

        pip install -r requirements.txt

Note that Ripser++, which is an MTopDiv requirement, will work on GPU only.


## Description

The work explores the problem of point clouds similarity estimation in an SSL framework. We have compared the performance of  Hausdorff and MTopDiv metrics on CIFAR10 dataset using embeddings extracted from linear layer of BarlowTwins and SimCLR models. To extract the embeddings, we used a single-class learning strategy. We claim that the metrics have failed to distinguish embeddings of the augmented classes due to low robustness to non-rigid augmentations.

## Augmentations

We use set of augmentation suggested by the authors of lightly for training of the model and implemented in their ImageCollateFunction (we use default parameters and input size of the CIFAR10 images - 32, the full list of augmentation can be found in lightly documentation).

## Models
**SimCLR.** In a Simple framework for Contrastive Learning of visual Representations (“SimCLR”) two separate data augmentation operators are sampled from the same family of augmentations and applied to each data example to obtain two correlated views.

**Barlow Twins.** The  objective function in “Barlow Twins” measures the cross-correlation matrix between the embeddings of two identical networks fed with distorted versions of a batch of samples, and tries to make this matrix close to the identity.

## Metrics
We have picked and evaluated metrics using several criteria:
1. Value of metric will allow measuring how our generated by SSL embedding representation stable to data argumentation. In other word from the view point of metric SLL capable to produce close enough embeddings to augmented images of the same class. 
2. Numerical value of the metrics will allow distinguishing images of the different classes, i.e. images which we consider as an anomaly. 
3. Metrics evaluation should not be computationally demanding.

**[Hausdorff distance.](https://doi.org/10.1109/tpami.2015.2408351)** The Hausdorff distance (HD) between two point sets is a commonly used dissimilarity measure for comparing point sets and image segmentations. 

**[MTopDiv.](https://doi.org/10.48550/arXiv.2106.04024)** Manifold Topology Divergence is a framework for comparing data manifolds, aimed, in particular, towards the evaluation of deep generative models. 

## Environment Requirements

All the tests were performed using Google Colab GPU's Tesla P100.

## Dependencies

The requred packages can be installed from ``requirements.txt``:

        pip install -r requirements.txt


**Note** that Ripser++, which is an MTopDiv requirement, will install and work on GPU only.
