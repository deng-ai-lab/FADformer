# Efficient Frequency-Domain Image Deraining with Contrastive Regularization (ECCV2024)
The official implementation of "Efficient Frequency-Domain Image Deraining with Contrastive Regularization" [ECCV2024]

2024/09/25 Update：The code is now open, welcome to ask questions in the issue, we will try our best to solve errors and doubts
2024/09/26 Update: Pre-trained weights and visualization images are provided by link.

## Authors

- Ning Gao
- Xingyu Jiang
- Xiuhui Zhang
- Yue Deng *

School of Astronautics, Beihang University, Beijing, China

## Abstract

Most current single image-deraining (SID) methods are based on the Transformer with global modeling for high-quality reconstruction. However, their architectures only build long-range features from the spatial domain, which suffers from a significant computational burden to keep effectiveness. Besides, these methods either overlook negative sample information in training or underutilize the rain streak patterns present in the negative ones. To tackle these problems, we propose a Frequency-Aware Deraining Transformer Framework (FADformer) that fully captures frequency domain features for efficient rain removal. Specifically, we construct the FADBlock, including the Fused Fourier Convolution Mixer (FFCM) and Prior-Gated Feed-forward Network (PGFN). Unlike self-attention mechanisms, the FFCM conducts convolution operations in both spatial and frequency domains, endowing it with local-global capturing capabilities and efficiency. Simultaneously, the PGFN introduces residue channel prior in a gating manner to enhance local details and retain feature structure. Furthermore, we introduce a Frequency-domain Contrastive Regularization (FCR) during training. The FCR facilitates contrastive learning in the frequency domain and leverages rain streak patterns in negative samples to improve performance. Extensive experiments show the efficiency and effectiveness of our FADformer.

## Keywords

- SID
- Frequency Learning
- Contrastive Regularization

## Motivation

![Motivation](figs/motivation.png)

## Method

![Method](figs/method.png)

## Results

![Results](figs/result.png)

## Supplementary Material

For more visualizations, see the supplementary material.

## References

Waiting for the publication of ECCV2024

```
