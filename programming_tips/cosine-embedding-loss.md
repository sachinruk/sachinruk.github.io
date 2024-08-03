---
layout: post
title: "Thoughts on CosineEmbeddingLoss"
modified:
categories: pytorch
excerpt: Explanation of Cosine Embedding Loss in Pytorch

date: 2023-08-15
share: true
ads: true
---

$$
\begin{align}
\text{loss}(x, y) = 
\begin{cases} 
1 - \cos(x_1, x_2) & \text{if } y = 1 \\
\max(0, \cos(x_1, x_2) - \text{margin}) & \text{if } y = -1
\end{cases}
\end{align}
$$

[nn.CosineEmbeddingLoss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) needs 3 arguments, the predicted embedding ($x_1$), label embedding ($x_2$) as well as a label ($y$) indicating that we need to move these embeddings closer ($y = 1$) and further if not ($y = 0$).

The distance between two points ($1 - \cos(x_1, x_2)$) is minimised, while in the $y = 0$ case, the similarity $\cos(x_1, x_2)$ is minimised. However, if the similarity is lower than a certain margin (defaulting to 0.5) nothing happens. This is due to the clipping that happens with the `max` function. This causes these examples to have zero gradient.
