---
layout: post
title: "Pytorch Lightning Learning rate finder"
modified:
categories: pytorch
excerpt: Get learning rate

date: 2023-01-01
share: true
ads: true
---

The following plots the loss against learning rate in order to discover an "optimal" learning rate.
```python
lr_finder = trainer.tuner.lr_find(lightning_module, train_dataloaders=train_dl)
fig = lr_finder.plot(suggest=True)
fig.show()
```

When all else fails, 1e-3.
