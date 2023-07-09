---
layout: post
title: "Timm choosing a model"
modified:
categories: pytorch
excerpt: Choose small model for transfer learning

date: 2023-01-02
share: true
ads: true
---

Choosing a backbone for transfer learning is easy thanks to Timm. However, which model is a mystery. I use the following to use a small model which has >80\% accuracy on imagenet and has less than 50M parameters.
```python
import pandas as pd

URL = "https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/results/results-imagenet.csv"
INFER_URL = "https://raw.githubusercontent.com/rwightman/pytorch-image-models/main/results/benchmark-infer-amp-nchw-pt112-cu113-rtx3090.csv"

df = pd.read_csv(URL)
df2 = pd.read_csv(INFER_URL)
# df["param_count"] = df["param_count"].astype(str)
# df2["param_count"] = df2["param_count"].astype(str)
df["model_base"] = df["model"].map(lambda x: x.split(".")[0])
df2.rename(columns={"model": "model_base"}, inplace=True)

df = df.join(df2.set_index("model_base").drop("param_count", axis=1), on="model_base")
df["param_count"] = df["param_count"].str.replace(",","")
df["param_count"] = df["param_count"].astype(float)

top1_cond = df["top1"] > 80
param_count_cond = df["param_count"] < 50
df[param_count_cond & top1_cond].sort_values("top1", ascending=False)
```

This is the same result but ordered by increasing number of parameters.
```python
df[param_count_cond & top1_cond].sort_values("param_count").iloc[:10]
```
