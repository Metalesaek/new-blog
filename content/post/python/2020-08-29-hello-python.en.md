---
title: hello python
author: Metalesaek
date: '2020-08-29'
slug: hello-python
categories: []
tags:
  - python
subtitle: ''
summary: ''
output:
  blogdown::html_page:
    toc: true
    fig_width: 6
    dev: "svg"
authors: []
lastmod: '2020-08-29T16:37:41+02:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

```python
# implementing PCA method and incremental pca in python
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA
iris=load_iris()
x = iris.data
y = iris.target
```


```python
ipca = IncrementalPCA(n_components=2, batch_size=10)
mod1 = ipca.fit_transform(x)
pca = PCA(n_components=2)
mod2 = pca.fit_transform(x)
```


```python
colors = ["red", "green", "blue"]
for x_transformed, title in [(mod1, "incrementalpca"),(mod2, "pca")]:
    plt.figure(figsize = (8,8))
    for color, i, target_name in zip(colors, [0,1,2], iris.target_names):
        plt.scatter(x_transformed[y==i,0], x_transformed[y==i,1],
                   color=color, lw = 2, label = target_name)
    if "incremental" in title:
        err =  np.abs(np.abs(mod1)-np.abs(mod2)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error")
    else:
        plt.title(title + " of iris dataset")
        plt.legend(loc="best", shadow = False, scatterpoints = 1)
        

plt.show()
        
```


![png](/img/python_img/pcanetbook_files/pcanetbook_3_0.png)



![png](/img/python_img/pcanetbook_files/pcanetbook_3_1.png)



```python
np.abs(np.abs(mod1)-np.abs(mod2)).mean()
```


```python

```
