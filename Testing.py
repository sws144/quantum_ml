# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: quantum_ml
#     language: python
#     name: quantum_ml
# ---

# %% [markdown]
# # Testing

# %%
from quantum_ml import metrics

# %%
import numpy as np

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
y_true = [100, 2, 3, 4, 1]
y_pred = [100, 2, 3, 4, 1]
sample_weight = None

# %% [markdown]
# ## Test 2

# %%
from quantum_ml import metrics

# %%
dir(metrics)

# %%
metrics.ginierror(y_true, y_pred ,sample_weight )

# %%
