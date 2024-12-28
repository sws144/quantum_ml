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
from quantumtools import metrics

# %%
import numpy as np

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
y_true = [-0.5, 0.1, 0.20, 0.31, 0.6]
y_pred = [-0.45, 0.05, 0.10, 0.3, 0.5]
sample_weight = None

# %% [markdown]
# ## Test 2

# %%
from quantumtools import metrics

# %%
dir(metrics)

# %%
metrics.ginierror(y_true, y_pred ,sample_weight )

# %% [markdown]
# ## Negative Values Test

# %%
# there is existing issue if negative values

# %%

# %%

# %%

# %%

# %%
