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
sample_weight = [1, 1, 1, 1, 10]

# %%
sorted_indices = np.argsort(y_pred)
sorted_indices

# %%
sample_weight = np.asarray(sample_weight)
sample_weight
y_true =  np.asarray(y_true)
y_pred = np.asarray(y_pred)

# %%
sorted_df = pd.DataFrame({
    "y_true":y_true[sorted_indices],
    "y_pred":y_pred[sorted_indices],
    "sample_weight":sample_weight[sorted_indices],
}
)

# %%
sorted_df

# %%
out = pd.qcut(sorted_df["y_pred"], 3,duplicates="drop")
out

# %%
sorted_df['pred_group'] = out

# %%
sorted_df

# %%
grouped_df = sorted_df.groupby('pred_group').mean()[["y_true","y_pred"]].join(sorted_df.groupby('pred_group').sum()[["sample_weight"]])
grouped_df

# %%
grouped_df.plot()

# %%
fig, ax = plt.subplots()
ax2 = ax.twinx()

grouped_df[['y_true','y_pred']].plot(linestyle='-',marker='o',ax=ax)
grouped_df['sample_weight'].plot(kind='bar',ax=ax2, alpha=0.25)

plt.show()

# %%
grouped_df

# %% [markdown]
# ## Test 2

# %%
from quantum_ml import metrics

# %%
dir(metrics)

# %%
metrics.ginierror(y_true, y_pred ,sample_weight )

# %%
