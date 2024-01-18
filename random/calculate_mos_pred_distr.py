#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
from pathlib import Path
import pandas as pd


# In[14]:


# paths
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
INSPECT = INPUT / "inspect"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"
MACRO_NORMALIZATION = INPUT / "macro-normalization"


# In[15]:


train_normalized_with_noise_info = pd.read_csv(
    DATA / "train_normalized_with_noise_info.csv",
    dtype={
        "id": str,
        "mos_pred": float,
        "noi_pred": float,
        "dis_pred": float,
        "col_pred": float,
        "loud_pred": float,
        "model": str,
    },
)


# In[16]:


# train_normalized における mos_pred, noi_pred, dis_pred の分布を重ねて表示
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(train_normalized_with_noise_info["mos_pred"], label="mos_pred")
sns.histplot(train_normalized_with_noise_info["noi_pred"], label="noi_pred")
sns.histplot(train_normalized_with_noise_info["dis_pred"], label="dis_pred")
ax.legend()

