#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
from pathlib import Path
import pandas as pd


# In[52]:


# paths
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
INSPECT = INPUT / "inspect"
TRAIN = DATA / "train_mp3s"
TRAIN_WAV = DATA / "train_wavs"
TRAIN_WAV_NOISE_REDUCED = DATA / "train_wavs_noise_reduced"
TEST = DATA / "test_mp3s"
MACRO_NORMALIZATION = INPUT / "macro-normalization"


# In[53]:


train = pd.read_csv(DATA / "train.csv")
normalized = pd.read_csv(MACRO_NORMALIZATION / "normalized.csv")
inspect = pd.read_csv(INSPECT / "NISQA_wavfiles.csv")


# In[54]:


print(train.head())
print(normalized.head())


# In[55]:


# normalized の id, normalized の列のみ抜き出し
normalized = normalized.loc[:, ["id", "normalized"]]


# In[56]:


# train と normalized["normalized"] を id で結合
train = pd.merge(train, normalized, on="id", how="left")


# In[57]:


# train の normalized を sentence_normalized に変更
train = train.rename(columns={"normalized": "sentence_normalized"})


# In[58]:


# inspect の列 "deg_mp3" を "id" に変更
inspect.rename(columns={"deg_mp3": "id"}, inplace=True)
# inspect の列 "id" の末尾にある .mp3 を削除
inspect["id"] = inspect["id"].str.replace(".mp3", "")
# inspect から 列 "deg" を削除
inspect.drop("deg", axis=1, inplace=True)


# In[59]:


inspect.head()


# In[60]:


# train と inspect を id で結合
train = pd.merge(train, inspect, on="id", how="left")


# In[61]:


train.head()


# In[ ]:


# train を上書き
train.to_csv(DATA / "train_normalized_with_noise_info.csv", index=False)

