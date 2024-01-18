#!/usr/bin/env python
# coding: utf-8

# In[6]:


import typing as tp
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
import matplotlib

import pandas as pd
import pyctcdecode
import numpy as np
from tqdm.notebook import tqdm

import librosa

import pyctcdecode
import kenlm
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from bnunicodenormalizer import Normalizer

import cloudpickle as cpkl


# In[7]:


ROOT = Path.cwd().parent
print(ROOT)
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TRAIN_WAV = INPUT / "train_wavs"
TEST = DATA / "test_mp3s"
NISQA = Path.cwd() / "nisqa"

SAMPLING_RATE = 16_000


# In[16]:


# TRAIN_WAV の中にあるファイルを100個取り出し、../input/train_wavs_100 に保存する
import os
import shutil
import random

os.mkdir(INPUT / "train_wavs_100")

files = os.listdir(TRAIN_WAV)

for i in range(1000):
    file = files[i]
    shutil.copyfile(TRAIN_WAV / file, INPUT / "train_wavs_100" / file)


# In[6]:


import os, shutil
from pathlib import Path
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"

files = os.listdir(TRAIN)

os.mkdir(INPUT / "train_mp3s_sample_9000")

for i in range(9000):
    file = files[i]
    shutil.copyfile(TRAIN / file, INPUT / "train_mp3s_sample_9000" / file)


# In[8]:


# 1000個のファイルを読み込む
train_mp3s_sample_9000_ids = os.listdir(INPUT / "train_mp3s_sample_9000")
len(train_mp3s_sample_9000_ids)


# In[10]:


get_ipython().run_line_magic('run', './NISQA/run_predict.py --mode predict_dir --pretrained_model ./NISQA/weights/nisqa.tar --data_dir ../input/train_wavs_1000 --num_workers 4 --bs 20 --output_dir ./results')


# In[12]:


get_ipython().run_line_magic('run', './NISQA/run_predict.py --mode predict_file --pretrained_model ./NISQA/weights/nisqa.tar --deg ../input/train_wavs/000005f3362c.wav --output_dir ./results')


# In[ ]:


test = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})


# In[ ]:


# train_mp3s の中にあるファイルを 8000 個取り出し、../input/train_mp3s_sample_8000 に保存する
import os
import shutil
import random

os.mkdir(INPUT / "train_mp3s_sample_8000")

files = os.listdir(TRAIN)

for i in range(8000):
    file = files[i]
    shutil.copyfile(TRAIN / file, INPUT / "train_mp3s_sample_8000" / file)

