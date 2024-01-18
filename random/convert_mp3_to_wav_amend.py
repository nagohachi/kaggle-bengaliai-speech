#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pydub import AudioSegment
import pandas as pd


# In[2]:


from pathlib import Path

ROOT_DIR = Path.cwd().parent
INPUT_DIR = ROOT_DIR / 'input'
DATA_DIR = INPUT_DIR / "bengaliai-speech"
TRAIN_MP3_DIR = DATA_DIR / "train_mp3s"
TRAIN_WAV_DIR = INPUT_DIR / "train_wavs"


# In[3]:


train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df.head()


# In[4]:


train_ids = train_df['id'].tolist()


# In[18]:


# train_wav_ids は、TRAIN_WAV_DIR の中にあるファイル名のリストを再帰的に取得する
import glob
train_wav_ids = os.listdir(TRAIN_WAV_DIR)

print(len(train_wav_ids))
print(train_wav_ids[:5])


# In[15]:


from pydub import AudioSegment
from pydub.playback import play

def playSound(fileName):
    format = os.path.splitext(fileName)[1][1:]
    if format == "mp3":
        sound = AudioSegment.from_mp3(TRAIN_MP3_DIR / fileName)
    else:
        prefix = fileName[:3]
        sound = AudioSegment.from_file(TRAIN_WAV_DIR / prefix / fileName, format=format)
    play(sound)


# In[16]:


playSound("000024b3d810.mp3")


# In[17]:


playSound("000024b3d810.wav")


# In[ ]:




