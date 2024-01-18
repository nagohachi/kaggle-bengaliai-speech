#!/usr/bin/env python
# coding: utf-8

# In[2]:


from os import path
import os
from pydub import AudioSegment
import pandas as pd


# In[3]:


from pathlib import Path

ROOT_DIR = Path.cwd().parent
INPUT_DIR = ROOT_DIR / 'input'
DATA_DIR = INPUT_DIR / "bengaliai-speech"
TRAIN_MP3_DIR = DATA_DIR / "train_mp3s"
TRAIN_WAV_DIR = INPUT_DIR / "train_wavs"

if os.path.exists(TRAIN_WAV_DIR) == False:
    os.mkdir(TRAIN_WAV_DIR)


# In[4]:


train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df.head()


# In[5]:


# TRAIN_WAV = DATA_DIR / "train_wavs" の中にあるファイルの個数
import os
print(len(os.listdir(TRAIN_WAV_DIR)))


# In[6]:


# test_df の、"id" 列の .mp3 を削除
# test_df = pd.read_csv(DATA_DIR / "test.csv")
# test_df["id"] = test_df["id"].str.replace(".mp3", "")


# In[7]:


train_ids = train_df['id'].tolist()
len(train_ids)


# In[8]:


from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count
cpu_count()


# In[9]:


def process(train_id):
    src_path = TRAIN_MP3_DIR / f'{train_id}.mp3'
    dst_path = TRAIN_WAV_DIR / f'{train_id}.wav'
    sound = AudioSegment.from_mp3(src_path)
    # もし 4 秒以上あるようなら 4 秒までを wav に変換
    if len(sound) >= 4000:
        sound = sound[:4000]
    sound.export(dst_path, format="wav")


# In[10]:


# from tqdm import tqdm
# _ = Parallel(n_jobs=cpu_count())(
#     delayed(process)(train_id)
#     for train_id in tqdm(train_ids)
# )


# In[21]:


# TRAIN_MP3_DIR の中にあるファイルを一つ取り、os.path.getsize() でサイズを取得して print
import os
max_size = 0

for train_id in train_ids:
    src_path = TRAIN_MP3_DIR / f'{train_id}'
    size = os.path.getsize(src_path)
    if size > max_size:
        max_size = size

print(max_size)


# In[12]:


import os
example_file_size = os.path.getsize(DATA_DIR / "examples/Slang Profanity.mp3")
print(example_file_size)


# In[13]:


import os
train_ids = sorted(os.listdir(TRAIN_MP3_DIR))
train_wav_ids = sorted(os.listdir(TRAIN_WAV_DIR))

train_ids_without_extension = [train_id.split('.')[0] for train_id in train_ids]
train_wav_ids_without_extension = [train_id.split('.')[0] for train_id in train_wav_ids]


# In[14]:


# train_wav_ids_without_extension_set = set(train_wav_ids_without_extension)

# for train_id in train_ids_without_extension:
#     if train_id not in train_wav_ids_without_extension_set:
#         process(train_id)


# In[15]:


# TRAIN_WAV = DATA_DIR / "train_wavs" の中にあるファイルの個数
import os
print(len(os.listdir(TRAIN_WAV_DIR)))


# In[16]:


# TRAIN_WAV の中に、0バイトのファイルがないか確かめる
import os
from pathlib import Path

ROOT_DIR = Path.cwd().parent
INPUT_DIR = ROOT_DIR / 'input'
DATA_DIR = INPUT_DIR / "bengaliai-speech"
TRAIN_WAV_DIR = INPUT_DIR / "train_wavs"

train_wav_ids = sorted(os.listdir(TRAIN_WAV_DIR))
train_wav_ids_without_extension = [train_id.split('.')[0] for train_id in train_wav_ids]

# train_wav_ids_without_extension の中にあるファイルのサイズの平均を算出
total_size = 0
for train_wav_id in train_wav_ids_without_extension:
    file_path = TRAIN_WAV_DIR / f"{train_wav_id}.wav"
    total_size += os.path.getsize(file_path)

average_size = total_size / len(train_wav_ids_without_extension)
print(f"The average size of files in TRAIN_WAV_DIR is {average_size} bytes.")

# 空のファイルがあればそのファイル名を print
nearly_empty_files = []
for train_wav_id in train_wav_ids_without_extension:
    file_path = TRAIN_WAV_DIR / f"{train_wav_id}.wav"
    if os.path.getsize(file_path) <= 100:
        print(f"{train_wav_id}.wav is empty.")
        nearly_empty_files.append(train_wav_id.split('.')[0])


# In[ ]:


# pydub を用いて、nearly_empty_files の最初 3 個のファイルを再生
from pydub import AudioSegment
from pydub.playback import play

for train_wav_id in nearly_empty_files[:3]:
    file_path = TRAIN_WAV_DIR / f"{train_wav_id}.wav"
    sound = AudioSegment.from_wav(file_path)
    play(sound)


# In[ ]:


# pydub を用いて、nearly_empty_fies の最初 3 個と同じ id の .mp3 ファイルを再生
for train_wav_id in nearly_empty_files[:3]:
    print(train_wav_id)
    file_path = TRAIN_MP3_DIR / f"{train_wav_id}.mp3"
    sound = AudioSegment.from_mp3(file_path)
    play(sound)


# In[ ]:


# pydub を用いて、nearly_empty_fies の最初 3 個と同じ id の .mp3 ファイルを再生
for train_id in train_ids[:3]:
    print(train_id)
    file_path = TRAIN_MP3_DIR / f"{train_id}"
    sound = AudioSegment.from_mp3(file_path)
    play(sound)

