#!/usr/bin/env python
# coding: utf-8


from os import path
import os
from pydub import AudioSegment
import pandas as pd


from pathlib import Path

ROOT_DIR = Path.cwd().parent
INPUT_DIR = ROOT_DIR / "input"
DATA_DIR = INPUT_DIR / "bengaliai-speech"
TRAIN_MP3_DIR = DATA_DIR / "train_mp3s"
TRAIN_WAV_DIR = INPUT_DIR / "train_wavs"

if os.path.exists(TRAIN_WAV_DIR) == False:
    os.mkdir(TRAIN_WAV_DIR)


train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df.head()

train_ids = sorted(train_df["id"].tolist())

from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count


def process(train_id):
    train_id_prefix = train_id[:3]
    src_path = TRAIN_MP3_DIR / f"{train_id}.mp3"
    dst_path = TRAIN_WAV_DIR / train_id_prefix / f"{train_id}.wav"
    if os.path.exists(TRAIN_WAV_DIR / train_id_prefix) == False:
        os.makedirs(TRAIN_WAV_DIR / train_id_prefix, exist_ok=True)
    sound = AudioSegment.from_mp3(src_path)
    # もし dst_path が存在していたら何もしない
    if os.path.exists(dst_path):
        print(f"{dst_path} exists")
    else:
        try:
            sound.export(dst_path, format="wav")
        except:
            print(f"{train_id} failed")


from tqdm import tqdm

_ = Parallel(n_jobs=cpu_count() * 4 // 5)(
    delayed(process)(train_id) for train_id in tqdm(train_ids)
)

# import os
# train_ids = sorted(os.listdir(TRAIN_MP3_DIR))
# train_wav_ids = sorted(os.listdir(TRAIN_WAV_DIR))

# train_ids_without_extension = [train_id.split('.')[0] for train_id in train_ids]
# train_wav_ids_without_extension = [train_id.split('.')[0] for train_id in train_wav_ids]


# train_wav_ids_without_extension_set = set(train_wav_ids_without_extension)

# for train_id in train_ids_without_extension:
#     if train_id not in train_wav_ids_without_extension_set:
#         process(train_id)


# TRAIN_WAV = DATA_DIR / "train_wavs" の中にあるファイルの個数
import os

print(len(os.listdir(TRAIN_WAV_DIR)))


# TRAIN_WAV_DIR の中のファイル名を取得 (フルパスで)
import glob

train_wav_files = glob.glob(str(TRAIN_WAV_DIR / "*.wav"))
print(len(train_wav_files))
print(train_wav_files[0])


# TRAIN_WAV_DIR の中にあるファイルを1つ再生 (再生には pydub を使用)
# from pydub import AudioSegment
# from pydub.playback import play

# sound = AudioSegment.from_file(train_wav_files[0], format="wav")
# play(sound)
