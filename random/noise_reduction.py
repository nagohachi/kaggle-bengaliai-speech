#!/usr/bin/env python
# coding: utf-8

# ## Reference
# 
# Firstly, Please upvote/refer to [@tawara's](https://www.kaggle.com/ttahara) discussions and inference [notebook](https://www.kaggle.com/code/ttahara/bengali-sr-public-wav2vec2-0-w-lm-baseline).
# 
# 

# In[99]:


ON_KAGGLE = False


# ## Import

# In[100]:


if ON_KAGGLE:
    import os
    os.system("!cp -r ../input/python-packagess2 ./")
    os.system("!tar xvfz ./python-packagess2/jiwer.tgz")
    os.system("!pip install ./jiwer/jiwer-2.3.0-py3-none-any.whl -f ./ --no-index")
    os.system("!tar xvfz ./python-packagess2/normalizer.tgz")
    os.system("!pip install ./normalizer/bnunicodenormalizer-0.0.24.tar.gz -f ./ --no-index")
    os.system("!tar xvfz ./python-packagess2/pyctcdecode.tgz")
    os.system("!pip install ./pyctcdecode/attrs-22.1.0-py2.py3-none-any.whl -f ./ --no-index --no-deps")
    os.system("!pip install ./pyctcdecode/exceptiongroup-1.0.0rc9-py3-none-any.whl -f ./ --no-index --no-deps")
    os.system("!pip install ./pyctcdecode/hypothesis-6.54.4-py3-none-any.whl -f ./ --no-index --no-deps")
    os.system("!pip install ./pyctcdecode/numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl -f ./ --no-index --no-deps")
    os.system("!pip install ./pyctcdecode/pygtrie-2.5.0.tar.gz -f ./ --no-index --no-deps")
    os.system("!pip install ./pyctcdecode/sortedcontainers-2.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps")
    os.system("!pip install ./pyctcdecode/pyctcdecode-0.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps")
    os.system("!tar xvfz ./python-packagess2/pypikenlm.tgz")
    os.system("!pip install ./pypikenlm/pypi-kenlm-0.1.20220713.tar.gz -f ./ --no-index --no-deps]")
    os.system("rm -r python-packagess2 jiwer normalizer pyctcdecode pypikenlm")


# In[101]:


import typing as tp
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field

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


# In[102]:


ROOT = Path.cwd().parent
print(ROOT)
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"

SAMPLING_RATE = 16_000
MODEL_PATH = INPUT / "bengali-wav2vec2-finetuned/"
LM_PATH = INPUT / "bengali-sr-download-public-trained-models/wav2vec2-xls-r-300m-bengali/language_model/"


# ### load model, processor, decoder

# In[103]:


model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)


# In[104]:


vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoder = pyctcdecode.build_ctcdecoder(
    list(sorted_vocab_dict.keys()),
    str(LM_PATH / "5gram.bin"),
)


# In[105]:


processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)


# ## prepare dataloader

# In[106]:


class BengaliSRTestDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        audio_paths: list[str],
        sampling_rate: int
    ):
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        
    def __len__(self,):
        return len(self.audio_paths)
    
    def __getitem__(self, index: int):
        audio_path = self.audio_paths[index]
        sr = self.sampling_rate
        # audio_path にある .mp3 ファイルを、PCEN を用いて前処理
        y1, s3 = librosa.load(audio_path, sr=sr, mono=False)
        S1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=128)
        D1 = librosa.power_to_db(S1, ref=np.max)
        Dp1 = librosa.pcen(S1 * (2**31), sr=sr, hop_length=512, gain=1.1, bias=2, power=0.25, time_constant=0.8, eps=1e-06, max_size=2)

        return Dp1

    def __getitem__(self, index: int):
        audio_path = self.audio_paths[index]
        sr = self.sampling_rate
        w = librosa.load(audio_path, sr=sr, mono=False)[0]
        # 例: メルスペクトログラムの計算
        S = librosa.feature.melspectrogram(y=w, sr=sr, n_mels=128)
        print(S.shape)
        return S


# In[107]:


test = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
print(test.head())

test_audio_paths = [str(TEST / f"{aid}.mp3") for aid in test["id"].values]

test_dataset = BengaliSRTestDataset(
    test_audio_paths, SAMPLING_RATE
)

collate_func = partial(
    processor_with_lm.feature_extractor,
    return_tensors="pt", sampling_rate=SAMPLING_RATE,
    padding=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=8, shuffle=False,
    num_workers=2, collate_fn=collate_func, drop_last=False,
    pin_memory=True,
)


# In[108]:


train = pd.read_csv(DATA / "train.csv", dtype={"id": str}).drop(["split"], axis=1)
# train からランダムに 100 個選ぶ
train_random_100 = train.sample(100, random_state=42)
train_audio_paths_random_100 = [str(TRAIN / f"{aid}.mp3") for aid in train_random_100["id"].values]

train_dataset_random_100 = BengaliSRTestDataset(
    train_audio_paths_random_100, SAMPLING_RATE
)

train_loader_random_100 = torch.utils.data.DataLoader(
    train_dataset_random_100, batch_size=8, shuffle=False,
    num_workers=2, collate_fn=collate_func, drop_last=False,
    pin_memory=True,
)


# ## Inference

# In[109]:


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print(device)


# In[110]:


model = model.to(device)
model = model.eval()
# model = model.half()


# In[111]:


pred_sentence_list = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(train_loader_random_100)):
        x = batch["input_values"]
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(True):
            y = model(x).logits
        y = y.detach().cpu().numpy()
        
        for l in y:  
            sentence = processor_with_lm.decode(l, beam_width=512).text
            pred_sentence_list.append(sentence)


# In[ ]:


print(len(pred_sentence_list))


# ## Make Submission

# In[ ]:


bnorm = Normalizer()

def postprocess(sentence):
    period_set = set([".", "?", "!", "।"])
    _words = [bnorm(word)['normalized']  for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    try:
        if sentence[-1] not in period_set:
            sentence+="।"
    except:
        # print(sentence)
        sentence = "।"
    return sentence


# In[ ]:


pp_pred_sentence_list = [
    postprocess(s) for s in tqdm(pred_sentence_list)
]


# In[ ]:


train_compare = train_random_100.copy()
train_compare["sentence2"] = pp_pred_sentence_list

print(train_compare.head())


# In[ ]:


import jiwer


# In[ ]:


def mean_wer(solution, submission):
    sum_wer = 0
    for s, t in zip(solution, submission):
        sum_wer += jiwer.wer(s, t)
    return sum_wer / len(solution)


# In[ ]:


mean_wer(train_compare["sentence"], train_compare["sentence2"])


# ## EOF
