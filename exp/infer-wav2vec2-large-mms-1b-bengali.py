#!/usr/bin/env python
# coding: utf-8

# ### install

# In[ ]:


# !cp -r ../input/python-packages2 ./

# !tar xvfz ./python-packages2/jiwer.tgz
# !pip install ./jiwer/jiwer-2.3.0-py3-none-any.whl -f ./ --no-index
# !tar xvfz ./python-packages2/normalizer.tgz
# !pip install ./normalizer/bnunicodenormalizer-0.0.24.tar.gz -f ./ --no-index
# !tar xvfz ./python-packages2/pyctcdecode.tgz
# !pip install ./pyctcdecode/attrs-22.1.0-py2.py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./pyctcdecode/exceptiongroup-1.0.0rc9-py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./pyctcdecode/hypothesis-6.54.4-py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./pyctcdecode/numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl -f ./ --no-index --no-deps
# !pip install ./pyctcdecode/pygtrie-2.5.0.tar.gz -f ./ --no-index --no-deps
# !pip install ./pyctcdecode/sortedcontainers-2.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./pyctcdecode/pyctcdecode-0.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps

# !tar xvfz ./python-packages2/pypikenlm.tgz
# !pip install ./pypikenlm/pypi-kenlm-0.1.20220713.tar.gz -f ./ --no-index --no-deps


# In[ ]:


# rm -r python-packages2 jiwer normalizer pyctcdecode pypikenlm


# ### import

# In[ ]:


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


# ### paths

# In[ ]:


ROOT = Path.cwd().parent
print(ROOT)
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"

SAMPLING_RATE = 16_000
MODEL_PATH = INPUT / "wav2vec2-large-mms-1b-bengali-45000-3fold/"
LM_PATH = INPUT / "bengali-sr-download-public-trained-models/wav2vec2-xls-r-300m-bengali/language_model/"


# ### load model, processor, decoder

# In[ ]:


model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

vocab_dict = processor.tokenizer.get_vocab()
vocab_dict = vocab_dict["ben"]
vocab_dict["<s>"] = 64
vocab_dict["</s>"] = 65
sorted_vocab_dict = {
    k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
}

decoder = pyctcdecode.build_ctcdecoder(
    list(sorted_vocab_dict.keys()),
    str(LM_PATH / "5gram.bin"),
)


# ### constants

# In[ ]:


SAMPLING_RATE = 16000


# ### dataloader

# In[ ]:


class BengaliSRTestDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths: list[str], sampling_rate: int):
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate

    def __len__(
        self,
    ):
        return len(self.audio_paths)

    def __getitem__(self, index: int):
        audio_path = self.audio_paths[index]
        sr = self.sampling_rate
        w = librosa.load(audio_path, sr=sr, mono=False)[0]

        return w


# In[ ]:


test = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
test_audio_paths = [str(TEST / f"{aid}.mp3") for aid in test["id"].values]

test_dataset = BengaliSRTestDataset(
    test_audio_paths, SAMPLING_RATE
)

collate_func = partial(
    processor.feature_extractor,
    return_tensors="pt", sampling_rate=SAMPLING_RATE,
    padding=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=8, shuffle=False,
    num_workers=2, collate_fn=collate_func, drop_last=False,
    pin_memory=True,
)


# ### inference

# In[ ]:


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print(device)

model = model.to(device)
model = model.eval()
model = model.half()


# In[ ]:


pred_sentence_list = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        x = batch["input_values"]
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(True):
            y = model(x).logits
        y = y.detach().cpu().numpy()
        
        for l in y:
            beam = decoder.decode_beams(l, beam_width=512)
            s = beam[0][0]
            pred_sentence_list.append(s)


# ### postprocess

# In[ ]:


bnorm = Normalizer()

def postprocess(sentence):
    period_set = set([".", "?", "!", "।"])
    _words = [bnorm(word)["normalized"] for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    try:
        if sentence[-1] not in period_set:
            sentence += "।"
    except:
        # print(sentence)
        sentence = "।"
    return sentence


# In[ ]:


pp_pred_sentence_list = [
    postprocess(s) for s in tqdm(pred_sentence_list)]


# ### make submission

# In[ ]:


test["sentence"] = pp_pred_sentence_list
test.to_csv("submission.csv", index=False)
print(test.head())

