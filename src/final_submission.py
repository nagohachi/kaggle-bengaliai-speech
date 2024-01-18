#!/usr/bin/env python
# coding: utf-8

# This notebook is based on : https://www.kaggle.com/code/ttahara/bengali-sr-umong-sain-s-wav2vec2-0-w-lm-baseline .
# 
# Make sure to execute files below before running this notebook:
# 1. train/finetune_fole_without_unigrams.py
# 2. train/finetune_with_commonvoice.py

# ## Install packages 
# This part is needed only on Kaggle Notebook. packages can be obtained from [here](https://www.kaggle.com/datasets/nagohachi/bengaliai-packages/)

# In[3]:


# !cp -r ../input/bengaliai-packages ./

# !pip install ./bengaliai-packages/setuptools-65.7.0-py3-none-any.whl -f ./ --no-index
# !pip install ./bengaliai-packages/jiwer-3.0.3-py3-none-any.whl -f ./ --no-index
# !pip install ./bengaliai-packages/bnunicodenormalizer-0.1.6/bnunicodenormalizer-0.1.6 -f ./ --no-index
# !pip install ./bengaliai-packages/attrs-23.1.0-py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./bengaliai-packages/exceptiongroup-1.1.3-py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./bengaliai-packages/hypothesis-6.87.0-py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./bengaliai-packages/pygtrie-2.5.0-py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./bengaliai-packages/sortedcontainers-2.4.0-py2.py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./bengaliai-packages/pyctcdecode-0.5.0-py2.py3-none-any.whl -f ./ --no-index --no-deps
# !pip install ./bengaliai-packages/pypi-kenlm-0.1.20220713/pypi-kenlm-0.1.20220713 -f ./ --no-index --no-deps

# !rm -rf ./bengaliai-packages


# In[5]:


from pathlib import Path
from functools import partial

import pandas as pd
import pyctcdecode
from tqdm.notebook import tqdm

import librosa

import pyctcdecode
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from bnunicodenormalizer import Normalizer


# In[6]:


ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"

SAMPLING_RATE = 16_000
MODEL_PATH = INPUT / "wav2vec2-small-finetuned-with-commonvoice/" # finetuned again with commonvoice
LM_PATH = INPUT / "bengali-sr-download-public-trained-models/wav2vec2-xls-r-300m-bengali/language_model/"


# ### load model, processor, decoder

# In[7]:


model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)


# In[8]:


vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoder = pyctcdecode.build_ctcdecoder(
    list(sorted_vocab_dict.keys()),
    str(LM_PATH / "5gram.bin"),
)


# In[9]:


processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)


# ## prepare dataloader

# In[10]:


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
        w = librosa.load(audio_path, sr=sr, mono=False)[0]
        
        return w


# In[11]:


test = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
print(test.head())


# In[12]:


test_audio_paths = [str(TEST / f"{aid}.mp3") for aid in test["id"].values]


# In[13]:


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


# ## Inference

# In[14]:


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print(device)


# In[15]:


model = model.to(device)
model = model.eval()
model = model.half()


# In[16]:


pred_sentence_list = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        x = batch["input_values"]
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(True):
            y = model(x).logits
        y = y.detach().cpu().numpy()
        
#         {'alpha': 0.345, 'beta': 0.06, 'beam_width': 768}
        for l in y:  
            sentence = processor_with_lm.decode(
                l, beam_width=2000, 
                alpha=0.345, 
                beta=0.06
            ).text
            pred_sentence_list.append(sentence)


# ## Make Submission

# In[17]:


bnorm = Normalizer()

def postprocess(sentence):
    period_set = set([".", "?", "!", "।"])
    _words = [bnorm(word)['normalized']  for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    try:
        if sentence[-1] not in period_set:
            sentence+="।"
    except:
        sentence = "।"
    return sentence


# In[18]:


pp_pred_sentence_list = [
    postprocess(s) for s in tqdm(pred_sentence_list)]


# In[19]:


test["sentence"] = pp_pred_sentence_list

test.to_csv("submission.csv", index=False)

pd.set_option("display.max_colwidth", 100)
print(test.head())


# ## EOF
