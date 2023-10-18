#!/usr/bin/env python
# coding: utf-8

# #### Reference
# 
# Firstly, Please upvote/refer to [@snnclsr](https://www.kaggle.com/snnclsr) discussions and inference [notebook]https://www.kaggle.com/code/snnclsr/0-444-optimize-decoding-parameters-with-optuna).
# 
# 
# Third, please upvote this one :)

# ## What this notebook features??
# 
# - I wanted to showcase the impact of finetuning the models on competition dataset.
# - Current version comprises of finetuned model only with 10% of competition training data.
# - I will publish the training code in upcoming days. You can refer to this [dataset]()
# 
# Public models from hugging faces:
# * `https://huggingface.co/ai4bharat/indicwav2vec_v1_bengali` for Wav2vec2CTC Model only
# * `https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali` for Language Model
# 
# I didn't trained these models using the competitaion data at all. I just want to know public models score as baseline.  
# 
# So we may get higher and higher score by fine-tuning on competition data.
# 
# **Note: I only finetuned the indicwav2vec_v1_bengali which is a CTC model. I am still using the public LM model mentioned above.**
# 
# 
# ### Everything above PLUS
# 
# - How to find the best decoding params using optuna on valid dataset.
# 
# It's being suggested by the authors of the [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode/tree/main) developers that we should perform a parameter search because it can improve our results on a specific tasks other than English such as ours.
# 
# > (Note: pyctcdecode contains several free hyperparameters that can strongly influence error rate and wall time. Default values for these parameters were (merely) chosen in order to yield good performance for one particular use case. For best results, especially when working with languages other than English, users are encouraged to perform a hyperparameter optimization study on their own data.)
# 
# So we will give it a try to find the best parameters in the validation split of the train dataset (because of the time constraints we will only use 5k). Here are the list of decoding params for easy access:
# 
# ```python
# # from: https://github.com/kensho-technologies/pyctcdecode/blob/main/pyctcdecode/constants.py
# # default parameters for decoding (can be modified)
# DEFAULT_ALPHA = 0.515
# DEFAULT_BETA = 1.665
# DEFAULT_UNK_LOGP_OFFSET = -10.0
# DEFAULT_BEAM_WIDTH = 100
# DEFAULT_HOTWORD_WEIGHT = 10.0
# DEFAULT_PRUNE_LOGP = -10.0
# DEFAULT_PRUNE_BEAMS = False
# DEFAULT_MIN_TOKEN_LOGP = -5.0
# DEFAULT_SCORE_LM_BOUNDARY = True
# 
# # other constants for decoding
# AVG_TOKEN_LEN = 6  # average number of characters expected per token (used for UNK scoring)
# MIN_TOKEN_CLIP_P = 1e-15  # clipping to avoid underflow in case of malformed logit input
# LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)  # kenlm returns base10 but we like natural
# ```

# ## Import

# In[1]:


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


# In[2]:


# !pip install ../input/jiwer-3-0-3/jiwer-3.0.3-py3-none-any.whl


# In[3]:


# rm -r python-packages2 jiwer normalizer pyctcdecode pypikenlm


# In[4]:


from pathlib import Path
from functools import partial

import pandas as pd
import pyctcdecode
from tqdm import tqdm

import librosa

import pyctcdecode
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from bnunicodenormalizer import Normalizer



# In[5]:


FIND_PARAMS = True

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"

SAMPLING_RATE = 16_000
MODEL_PATH = INPUT / INPUT / "saved_model-finetune-with-commonvoice-without-unigram/ensemble/"
LM_PATH = INPUT / "arijitx-full-model/wav2vec2-xls-r-300m-bengali/language_model/"


# ### load model, processor, decoder

# In[6]:


model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)


# In[7]:


vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoder = pyctcdecode.build_ctcdecoder(
    list(sorted_vocab_dict.keys()),
    str(LM_PATH / "5gram.bin"),
)


# In[8]:


processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)


# ## prepare dataloader

# In[9]:


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


# In[10]:


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

model = model.to(device)
model = model.eval()
model = model.half()


# # Finding the best decoding params

# In[11]:


import jiwer

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


def score(gts, preds):
    return jiwer.wer(gts, preds)


def inference(m, data_loader):
    logits = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch["input_values"]
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(True):
                y = model(x).logits
            y = y.detach().cpu().numpy()
            logits.extend(y)
    return logits


def decode(logits, params={"beam_width": 2000}, pp=True):    
    pred_sentence_list = [processor_with_lm.decode(sentence, **params).text for sentence in tqdm(logits)]
    if pp:
        pred_sentence_list = [postprocess(s) for s in pred_sentence_list]
    return pred_sentence_list


# In[12]:


constants = """
# from: https://github.com/kensho-technologies/pyctcdecode/blob/main/pyctcdecode/constants.py
# default parameters for decoding (can be modified)
DEFAULT_ALPHA = 0.495
DEFAULT_BETA = 1.275
DEFAULT_UNK_LOGP_OFFSET = -10.0
DEFAULT_BEAM_WIDTH = 100
DEFAULT_HOTWORD_WEIGHT = 10.0
DEFAULT_PRUNE_LOGP = -10.0
DEFAULT_PRUNE_BEAMS = False
DEFAULT_MIN_TOKEN_LOGP = -5.0
DEFAULT_SCORE_LM_BOUNDARY = True

# other constants for decoding
AVG_TOKEN_LEN = 6  # average number of characters expected per token (used for UNK scoring)
MIN_TOKEN_CLIP_P = 1e-15  # clipping to avoid underflow in case of malformed logit input
LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)  # kenlm returns base10 but we like natural
"""


# In[13]:


def objective(trial):
    """
    alpha: weight for language model during shallow fusion
    beta: weight for length score adjustment of during scoring
    unk_score_offset: amount of log score offset for unknown tokens
    lm_score_boundary: whether to have kenlm respect boundaries when scoring
    """
    alpha = trial.suggest_float("alpha", 0.0, 2.15)
    beta = trial.suggest_float("beta", 0.0, 2.05)
    beam_width = trial.suggest_categorical("beam_width", [2000,])
    gts = valid["sentence"].values.tolist()
    decode_params = {
        "alpha": alpha,
        "beta": beta,
        "beam_width": beam_width
    }
    preds = decode(logits, params=decode_params, pp=True)
    wer_score = score(gts, preds)
    return wer_score


# In[14]:


# Default decoding configuration in the public notebook.
best_params = {"beam_width": 2000}

if FIND_PARAMS:
    import optuna
    from optuna.trial import TrialState
    
    # valid = pd.read_csv(DATA / "excluded_valid.csv") # dtype={"id": str}
    # valid_audio_paths = [str(TRAIN / f"{aid}.mp3") for aid in valid["id"].values]
    valid = pd.read_csv(DATA / "train.csv") # dtype={"id": str}
    valid = valid[valid["split"] == "valid"]
    valid_audio_paths = [str(TRAIN / f"{aid}.mp3") for aid in valid["id"].values]

    valid_dataset = BengaliSRTestDataset(
        valid_audio_paths, SAMPLING_RATE
    )

    collate_func = partial(
        processor_with_lm.feature_extractor,
        return_tensors="pt", sampling_rate=SAMPLING_RATE,
        padding=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False,
        num_workers=2, collate_fn=collate_func, drop_last=False,
        pin_memory=True,
    )
    # Calculating the base score
    print(constants)
    logits = inference(model, valid_loader)
    base_preds = decode(logits)
    gts = valid["sentence"].values.tolist()
    base_wer_score = score(gts, base_preds)
    print(f"Base wer score: {base_wer_score}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    if study.best_value < base_wer_score:
        print(f"Base score improved to {study.best_value} from {base_wer_score}. Assigning {study.best_params} to best_params")
        best_params = study.best_params


# # Inference with the best params

# In[ ]:


# Please see the Version 3. of this notebook to see the results.
# best_params = {'alpha': 0.345, 'beta': 0.06, 'beam_width': 768}


# In[ ]:


print(f"Running the inference with params: {best_params}")


# In[ ]:


# test = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
# test_audio_paths = [str(TEST / f"{aid}.mp3") for aid in test["id"].values]

# test_dataset = BengaliSRTestDataset(
#     test_audio_paths, SAMPLING_RATE
# )
# collate_func = partial(
#     processor_with_lm.feature_extractor,
#     return_tensors="pt", sampling_rate=SAMPLING_RATE,
#     padding=True,
# )
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=8, shuffle=False,
#     num_workers=2, collate_fn=collate_func, drop_last=False,
#     pin_memory=True,
# )

# pred_sentence_list = []

# with torch.no_grad():
#     for batch in tqdm(test_loader):
#         x = batch["input_values"]
#         x = x.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast(True):
#             y = model(x).logits
#         y = y.detach().cpu().numpy()
        
#         for l in y:  
#             sentence = processor_with_lm.decode(l, **best_params).text
#             pred_sentence_list.append(sentence)


# pp_pred_sentence_list = [postprocess(s) for s in tqdm(pred_sentence_list)]


# ## Make Submission

# In[ ]:


# test["sentence"] = pp_pred_sentence_list
# test.to_csv("submission.csv", index=False)
# print(test.head())


# ## EOF
