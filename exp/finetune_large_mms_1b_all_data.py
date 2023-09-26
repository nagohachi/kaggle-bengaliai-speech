#!/usr/bin/env python
# coding: utf-8

# - This is a training demo, you can run this code locally, using better GPUs.
# - The inference part is here: [Bengali SR wav2vec_v1_bengali [Inference]](https://www.kaggle.com/takanashihumbert/bengali-sr-wav2vec-v1-bengali-inference), it scores **0.445** on the leaderboard.
# - Feel free to upvote, thanks!


# this part is not needed because the packages are already described in pyproject.toml

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


import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as tat
from datasets import load_dataset, load_metric, Audio
import os

import typing as tp
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyctcdecode
import numpy as np
from tqdm import tqdm

import pyctcdecode
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import warnings

from sklearn.model_selection import KFold

import wandb

wandb.init(
    project="wav2vec2-large-mms-1b-bengali",
    name="nagohachi",
    config={
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "folds": 3,
        "model": "wav2vec2-large-mms-1b-bengali",
    },
)

warnings.filterwarnings("ignore")
torchaudio.set_audio_backend("soundfile")


# hyper-parameters
SR = 16000
torch.backends.cudnn.benchmark = True
num_folds = 3

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"

output_dir = INPUT / "saved_model_large-mms-1b-bengali-fold"
MODEL_PATH = INPUT / "wav2vec2-large-mms-1b-bengali/"
LM_PATH = INPUT / "arijitx-full-model/wav2vec2-xls-r-300m-bengali/language_model"

SENTENCES_PATH = INPUT / "macro-normalization/normalized.csv"
INDEXES_PATH = INPUT / "dataset-overlaps-with-commonvoice-11-bn/indexes.csv"


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
    str(LM_PATH) + "/5gram.bin",
)

# - From @mbmmurad's [Dataset overlaps with CommonVoice 11 bn](https://www.kaggle.com/code/mbmmurad/dataset-overlaps-with-commonvoice-11-bn), The competition dataset might contain the audios of the mozilla-foundation/common_voice_11_0 dataset. Here I just simply exclude them from the validation set.
# - Also, I use @UmongSain's normalized data [here](https://www.kaggle.com/code/umongsain/macro-normalization/notebook). Thanks to him!


class W2v2Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.pathes = df["id"].values
        self.sentences = df["sentence_normalized"].values
        self.resampler = tat.Resample(32000, SR)

    def __getitem__(self, idx):
        apath = TRAIN / f"{self.pathes[idx]}.mp3"
        waveform, sample_rate = torchaudio.load(apath, format="mp3")
        waveform = self.resampler(waveform)
        batch = dict()
        y = processor(waveform.reshape(-1), sampling_rate=SR).input_values[0]
        batch["input_values"] = y
        with processor.as_target_processor():
            batch["labels"] = processor(self.sentences[idx]).input_ids

        return batch

    def __len__(self):
        return len(self.df)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_PATH,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    # gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ctc_zero_infinity=True,
    diversity_loss_weight=100,
)


# you can freeze some params
model.freeze_feature_extractor()


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    group_by_length=False,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    save_strategy="steps",
    # max_steps=80000,  # you can change to "num_train_epochs"
    num_train_epochs=3,
    fp16=True,
    save_steps=5000,
    eval_steps=5000,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=600,
    save_total_limit=1,
    load_best_model_at_end=True,
    # metric_for_best_model="wer",
    # greater_is_better=False,
    prediction_loss_only=False,
    auto_find_batch_size=True,
    report_to="wandb",
)


indexes = set(pd.read_csv(INDEXES_PATH)["id"])
sentences = pd.read_csv(
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

# sentence の中で、mos_pred が NaN または mos_pred が 1.5 以下のものを除外
# sentences = sentences[
#     ~((sentences["mos_pred"].isnull()) | (sentences["mos_pred"] <= 1.5))
# ]

# sentences の中で、mos_pred が 1.5 以上のものを 70 %, それ以外のものを 30 % で構成されるようにする
sentences = pd.concat(
    [
        sentences[sentences["mos_pred"] < 1.5].sample(frac=0.3, random_state=42),
        sentences[sentences["mos_pred"] >= 1.5].sample(frac=0.7, random_state=42),
    ]
).reset_index(drop=True)

# sentences がランダムな順番になるようにシャッフル
sentences = sentences.sample(frac=1, random_state=42).reset_index(drop=True)

sentences = sentences[
    ~((sentences.index.isin(indexes)) & (sentences["split"] == "train"))
].reset_index(drop=True)

print("sentences_size", len(sentences))

sentences_split_train = sentences[sentences["split"] == "train"].reset_index(drop=True)
sentences_split_valid = sentences[sentences["split"] == "valid"].reset_index(drop=True)

# sample 8% of train split and 80% of valid split
sentences_split_train = sentences_split_train.sample(frac=0.08, random_state=42)
sentences_split_valid = sentences_split_valid.sample(frac=0.8, random_state=42)

print("sentences_split_train_size", len(sentences_split_train))
print("sentences_split_valid_size", len(sentences_split_valid))

from transformers import TrainerCallback


class ProgressLoggingCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        progress_percent = (state.global_step / self.total_steps) * 100
        wandb.log({"progress_percent": progress_percent})


kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold in range(num_folds):
    print(f"Training for fold {fold}")

    # 'train' と 'valid' それぞれに対して KFold を適用
    train_index_train, val_index_train = list(kf.split(sentences_split_train))[fold]
    train_index_valid, val_index_valid = list(kf.split(sentences_split_valid))[fold]

    # サブセットを取得
    train_fold_train = sentences_split_train.iloc[train_index_train].reset_index(
        drop=True
    )
    valid_fold_train = sentences_split_train.iloc[val_index_train].reset_index(
        drop=True
    )

    train_fold_valid = sentences_split_valid.iloc[train_index_valid].reset_index(
        drop=True
    )
    valid_fold_valid = sentences_split_valid.iloc[val_index_valid].reset_index(
        drop=True
    )

    # 各foldでの 'train' と 'valid' のデータを結合
    train_fold = pd.concat([train_fold_train, train_fold_valid], axis=0).reset_index(
        drop=True
    )
    valid_fold = pd.concat([valid_fold_train, valid_fold_valid], axis=0).reset_index(
        drop=True
    )

    print("train_fold size", len(train_fold))
    print("valid_fold size", len(valid_fold))

    train_dataset_fold = W2v2Dataset(train_fold)
    valid_dataset_fold = W2v2Dataset(valid_fold)

    total_steps = (
        len(train_dataset_fold)
        // training_args.per_device_train_batch_size
        * training_args.num_train_epochs
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset_fold,
        eval_dataset=valid_dataset_fold,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            ProgressLoggingCallback(total_steps),
        ],
    )

    trainer.train()
    fold_output_dir = f"{output_dir}/fold_{fold}"
    trainer.save_model(fold_output_dir)
    model.save_pretrained(fold_output_dir)
    processor.feature_extractor.save_pretrained(fold_output_dir)


# 各 fold で保存されたモデルの path
fold_model_paths = [f"{output_dir}/fold_{fold}/" for fold in range(num_folds)]

models = [
    Wav2Vec2ForCTC.from_pretrained(fold_model_path)
    for fold_model_path in fold_model_paths
]

from collections import OrderedDict

average_params = OrderedDict()

for name, param in models[0].named_parameters():
    average_params[name] = sum(model.state_dict()[name] for model in models) / len(
        models
    )

ensemble_model = Wav2Vec2ForCTC.from_pretrained(fold_model_paths[0])
ensemble_model.load_state_dict(average_params)

ensemble_model.save_pretrained(f"{output_dir}/ensemble")
