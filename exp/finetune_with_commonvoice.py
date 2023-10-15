#!/usr/bin/env python
# coding: utf-8

# - This is a training demo, you can run this code locally, using better GPUs.
# - The inference part is here: [Bengali SR wav2vec_v1_bengali [Inference]](https://www.kaggle.com/takanashihumbert/bengali-sr-wav2vec-v1-bengali-inference), it scores **0.445** on the leaderboard.
# - Feel free to upvote, thanks!

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyctcdecode
import torch
import torchaudio
import torchaudio.transforms as tat
from datasets import load_metric
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)

from sklearn.model_selection import KFold

import wandb

wandb.init(
    project="wav2vec2-small-training-with-commonvoice",
    name="nagohachi",
    config={
        "epochs": 6,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "folds": 4,
    },
)

warnings.filterwarnings("ignore")
torchaudio.set_audio_backend("soundfile")


### hyper-parameters
SR = 16000
torch.backends.cudnn.benchmark = True
from pathlib import Path

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "common-voice-13-bengali-normalized"
TRAIN_TEST = DATA / "train_test"

output_dir = INPUT / "saved_model-finetune-with-commonvoice"
MODEL_PATH = INPUT / "wav2vec2-small-70000-clear80p-with-unigram/ensemble"
LM_PATH = INPUT / "arijitx-full-model/wav2vec2-xls-r-300m-bengali/language_model"

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {
    k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
}

decoder = pyctcdecode.build_ctcdecoder(
    list(sorted_vocab_dict.keys()),
    str(LM_PATH) + "/5gram.bin",
    str(LM_PATH) + "/unigrams.txt",
)
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder,
)


# - From @mbmmurad's [Dataset overlaps with CommonVoice 11 bn](https://www.kaggle.com/code/mbmmurad/dataset-overlaps-with-commonvoice-11-bn), The competition dataset might contain the audios of the mozilla-foundation/common_voice_11_0 dataset. Here I just simply exclude them from the validation set.
# - Also, I use @UmongSain's normalized data [here](https://www.kaggle.com/code/umongsain/macro-normalization/notebook). Thanks to him!

train_df = pd.read_csv(DATA / "train.tsv", sep="\t")
valid_df = pd.read_csv(DATA / "test.tsv", sep="\t")
all_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

# all_df の "path" 列からすべて ".mp3" を除く
all_df["path"] = all_df["path"].str.replace(".mp3", "")

# train_ids と valid_ids を .csv に保存、"id" 列に id を入れる
# pd.DataFrame({"id": train_ids}).to_csv(INPUT / "train_ids.csv", index=False)
# pd.DataFrame({"id": valid_ids}).to_csv(INPUT / "valid_ids.csv", index=False)

# # all_ids から train_ids と valid_ids を除外したものを exclusive_ids とする
# train_ids_set = set(train_ids)
# valid_ids_set = set(valid_ids)

# exclusive_ids = [
#     all_id
#     for all_id in all_ids
#     if (all_id not in train_ids_set) and (all_id not in valid_ids_set)
# ]

# # INPUT / exclusive_ids.csv に保存
# pd.DataFrame({"id": exclusive_ids}).to_csv(INPUT / "exclusive_ids.csv", index=False)
# exit(0)


class W2v2Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.pathes = df["path"].values
        self.sentences = df["sentence"].values
        self.resampler = tat.Resample(32000, SR)

    def __getitem__(self, idx):
        apath = TRAIN_TEST / f"{self.pathes[idx]}.mp3"
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


# - In kaggle notebook, there is an error: **cannot import name 'compute_measures' from 'jiwer' (unknown location)**. But in my local notebook, there is no such error.


wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


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

# _ = model.half()
# _ = model.to("cuda")

# you can freeze some params
model.freeze_feature_extractor()


# - As a demo, "**num_train_epochs**", "**eval_steps**" and "**early_stopping_patience**" are set to very small values, you can make them larger.
# - If there is no error about jiwer, you can set **metric_for_best_model**="wer", and remember to set **greater_is_better**=False and use **compute_metrics**.


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    group_by_length=False,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    save_strategy="steps",
    # max_steps=12000,  # you can change to "num_train_epochs"
    num_train_epochs=5,
    fp16=True,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=500,
    learning_rate=5e-5,
    warmup_steps=600,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    prediction_loss_only=False,
    auto_find_batch_size=True,
    report_to="wandb",
    remove_unused_columns=True,
)

num_folds = 4
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

fold = 0
for train_index, val_index in kf.split(all_df):
    # 各foldでの 'train' と 'valid' のデータを結合
    train_fold = all_df.iloc[train_index]
    valid_fold = all_df.iloc[val_index]

    print("train_fold size", len(train_fold))
    print("valid_fold size", len(valid_fold))

    train_dataset_fold = W2v2Dataset(train_fold)
    valid_dataset_fold = W2v2Dataset(valid_fold)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset_fold,
        eval_dataset=valid_dataset_fold,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    fold_output_dir = f"{output_dir}/fold_{fold}"
    trainer.save_model(fold_output_dir)
    model.save_pretrained(fold_output_dir)
    processor.feature_extractor.save_pretrained(fold_output_dir)

    fold += 1


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
