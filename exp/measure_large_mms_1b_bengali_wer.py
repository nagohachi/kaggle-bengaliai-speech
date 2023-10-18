import datetime
import jiwer
from pathlib import Path
from functools import partial

import pandas as pd
import pyctcdecode
from tqdm import tqdm

import librosa

import pyctcdecode
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from bnunicodenormalizer import Normalizer

# import cloudpickle as cpkl

# paths
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"
TRAIN = DATA / "train_mp3s"
TEST = DATA / "test_mp3s"

MODEL_PATH = INPUT / "saved_model_large-mms-1b-bengali/"
LM_PATH = INPUT / "arijitx-full-model/wav2vec2-xls-r-300m-bengali/language_model/"

model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# constants
SAMPLING_RATE = 16000

# split = train のもののみを使う
train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
train = train[train["split"] == "train"]

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

# processor_with_lm = Wav2Vec2ProcessorWithLM(
#     feature_extractor=processor.feature_extractor,
#     tokenizer=processor.tokenizer,
#     decoder=decoder,
# )


# prepare dataloader
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


collate_func = partial(
    processor.feature_extractor,
    # processor_with_lm.feature_extractor,
    return_tensors="pt",
    sampling_rate=SAMPLING_RATE,
    padding=True,
)


def create_test_loader():
    test = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
    test_audio_paths = [str(TEST / f"{aid}.mp3") for aid in test["id"].values]
    test_dataset = BengaliSRTestDataset(test_audio_paths, SAMPLING_RATE)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_func,
        drop_last=False,
        pin_memory=True,
    )
    return test_loader


def create_train_loader(sampling_size, random_seed):
    train_random = train.sample(sampling_size, random_state=random_seed)
    train_audio_paths = [str(TRAIN / f"{aid}.mp3") for aid in train_random["id"].values]
    train_dataset = BengaliSRTestDataset(train_audio_paths, SAMPLING_RATE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_func,
        drop_last=False,
        pin_memory=True,
    )
    return train_random, train_loader


# process
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


# calculate wer


def mean_wer(solution, submission):
    sum_wer = 0
    for s, t in zip(solution, submission):
        sum_wer += jiwer.wer(s, t)
    return sum_wer / len(solution)


# inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


def inference(sampling_size, random_seed, beam_width) -> float:
    train_random, train_loader = create_train_loader(sampling_size, random_seed)
    pred_sentence_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader)):
            x = batch["input_values"]
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(True):
                y = model(x).logits
            y = y.detach().cpu().numpy()

            # for l in y:
            #     sentence = processor_with_lm.decode(
            #         l,
            #         beam_width=beam_width,
            #         alpha=0.3802723523729998,
            #         beta=0.053996879617918436,
            #     ).text
            #     pred_sentence_list.append(sentence)
            for l in y:
                beam = decoder.decode_beams(l, beam_width=beam_width)
                s = beam[0][0]
                pred_sentence_list.append(s)

            del x, y
    del train_loader

    pp_pred_sentence_list = [postprocess(sentence) for sentence in pred_sentence_list]
    wer = mean_wer(train_random["sentence"], pp_pred_sentence_list)
    return wer


# run
if __name__ == "__main__":
    import time

    sampling_size = 3000
    beam_widths = [512, 768, 1024]
    current_time_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    for beam_width in beam_widths:
        # 時間計測
        start = time.time()
        wers = []
        measured_cases = 0
        wer = inference(sampling_size, 42, beam_width)
        measured_cases += sampling_size
        wers.append(wer)

        end = time.time()
        with open("measure_saved_model_wer.txt", "a") as f:
            # かかった時間
            f.write(
                f"beam_width: {beam_width}, mean_wer:{sum(wers)/len(wers):6f}, time per case: {(end - start) / measured_cases} at {current_time_str}\n"
            )
        print(f"beam_width: {beam_width}, mean_wer:{sum(wers)/len(wers)}")
