from pathlib import Path
import pandas as pd
from bnunicodenormalizer import Normalizer

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "bengaliai-speech"

train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
choose_num = 20000
# train["split"] が "valid" のもののみを抽出
train = train[train["split"] == "valid"].sample(choose_num, random_state=42)

bnorm = Normalizer()
def postprocess(sentence):
    words = [bnorm(word)["normalized"] for word in sentence.split()]
    sentence = " ".join([word for word in words if word is not None])
    return sentence

print(train["sentence"][:5])

# train の "sentence" を正規化
train["sentence_postprocessed"] = train["sentence"].apply(postprocess)

final_letter_dict = {}
# sentence_postprocessed の最後の文字を集計
for sentence in train["sentence_postprocessed"]:
    final_letter = sentence[-1]
    if final_letter in final_letter_dict:
        final_letter_dict[final_letter] += 1
    else:
        final_letter_dict[final_letter] = 1

print(final_letter_dict)