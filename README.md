# kaggle-Bengali.AI_Speech-Recognition
https://www.kaggle.com/competitions/bengaliai-speech

## Environment
- Ubuntu 22.04 LTS
- DRAM 64GB
- VRAM 12GB (RTX 3060)

## Installation
```poetry install```

## Directories, files
- [src](src) : 提出用プログラムなど、重要部分
  - [bengali-finetuning-baseline-wav2vec2-inference.ipynb](src/bengali-finetuning-baseline-wav2vec2-inference.ipynb)
    - 提出用ノートブック
  - [convert_to_wav.sh](src/convert_to_wav.sh)
    - .mp3 を .wav に変換するためのスクリプト。手元環境で 2 日くらいかかった
- [random](random) : 手元での試行錯誤など
  - [beam_range_test.py](random/beam_range_test.py)
    - beam_width その他のハイパーパラメータを調整して train 内の wer を計測するためのファイル
  - [finetune.ipynb](random/finetune.ipynb)
    - finetuning するために作ったけど後回し
  - [measure_train_wer.ipynb](random/measure_train_wer.ipynb)
    - train 内の wer を計測するためのプログラム。beam_range_test の下位互換
  - [noise_reduction_test.ipynb](random/noise_reduction.ipynb), [noise_reduction.ipynb](random/noise_reduction.ipynb)
    - ノイズ減らしたら wer 下がるかなと思ったけどダダ上がりだったのでゴミになったやつ