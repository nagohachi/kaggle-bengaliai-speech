#!/bin/bash

# ディレクトリのパスを変数に設定
SRC_DIR="../input/bengaliai-speech/train_mp3s"
DST_DIR="../input/train_wavs"

# .wav 用のディレクトリが存在しない場合、作成
mkdir -p "$DST_DIR"

# .mp3 ファイルの数をカウント
TOTAL_FILES=$(find "$SRC_DIR" -type f -name "*.mp3" | wc -l)
echo "Total .mp3 files: $TOTAL_FILES"

# .mp3 ファイルを .wav ファイルに変換
COUNTER=0
for mp3file in "$SRC_DIR"/*.mp3; do
    # ファイル名のみを取得
    filename=$(basename -- "$mp3file")
    # .mp3 拡張子を削除
    filename_noext="${filename%.*}"
    # ffmpeg を使って .mp3 から .wav に変換（出力を非表示にする）
    ffmpeg -i "$mp3file" "$DST_DIR/$filename_noext.wav" > /dev/null 2>&1
    COUNTER=$((COUNTER+1))
    # 100 の倍数のときに進捗を表示
    if [ $((COUNTER % 100)) -eq 0 ]; then
        echo "Converted $COUNTER/$TOTAL_FILES"
    fi
done
