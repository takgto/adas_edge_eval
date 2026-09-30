# adas-edge-eval

KV260 上の YOLOv3 系検出器の mAP 評価ツール。

評価動画 `adas.webm`（512x256, 60fps, 1616 フレーム）上で FPS と mAP を同時に測定する。

## 使い方

### 1. KV260 でログを取る

テンプレート（`-DSAVE_LOG=1` でコンパイル）を KV260 上で実行し、
最後まで走らせて `detection_log.txt` を出力する。

### 2. ログを WSL2 に転送

```bash
scp kv260_ip:~/work/detection_log.txt logs/my_log.txt
```

### 3. mAP を計算

```bash
# 1 本のログを評価
python eval/eval_log.py logs/my_log.txt

# 複数ログを一覧表にする
python eval/report.py logs/*.txt

# Excel 出力
python eval/report.py logs/*.txt --tsv report.tsv --xlsx report.xlsx
```

## 評価指標

| 指標 | 意味 |
|---|---|
| **mAP_frame** | 各フレームの検出結果をそのフレームの GT と照合（モデル精度） |
| **mAP_display** | 実際に表示された結果を間引き・逆行を反映して照合（実検出器の精度） |

IoU 0.75 を主指標、0.5 を参考値として両方出す。

## ディレクトリ構成

```
adas-edge-eval/
├── config.yaml          # 評価設定（クラス対応、マスク、IoU 閾値）
├── ground-truth/        # 擬似 GT（1616 個の txt）
├── eval/
│   ├── eval_log.py      # ログ → mAP 計算
│   ├── report.py        # 複数ログの一覧表
│   └── mAP/             # Cartucho mAP ツール
└── logs/                # 学生のログ置き場
```

## 依存パッケージ

```
python >= 3.8
numpy, opencv-python, pyyaml
openpyxl  # --xlsx を使う場合
```
