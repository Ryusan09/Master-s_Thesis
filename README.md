# Jomon Pattern Analyzer

縄文土器破片の3Dスキャン(PLY点群)から繰り返し模様を検出し、模様が一周している区間を特定するためのツール。

## できること

- PLY点群を上(Z軸方向)から見た深度マップに変換
- 局所適応的二値化(Sauvola法)で模様の線を抽出
- テンプレートマッチング(マルチスケール+回転)で同じ模様を検出
- ORB特徴量で回転・スケール変化に強い類似領域検出
- 結果を画像とCSVで出力

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

### 自動ROI(おまかせモード)

```bash
python jomon_pattern_analyzer.py data/shard.ply --output_dir results/
```

模様密度が最も高い領域を自動でテンプレートとして選んで類似領域を探す。

### 手動ROIで精密に

1. まず1回走らせて `results/<name>_01_depth.png` を確認
2. 「これが模様1単位だ」と思う部分の (X, Y, 幅, 高さ) をピクセル座標で読み取る
3. `--roi` で指定して再実行

```bash
python jomon_pattern_analyzer.py data/shard.ply \
    --roi 120 80 60 60 \
    --resolution 0.3 \
    --match_threshold 0.6
```

## 出力

`results/` に以下が生成される:

| ファイル | 内容 |
|---|---|
| `<name>_01_depth.png` | Z軸方向から見た深度マップ |
| `<name>_02_binary.png` | 局所二値化で抽出した模様線 |
| `<name>_03_template.png` | 検出に使ったテンプレート画像 |
| `<name>_04_result.png` | 3枚並びの可視化結果 |
| `<name>_matches.csv` | 検出された全マッチの座標 |

## 主なオプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--resolution` | `0.5` | 1ピクセルあたりのワールド単位。小さいほど高精細 |
| `--window_size` | `51` | Sauvola二値化の窓サイズ(奇数) |
| `--sauvola_k` | `0.2` | 二値化の厳しさ。大きいほど厳しい |
| `--template_size` | `80` | 自動ROIのサイズ(px) |
| `--roi X Y W H` | (自動) | テンプレートROIを手動指定 |
| `--match_threshold` | `0.55` | 相関しきい値 0..1 |

## うまくいかないとき

詳細は `CLAUDE.md` の「デバッグ時のチェックポイント」を参照。

- 深度マップが真っ黒 → 点群の座標スケールを確認
- 二値化がノイズだらけ → `--window_size` を大きく(75〜101)
- マッチが0件 → `--match_threshold` を 0.4 まで下げる
- マッチが多すぎる → しきい値を上げる、`--template_size` を大きく

## ライセンス・引用

研究目的での使用を想定。商用利用や論文引用前にプロジェクト管理者へ確認のこと。
