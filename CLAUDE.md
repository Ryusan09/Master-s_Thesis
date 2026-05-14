# 縄文土器模様解析プロジェクト

縄文土器破片の3Dスキャン(PLY点群)から繰り返し模様を検出し、模様がどこから始まりどこで一周しているかを特定するためのツール。

## プロジェクトの目的

縄文土器の表面には縄目文・押文・沈線文など繰り返し単位を持つ模様が施されている。本プロジェクトは、3Dスキャンされた破片の点群データから、

1. 模様を画像として抽出する(深度マップ→二値化)
2. 模様の繰り返し単位を検出する(テンプレートマッチ + ORB)
3. その単位がどこからどこまで繰り返されているか(=一周している区間)を特定する

を自動化することを目指す。

## ディレクトリ構成

```
.
├── CLAUDE.md                       # このファイル(Claude Codeへの指示書)
├── README.md                       # 人間向けの使い方
├── jomon_pattern_analyzer.py       # メインスクリプト
├── requirements.txt
├── data/                           # 入力PLYファイルを置く場所
│   └── *.ply
└── results/                        # 出力先
    ├── <name>_01_depth.png         # 深度マップ
    ├── <name>_02_binary.png        # 二値化結果
    ├── <name>_03_template.png      # 検出に使ったテンプレート
    ├── <name>_04_result.png        # 可視化(3枚並び)
    └── <name>_matches.csv          # 検出座標一覧
```

## 環境

- Python 3.10+
- 必要ライブラリ: `numpy`, `opencv-python`, `opencv-contrib-python`, `scikit-image`, `scipy`, `matplotlib`, `plyfile`

```bash
pip install -r requirements.txt
```

## 実行方法

```bash
# 自動ROI(模様密度の最も高い領域をテンプレートとして自動選択)
python jomon_pattern_analyzer.py data/shard.ply --output_dir results/

# ROIを手動指定する場合(まず1回走らせて *_01_depth.png で座標を見てから)
python jomon_pattern_analyzer.py data/shard.ply \
    --roi 120 80 60 60 --resolution 0.3 --match_threshold 0.6
```

### 主な引数

| 引数 | 既定 | 意味 |
|---|---|---|
| `--resolution` | `0.5` | 深度マップ1pxのワールド単位(小さいほど高精細・穴が増える) |
| `--window_size` | `51` | Sauvola二値化の窓サイズ(奇数)。模様1単位より少し大きく |
| `--sauvola_k` | `0.2` | Sauvolaのkパラメータ。大きくすると検出が厳しくなる |
| `--template_size` | `80` | 自動ROIのサイズ(px) |
| `--roi X Y W H` | None | テンプレートROIを手動指定 |
| `--match_threshold` | `0.55` | 正規化相関しきい値(0..1) |

## パイプライン(`jomon_pattern_analyzer.py`)

`run_pipeline()` が以下を順に実行する。各関数は独立しており差し替え可能。

1. **`load_ply_points`** — PLYからXYZ点群を読む(点群のみ前提、メッシュ非対応)
2. **`align_to_principal_plane`** — PCAで主平面を求め、最小分散方向を新Z軸にして表面を水平化
3. **`points_to_depth_map`** — XYグリッドに点群を落として各セルの最大Z値を採用、欠損は線形+最近傍補間
4. **`local_binarize`** — 大局曲面を引いた上でSauvola局所適応二値化、模様線を抽出
5. **`suggest_roi`** — 自動ROI候補(模様密度最大領域)を返す
6. **`multiscale_rotated_match`** — マルチスケール×回転で正規化相関、NMS付き
7. **`orb_similar_regions`** — ORB特徴+クラスタリングで回転・スケール不変な類似検出
8. **`visualize_results`** — 3枚並びの結果図を保存

## 今後の拡張タスク(優先度順)

Claude Codeで作業するときは、以下のいずれかから選んで進めてほしい。**進める前に必ず作業内容をユーザーに確認し、テストが通ることを確認してからコミットすること。**

### Priority 1: 一周検出ロジックの実装

現状は類似領域を点で列挙するだけ。これを「一周区間」として解釈する処理を `detect_full_cycle()` として追加する。

- 検出マッチの中心座標を主軸(PCA最大方向)に沿って並べる
- 隣接マッチ間の距離(=模様の周期)を推定(中央値ベース)
- 周期から外れる外れ値を除去
- 「最初の単位」と「最後の単位」を起点・終点としてマーク
- 弧長を計算して周回距離を出力(破片が平面近似のため曲面長は概算)

成果物: `results/<name>_05_cycle.png`(起点・終点・周期線をハイライト)、CSVに `cycle_index` 列を追加

### Priority 2: 円筒展開モードの追加

破片が大きく湾曲している場合、Z軸投影だけでは模様が歪む。`--mode cylinder` を追加して、PCAではなくRANSACで円筒面を当てはめ、表面を (θ, h, r) の円筒座標に展開する処理を `cylindrical_unwrap()` として実装する。

- 入力: 点群
- 出力: 展開後の2D画像(θ × h)
- 既存パイプライン(二値化以降)はそのまま使える

### Priority 3: 対話的ROI選択GUI

現状はROIを引数で渡すか自動。`--interactive` フラグでmatplotlibのウィジェット(RectangleSelector)で深度マップ上をドラッグしてROIを選べるようにする。

### Priority 4: バッチ処理対応

ディレクトリ内の全PLYを一括処理する `--batch` モードを追加。結果サマリーCSVも出力する。

### Priority 5: 単体テスト

`tests/test_pipeline.py` を追加し、`scripts/make_synthetic_ply.py` で生成する合成データに対する回帰テストを書く。

- 深度マップの寸法とfill ratio
- 二値化の白画素率(0でも100でもない)
- テンプレートマッチが既知の繰り返し位置を検出すること

## 開発上の注意

- **PLYは点群のみ対応**。メッシュ(faces付き)を投入したい場合は `load_ply_points` に分岐を追加する必要がある
- **座標単位はワールド単位**。PLYの単位がmmなのかmなのかは事前に確認。`--resolution` の解釈に影響
- **破片表面が「外側」か「内側」か**は `align_to_principal_plane` 内の符号反転ロジックで自動推定しているが、模様が逆向きになる場合は手動フラグを足すべき
- **依存追加は最小限に**。Open3Dは強力だがインストールが重いので、現状scipy+plyfileで完結させている
- **コミット前に必ず合成データでパイプラインが通ることを確認**(`python scripts/make_synthetic_ply.py && python jomon_pattern_analyzer.py data/synthetic.ply`)

## デバッグ時のチェックポイント

問題が起きたとき、まずどの段階で壊れているか中間出力を見る:

1. `*_01_depth.png` が真っ黒/真っ白 → 点群の座標スケールがおかしい、`align_to_principal_plane` が想定外の向きに揃えている
2. `*_01_depth.png` に穴が多い → `--resolution` を上げる、点群が疎すぎる可能性
3. `*_02_binary.png` がノイズだらけ → `--window_size` を大きく、`--sauvola_k` を上げる
4. `*_02_binary.png` がほぼ真っ黒 → 模様の凹凸が小さすぎる、`local_binarize` 内のhigh-pass強度を上げる
5. テンプレートマッチが0件 → `--match_threshold` を 0.4 程度まで下げる、ROIが模様を含んでいるか `*_03_template.png` で確認
6. テンプレートマッチが多すぎる → しきい値を上げる、`--template_size` を大きく(模様2単位を含むサイズに)
