# 土器表面パターン セグメンテーション

## プロジェクト概要
修士論文用。PLY形式の土器3Dスキャンデータから表面の凹凸パターンを検出し、
一つの模様単位を自動セグメンテーションするC++プログラム。

## スタック
- 言語: C++17
- ビルド: CMake 3.16+
- ライブラリ: OpenCV, OpenGL, GLFW, GLEW, GLM
- パッケージ管理: vcpkg（Windows） / Homebrew（macOS）

## ディレクトリ構成
```
pottery_segmentation/
├── CLAUDE.md
├── CMakeLists.txt
├── pottery_segmentation.cpp   # メインプログラム
├── data/                      # PLYファイル置き場
└── build/                     # ビルド生成物（gitignore）
```

## ビルドコマンド
```bash
# Windows
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# macOS
cmake -B build -S .
cmake --build build
```

## 実行コマンド
```bash
# Windows
.\build\Release\pottery_segmentation.exe data\pottery.ply

# macOS
./build/pottery_segmentation data/pottery.ply
```

## 処理パイプライン（変更時は必ずこの順序を守る）
1. PLYバイナリ読み込み（float x,y,z + face list形式）
2. 法線ベクトル計算（面法線の頂点への加算・正規化）
3. 隣接リスト構築
4. 平均曲率の近似計算（法線変化量/距離）
5. 残差曲率計算（局所中央値との差分で土器本体の湾曲を除去）
6. 領域成長セグメンテーション（BFS）
7. OpenGLで色分け可視化

## 重要な設計判断
- 土器全体の湾曲と模様の凹凸を分離するために「残差曲率」を使用
- PLYデータに法線情報がないため、メッシュ面から計算している
- macOSはOpenGL 4.1までのため、シェーダーは`#version 410 core`を使用

## 調整パラメータ（main関数内）
- `threshold`: 残差曲率の閾値（小さい→細かく分割、大きい→大まかに分割）
- `min_segment_size`: 最小セグメント頂点数（ノイズ除去）

## 注意事項
- PLYファイルはbinary_little_endian形式（テキスト形式は未対応）
- 頂点数が700万超のため、隣接リスト構築に時間がかかる（正常）
- macOSでOpenGL非推奨警告が出るが動作に影響なし（GL_SILENCE_DEPRECATIONで抑制済み）
