"""
PLYファイルからz軸+方向を正面とした深度画像を生成するスクリプト
- カメラ位置: z = -∞、+z 方向を見る（正射影）
- 明るい = 手前（z大）、暗い = 奥（z小）
- 出力: 16bit グレースケールPNG（精度重視）および 8bit プレビュー用PNG
"""

import struct
import numpy as np
from pathlib import Path
import sys


def read_ply_vertices(ply_path: str) -> np.ndarray:
    """binary_little_endian PLY から頂点(x,y,z)を読み込む"""
    path = Path(ply_path)
    with open(path, "rb") as f:
        # ヘッダー解析
        n_vertices = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        print(f"  頂点数: {n_vertices:,}, 面数: {n_faces:,}")

        # 頂点データ読み込み (float32 x3)
        vertex_bytes = n_vertices * 3 * 4  # 3 floats × 4 bytes
        raw = f.read(vertex_bytes)
        vertices = np.frombuffer(raw, dtype=np.float32).reshape(n_vertices, 3)

    return vertices  # shape: (N, 3), columns: [x, y, z]


def generate_depth_image(
    vertices: np.ndarray,
    resolution: float = None,
    image_size: int = 2048,
) -> tuple[np.ndarray, dict]:
    """
    XY平面に正射影してz値を深度画像に変換

    Args:
        vertices: (N, 3) array of [x, y, z]
        resolution: メートル/ピクセル。Noneなら image_size に合わせて自動計算
        image_size: 自動計算時の長辺ピクセル数

    Returns:
        depth_norm: float32 [0,1] の深度マップ（0=背景/奥, 1=最手前）
        info: スケール情報など
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if resolution is None:
        # 長辺が image_size ピクセルになるよう解像度を決定
        resolution = max(x_range, y_range) / image_size

    w = max(1, int(np.ceil(x_range / resolution)))
    h = max(1, int(np.ceil(y_range / resolution)))

    print(f"  空間範囲: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}], z=[{z_min:.3f}, {z_max:.3f}]")
    print(f"  解像度: {resolution:.6f} units/pixel → 画像サイズ: {w}×{h}")

    # ピクセル座標に変換（y軸は画像では上下反転）
    px = ((x - x_min) / resolution).astype(np.int32)
    py = ((y_max - y) / resolution).astype(np.int32)  # 上下反転でy+が画像上部

    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    # 深度バッファ: 初期値 -inf（未到達）
    depth_buf = np.full((h, w), -np.inf, dtype=np.float32)

    # 各ピクセルに最大z（最手前）を書き込む
    # numpy の scatter で高速処理
    flat_idx = py * w + px
    order = np.argsort(z)  # z昇順でソートし後から書くほど大きい値
    np.maximum.at(depth_buf.ravel(), flat_idx, z)

    # 背景マスク（点が投影されていないピクセル）
    bg_mask = depth_buf == -np.inf

    # [0, 1] に正規化（背景は 0）
    depth_norm = np.zeros((h, w), dtype=np.float32)
    if z_max > z_min:
        depth_norm[~bg_mask] = (depth_buf[~bg_mask] - z_min) / (z_max - z_min)

    info = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "resolution": resolution,
        "width": w, "height": h,
        "background_ratio": bg_mask.sum() / (w * h),
    }
    return depth_norm, info


def save_depth_images(depth_norm: np.ndarray, out_stem: str, info: dict):
    """16bit PNG と 8bit プレビュー PNG を保存"""
    try:
        import cv2
    except ImportError:
        print("  OpenCVが見つかりません。pip install opencv-python でインストールしてください。")
        _save_without_cv2(depth_norm, out_stem)
        return

    # 16bit グレースケール（精度重視・研究用）
    depth_16 = (depth_norm * 65535).astype(np.uint16)
    out_16 = f"{out_stem}_depth_16bit.png"
    cv2.imwrite(out_16, depth_16)
    print(f"  16bit PNG 保存: {out_16}")

    # 8bit プレビュー（視認用）
    depth_8 = (depth_norm * 255).astype(np.uint8)
    out_8 = f"{out_stem}_depth_8bit.png"
    cv2.imwrite(out_8, depth_8)
    print(f"  8bit  PNG 保存: {out_8}")

    # カラーマップ版（Jet: 奥=青, 手前=赤）
    color = cv2.applyColorMap(depth_8, cv2.COLORMAP_JET)
    # 背景（depth=0）を黒に戻す
    bg = depth_norm == 0
    color[bg] = [0, 0, 0]
    out_color = f"{out_stem}_depth_color.png"
    cv2.imwrite(out_color, color)
    print(f"  カラー PNG 保存: {out_color}")


def _save_without_cv2(depth_norm: np.ndarray, out_stem: str):
    """OpenCVなしで保存（pillow フォールバック）"""
    try:
        from PIL import Image
        img = Image.fromarray((depth_norm * 255).astype(np.uint8), mode="L")
        out = f"{out_stem}_depth_8bit.png"
        img.save(out)
        print(f"  8bit PNG 保存 (Pillow): {out}")
    except ImportError:
        # 生バイトでPPM保存（最終手段）
        h, w = depth_norm.shape
        out = f"{out_stem}_depth.pgm"
        with open(out, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write((depth_norm * 255).astype(np.uint8).tobytes())
        print(f"  PGM 保存: {out}")


def flatten_depth_curvature(depth_16bit_path: str, degree: int = 3) -> None:
    """
    既存の16bit深度PNGを読み込み、全体の湾曲を多項式フィッティングで除去する。
    residual = actual - fitted を[0,1]に再正規化して保存。
    """
    try:
        import cv2
    except ImportError:
        print("OpenCVが必要です: pip install opencv-python")
        return

    path = Path(depth_16bit_path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(depth_16bit_path)
    if raw.dtype != np.uint16:
        print(f"  警告: uint16 ではなく {raw.dtype} です。結果が不正確な可能性があります。")

    depth = raw.astype(np.float32) / 65535.0
    H, W = depth.shape
    bg_mask = depth == 0.0
    fg_mask = ~bg_mask

    if fg_mask.sum() < 100:
        print("  警告: 前景ピクセルが少なすぎます。処理を中断します。")
        return

    ys, xs = np.where(fg_mask)
    x_norm = (xs / (W - 1)) * 2.0 - 1.0
    y_norm = (ys / (H - 1)) * 2.0 - 1.0
    z_vals = depth[fg_mask].astype(np.float64)

    N_FIT = 50_000
    n_fg = len(z_vals)
    if n_fg > N_FIT:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_fg, size=N_FIT, replace=False)
        x_fit, y_fit, z_fit = x_norm[idx], y_norm[idx], z_vals[idx]
    else:
        x_fit, y_fit, z_fit = x_norm, y_norm, z_vals

    def poly_features(x, y, deg):
        cols = []
        for total in range(deg + 1):
            for a in range(total + 1):
                cols.append((x ** a) * (y ** (total - a)))
        return np.column_stack(cols)

    coeffs, _, _, _ = np.linalg.lstsq(poly_features(x_fit, y_fit, degree), z_fit, rcond=None)

    z_fitted = poly_features(x_norm, y_norm, degree) @ coeffs
    residual = z_vals - z_fitted

    r_min, r_max = residual.min(), residual.max()
    flat = np.zeros((H, W), dtype=np.float32)
    if r_max - r_min > 1e-9:
        flat[fg_mask] = ((residual - r_min) / (r_max - r_min)).astype(np.float32)

    stem = str(path).replace("_depth_16bit.png", "")

    flat_16 = (flat * 65535).astype(np.uint16)
    cv2.imwrite(f"{stem}_depth_flat_16bit.png", flat_16)
    print(f"  16bit PNG 保存: {stem}_depth_flat_16bit.png")

    flat_8 = (flat * 255).astype(np.uint8)
    cv2.imwrite(f"{stem}_depth_flat_8bit.png", flat_8)
    print(f"  8bit  PNG 保存: {stem}_depth_flat_8bit.png")

    color = cv2.applyColorMap(flat_8, cv2.COLORMAP_JET)
    color[bg_mask] = [0, 0, 0]
    cv2.imwrite(f"{stem}_depth_flat_color.png", color)
    print(f"  カラー PNG 保存: {stem}_depth_flat_color.png")


def segment_depth_regions(
    depth_flat_path: str,
    n_bands: int = 16,
    min_size: int = 500,
    blur_sigma: float = 1.5,
) -> None:
    """
    フラット化済み16bit深度PNGを深度バンドに量子化し、
    連結成分ごとに固有色で塗り分けたセグメンテーション画像を保存する。
    """
    try:
        import cv2
    except ImportError:
        print("OpenCVが必要です: pip install opencv-python")
        return

    path = Path(depth_flat_path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(depth_flat_path)

    depth = raw.astype(np.float32) / 65535.0
    H, W = depth.shape
    bg_mask = depth == 0.0

    # Gaussianブラーでノイズ除去（前景のみ有効）
    blurred = cv2.GaussianBlur(depth, (0, 0), blur_sigma)
    blurred[bg_mask] = 0.0

    # 前景の深度範囲を計算してバンド境界を決定
    fg_vals = blurred[~bg_mask]
    d_min, d_max = fg_vals.min(), fg_vals.max()
    band_edges = np.linspace(d_min, d_max, n_bands + 1)

    # 各バンドで連結成分を抽出し、有効領域にラベルを割り当てる
    label_map = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    for i in range(n_bands):
        lo, hi = band_edges[i], band_edges[i + 1]
        mask = ((blurred >= lo) & (blurred < hi) & (~bg_mask)).astype(np.uint8)
        if i == n_bands - 1:
            mask = ((blurred >= lo) & (~bg_mask)).astype(np.uint8)

        n_comp, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for c in range(1, n_comp):
            if stats[c, cv2.CC_STAT_AREA] >= min_size:
                label_map[comp_labels == c] = next_label
                next_label += 1

    n_labels = next_label - 1
    print(f"  検出領域数: {n_labels}")

    # HSV色空間で等間隔に色を割り当て
    colormap = np.zeros((H, W, 3), dtype=np.uint8)
    if n_labels > 0:
        hsv_lut = np.zeros((n_labels, 1, 3), dtype=np.uint8)
        for lbl in range(n_labels):
            hue = int((lbl / n_labels) * 180)
            hsv_lut[lbl, 0] = [hue, 220, 200]
        rgb_lut = cv2.cvtColor(hsv_lut, cv2.COLOR_HSV2BGR).reshape(n_labels, 3)

        for lbl in range(1, n_labels + 1):
            colormap[label_map == lbl] = rgb_lut[lbl - 1]

    stem = str(path).replace("_depth_flat_16bit.png", "")

    cv2.imwrite(f"{stem}_depth_seg_colormap.png", colormap)
    print(f"  セグメンテーション保存: {stem}_depth_seg_colormap.png")

    # flat_color画像との半透明オーバーレイ
    flat_color_path = str(path).replace("_depth_flat_16bit.png", "_depth_flat_color.png")
    base = cv2.imread(flat_color_path)
    if base is not None:
        overlay = cv2.addWeighted(base, 0.5, colormap, 0.5, 0)
        overlay[bg_mask] = [0, 0, 0]
        cv2.imwrite(f"{stem}_depth_seg_overlay.png", overlay)
        print(f"  オーバーレイ保存: {stem}_depth_seg_overlay.png")
    else:
        print(f"  警告: {flat_color_path} が見つからないためオーバーレイをスキップ")


def detect_motif_patterns(
    depth_flat_path: str,
    patch_size: int = 64,
    stride: int = 32,
    n_clusters: int = 10,
    n_orient_bins: int = 8,
    min_fg_ratio: float = 0.5,
    min_grad_mean: float = 0.003,
) -> None:
    """
    フラット化済み深度画像の勾配方向ヒストグラムをパッチ単位で抽出し、
    K-meansで文様パターンをクラスタリングして色分けする。
    """
    try:
        import cv2
    except ImportError:
        print("OpenCVが必要です: pip install opencv-python")
        return

    path = Path(depth_flat_path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(depth_flat_path)

    depth = raw.astype(np.float32) / 65535.0
    H, W = depth.shape
    bg_mask = depth == 0.0

    blurred = cv2.GaussianBlur(depth, (0, 0), 1.0)
    blurred[bg_mask] = 0.0

    # 勾配計算
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(np.abs(gy), np.abs(gx))  # [0, π/2]、符号なし方向

    # パッチ単位で特徴量抽出
    patch_coords = []
    features = []
    flat_labels = []  # -1=背景スキップ、-2=平坦パッチ

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch_bg = bg_mask[r:r + patch_size, c:c + patch_size]
            fg_ratio = 1.0 - patch_bg.mean()
            if fg_ratio < min_fg_ratio:
                patch_coords.append((r, c))
                flat_labels.append(-1)
                continue

            patch_mag = magnitude[r:r + patch_size, c:c + patch_size]
            patch_ang = angle[r:r + patch_size, c:c + patch_size]

            if patch_mag.mean() < min_grad_mean:
                patch_coords.append((r, c))
                flat_labels.append(-2)
                continue

            # 8方向ビン勾配ヒストグラム（勾配強度で重み付け）
            bins = np.linspace(0, np.pi / 2, n_orient_bins + 1)
            hist, _ = np.histogram(patch_ang.ravel(), bins=bins,
                                   weights=patch_mag.ravel())
            norm = np.linalg.norm(hist)
            feat = hist / (norm + 1e-8)
            patch_coords.append((r, c))
            features.append(feat.astype(np.float32))
            flat_labels.append(len(features) - 1)

    print(f"  有効パッチ数: {len(features)}")

    # K-means クラスタリング
    cluster_ids = np.full(len(patch_coords), -1, dtype=np.int32)
    if len(features) >= n_clusters:
        feat_mat = np.array(features, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(feat_mat, n_clusters, None, criteria, 10,
                                  cv2.KMEANS_PP_CENTERS)
        label_iter = iter(labels.ravel())
        for i, fl in enumerate(flat_labels):
            if fl >= 0:
                cluster_ids[i] = next(label_iter)
            elif fl == -2:
                cluster_ids[i] = -2  # 平坦
    else:
        print(f"  警告: 有効パッチが少なすぎます（{len(features)} < {n_clusters}）")

    # HSV色テーブル（クラスタ数分）
    hsv_lut = np.zeros((n_clusters, 1, 3), dtype=np.uint8)
    for k in range(n_clusters):
        hsv_lut[k, 0] = [int((k / n_clusters) * 180), 220, 200]
    rgb_lut = cv2.cvtColor(hsv_lut, cv2.COLOR_HSV2BGR).reshape(n_clusters, 3)

    # パッチ位置に色を塗る
    colormap = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (r, c) in enumerate(patch_coords):
        cid = cluster_ids[idx]
        if cid >= 0:
            colormap[r:r + patch_size, c:c + patch_size] = rgb_lut[cid]
        elif cid == -2:
            colormap[r:r + patch_size, c:c + patch_size] = [40, 40, 40]
    colormap[bg_mask] = [0, 0, 0]

    stem = str(path).replace("_depth_flat_16bit.png", "")
    cv2.imwrite(f"{stem}_depth_motif_colormap.png", colormap)
    print(f"  文様マップ保存: {stem}_depth_motif_colormap.png")

    flat_color_path = str(path).replace("_depth_flat_16bit.png", "_depth_flat_color.png")
    base = cv2.imread(flat_color_path)
    if base is not None:
        overlay = cv2.addWeighted(base, 0.5, colormap, 0.5, 0)
        overlay[bg_mask] = [0, 0, 0]
        cv2.imwrite(f"{stem}_depth_motif_overlay.png", overlay)
        print(f"  オーバーレイ保存: {stem}_depth_motif_overlay.png")


def process_ply(ply_path: str, image_size: int = 2048, resolution: float = None):
    print(f"\n[処理中] {ply_path}")
    vertices = read_ply_vertices(ply_path)

    depth_norm, info = generate_depth_image(vertices, resolution=resolution, image_size=image_size)

    print(f"  背景率: {info['background_ratio']:.1%}")

    stem = Path(ply_path).stem
    out_dir = Path(ply_path).parent
    out_stem = str(out_dir / stem)

    save_depth_images(depth_norm, out_stem, info)

    # スケール情報をテキストで保存
    info_path = f"{out_stem}_depth_info.txt"
    with open(info_path, "w") as f:
        f.write(f"PLY: {ply_path}\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print(f"  スケール情報: {info_path}")


if __name__ == "__main__":
    # --flatten モード: 既存の16bit深度PNGから全体湾曲を除去
    if "--flatten" in sys.argv:
        fidx = sys.argv.index("--flatten")
        if fidx + 1 >= len(sys.argv):
            print("使用法: generate_depth.py --flatten <16bit_png_path> [--degree 3]")
            sys.exit(1)
        depth_png = sys.argv[fidx + 1]
        degree = 3
        if "--degree" in sys.argv:
            didx = sys.argv.index("--degree")
            degree = int(sys.argv[didx + 1])
        print(f"\n[フラット化] {depth_png}  (degree={degree})")
        flatten_depth_curvature(depth_png, degree=degree)
        sys.exit(0)

    # --segment モード: フラット化済み16bit PNGを深度バンドで領域分割・色分け
    if "--segment" in sys.argv:
        sidx = sys.argv.index("--segment")
        if sidx + 1 >= len(sys.argv):
            print("使用法: generate_depth.py --segment <flat_16bit_png> [--bands 16] [--min-size 500]")
            sys.exit(1)
        flat_png = sys.argv[sidx + 1]
        n_bands = 16
        min_size = 500
        if "--bands" in sys.argv:
            n_bands = int(sys.argv[sys.argv.index("--bands") + 1])
        if "--min-size" in sys.argv:
            min_size = int(sys.argv[sys.argv.index("--min-size") + 1])
        print(f"\n[セグメンテーション] {flat_png}  (bands={n_bands}, min_size={min_size})")
        segment_depth_regions(flat_png, n_bands=n_bands, min_size=min_size)
        sys.exit(0)

    # --motif モード: 勾配パターンで文様を自動クラスタリング
    if "--motif" in sys.argv:
        midx = sys.argv.index("--motif")
        if midx + 1 >= len(sys.argv):
            print("使用法: generate_depth.py --motif <flat_16bit_png> [--patch 64] [--clusters 10] [--stride 32]")
            sys.exit(1)
        motif_png = sys.argv[midx + 1]
        patch_size = 64
        n_clusters = 10
        stride = 32
        if "--patch" in sys.argv:
            patch_size = int(sys.argv[sys.argv.index("--patch") + 1])
        if "--clusters" in sys.argv:
            n_clusters = int(sys.argv[sys.argv.index("--clusters") + 1])
        if "--stride" in sys.argv:
            stride = int(sys.argv[sys.argv.index("--stride") + 1])
        print(f"\n[文様検出] {motif_png}  (patch={patch_size}, clusters={n_clusters}, stride={stride})")
        detect_motif_patterns(motif_png, patch_size=patch_size, stride=stride, n_clusters=n_clusters)
        sys.exit(0)

    # 引数: PLYファイルパス [画像長辺ピクセル数] [解像度 units/pixel]
    if len(sys.argv) < 2:
        # デフォルト: dataディレクトリの全PLYを処理
        data_dir = Path(__file__).parent
        ply_files = sorted(data_dir.glob("*.ply"))
        if not ply_files:
            print("PLYファイルが見つかりません。引数でパスを指定してください。")
            sys.exit(1)
        for p in ply_files:
            process_ply(str(p))
    else:
        ply_path = sys.argv[1]
        image_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        resolution = float(sys.argv[3]) if len(sys.argv) > 3 else None
        process_ply(ply_path, image_size=image_size, resolution=resolution)
