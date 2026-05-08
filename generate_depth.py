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
