"""
縄文土器模様解析ツール
=======================

破片PLY(点群)からZ軸方向の深度マップを生成し、局所二値化で模様を抽出、
テンプレートマッチング + ORB特徴量で類似模様を検出してハイライトする。

模様が「どこから始まりどこで一周するか」を視覚的に特定することが目的。

使い方:
    python jomon_pattern_analyzer.py input.ply --resolution 0.5 --output_dir results/

主な処理ステップ:
    1. PLYから点群読み込み
    2. PCAで主平面に整列(破片の表面を上向きに)
    3. Z軸方向の深度マップへ投影(欠損は補間)
    4. 局所適応的二値化(Sauvola法)で模様線を抽出
    5. ユーザーが選択したROI(模様の1単位)をテンプレートとして
       (a) マルチスケール正規化相関で類似領域を探索
       (b) ORB特徴量で回転・スケール不変な類似領域を探索
    6. ヒートマップとマーカーで結果を可視化
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from plyfile import PlyData
from scipy.interpolate import griddata
from skimage.filters import threshold_sauvola


# ---------------------------------------------------------------------------
# 1. PLY読み込み & 主平面アラインメント
# ---------------------------------------------------------------------------

def load_ply_points(ply_path: str) -> np.ndarray:
    """PLYファイルから (N, 3) のXYZ点群を読み込む。"""
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    pts = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    pts = pts.astype(np.float64)
    print(f"[load] {len(pts):,} points loaded from {ply_path}")
    return pts


def align_to_principal_plane(pts: np.ndarray) -> np.ndarray:
    """
    PCAで点群の主平面を求め、その平面が XY 平面と一致するように回転する。
    破片はおおむね平らなので、最小分散方向がZ軸(法線方向)になる。
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # 共分散行列の固有ベクトル。最小固有値の方向が法線。
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvecs[:, 0] が最小固有値方向 → 新Z軸。
    new_z = eigvecs[:, 0]
    new_x = eigvecs[:, 2]  # 最大分散 → 新X軸
    new_y = np.cross(new_z, new_x)

    R = np.stack([new_x, new_y, new_z], axis=1)  # 列ベクトルが新軸
    aligned = centered @ R

    # 表面側を上(+Z)に向ける。中央値より上の点が多くなるよう向きを揃える。
    if np.median(aligned[:, 2]) > 0:
        aligned[:, 2] *= -1

    print(f"[align] eigenvalues={eigvals}, flipped if needed.")
    return aligned


# ---------------------------------------------------------------------------
# 2. 深度マップ生成
# ---------------------------------------------------------------------------

@dataclass
class DepthMap:
    image: np.ndarray         # 正規化された 0..255 のグレースケール
    raw_z: np.ndarray         # 補間前の生Z値(NaN付き)
    z_interp: np.ndarray      # 補間済みZ値
    mask: np.ndarray          # 有効領域マスク(True=データあり)
    extent: tuple             # (xmin, xmax, ymin, ymax) ワールド座標
    resolution: float         # 1ピクセル = 何ワールド単位か


def points_to_depth_map(pts: np.ndarray, resolution: float = 0.5,
                        smooth_sigma: float = 1.0) -> DepthMap:
    """
    XY平面に等間隔グリッドを敷き、各セルの最大Z(=表面)を採用して深度画像化。
    欠損セルは griddata で線形補間 → ガウシアンで軽く平滑化。
    """
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    W = int(np.ceil((xmax - xmin) / resolution)) + 1
    H = int(np.ceil((ymax - ymin) / resolution)) + 1

    # ピクセル座標へ
    ix = ((pts[:, 0] - xmin) / resolution).astype(np.int32)
    iy = ((pts[:, 1] - ymin) / resolution).astype(np.int32)
    iy = (H - 1) - iy  # 画像のy軸は下向きなので反転

    raw_z = np.full((H, W), np.nan, dtype=np.float64)
    # 同じセルに複数点が落ちる場合は最大値(=表面)を採用
    # np.maximum.at は重複インデックスを正しく処理する。
    # 初期値NaNだとmax比較が壊れるので-infで初期化してから戻す。
    work = np.full((H, W), -np.inf, dtype=np.float64)
    np.maximum.at(work, (iy, ix), pts[:, 2])
    valid = np.isfinite(work) & (work > -np.inf)
    raw_z[valid] = work[valid]

    mask = ~np.isnan(raw_z)
    print(f"[depth] grid={W}x{H}, fill ratio={mask.mean():.1%}")

    # 欠損補間(有効点が少なすぎる場合はスキップ)
    z_interp = raw_z.copy()
    if mask.sum() > 10 and (~mask).any():
        yy, xx = np.indices(raw_z.shape)
        known_pts = np.stack([xx[mask], yy[mask]], axis=1)
        known_vals = raw_z[mask]
        unknown_pts = np.stack([xx[~mask], yy[~mask]], axis=1)
        # 線形補間 → 凸包外はnearestで埋める
        filled = griddata(known_pts, known_vals, unknown_pts, method="linear")
        nan_after = np.isnan(filled)
        if nan_after.any():
            filled[nan_after] = griddata(known_pts, known_vals,
                                          unknown_pts[nan_after], method="nearest")
        z_interp[~mask] = filled

    # 平滑化(模様の微細凹凸は残るよう軽め)
    if smooth_sigma > 0:
        k = max(3, int(smooth_sigma * 3) | 1)  # 奇数化
        z_interp = cv2.GaussianBlur(z_interp, (k, k), smooth_sigma)

    # 0..255に正規化
    zmin, zmax = np.nanmin(z_interp), np.nanmax(z_interp)
    if zmax - zmin < 1e-9:
        norm = np.zeros_like(z_interp, dtype=np.uint8)
    else:
        norm = ((z_interp - zmin) / (zmax - zmin) * 255).astype(np.uint8)

    return DepthMap(
        image=norm, raw_z=raw_z, z_interp=z_interp,
        mask=mask, extent=(xmin, xmax, ymin, ymax),
        resolution=resolution,
    )


# ---------------------------------------------------------------------------
# 3. 局所適応的二値化(模様線抽出)
# ---------------------------------------------------------------------------

def local_binarize(depth_img: np.ndarray, mask: np.ndarray,
                   window_size: int = 51, k: float = 0.2) -> np.ndarray:
    """
    Sauvola法による局所適応的二値化。
    縄文土器の押文・縄文は周囲より「凹んで」見えるので、
    そのままの深度では暗い→凹模様。凸線が欲しい場合は反転して再実行できる。

    返り値: 0/255 の uint8。模様(凹凸の線)が白(255)。
    """
    # 高周波成分を強調(背景の大きな曲面を引き算)
    bg = cv2.GaussianBlur(depth_img, (0, 0), sigmaX=window_size * 0.5)
    high = cv2.subtract(depth_img, bg)  # 凸線が明るく残る
    high_abs = np.abs(high.astype(np.int16)) + np.abs(
        cv2.subtract(bg, depth_img).astype(np.int16)
    )
    high_abs = np.clip(high_abs, 0, 255).astype(np.uint8)

    if window_size % 2 == 0:
        window_size += 1
    thresh = threshold_sauvola(high_abs, window_size=window_size, k=k)
    binary = (high_abs > thresh).astype(np.uint8) * 255

    # マスク外は0に
    binary[~mask] = 0

    # 孤立点の除去
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return binary


# ---------------------------------------------------------------------------
# 4. テンプレートマッチング(マルチスケール + 回転)
# ---------------------------------------------------------------------------

def multiscale_rotated_match(binary: np.ndarray, template: np.ndarray,
                              scales=(0.8, 0.9, 1.0, 1.1, 1.2),
                              angles=range(0, 360, 15),
                              threshold: float = 0.55,
                              nms_radius: Optional[int] = None) -> list:
    """
    二値画像 binary 上で、template と類似する領域を
    複数スケール × 複数回転で正規化相関を計算し探す。

    返り値: [(x, y, w, h, score, scale, angle), ...]
    """
    th, tw = template.shape
    if nms_radius is None:
        nms_radius = max(th, tw) // 2

    detections = []  # (score, x, y, w, h, scale, angle)
    bin_f = binary.astype(np.float32)

    for scale in scales:
        sh, sw = int(th * scale), int(tw * scale)
        if sh < 8 or sw < 8 or sh >= binary.shape[0] or sw >= binary.shape[1]:
            continue
        tmpl_s = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA)

        for angle in angles:
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw = int(sh * sin + sw * cos)
            nh = int(sh * cos + sw * sin)
            M[0, 2] += nw / 2 - sw / 2
            M[1, 2] += nh / 2 - sh / 2
            tmpl_r = cv2.warpAffine(tmpl_s, M, (nw, nh), borderValue=0)

            if nh >= binary.shape[0] or nw >= binary.shape[1]:
                continue

            res = cv2.matchTemplate(bin_f, tmpl_r.astype(np.float32),
                                     cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for y, x in zip(*loc):
                detections.append((float(res[y, x]), int(x), int(y),
                                    nw, nh, scale, angle))

    # スコア降順 → NMS(Non-Max Suppression)
    detections.sort(reverse=True, key=lambda d: d[0])
    kept = []
    for det in detections:
        score, x, y, w, h, scale, angle = det
        cx, cy = x + w / 2, y + h / 2
        ok = True
        for k_det in kept:
            kx, ky = k_det[1] + k_det[3] / 2, k_det[2] + k_det[4] / 2
            if (cx - kx) ** 2 + (cy - ky) ** 2 < nms_radius ** 2:
                ok = False
                break
        if ok:
            kept.append(det)

    print(f"[template] {len(detections)} raw → {len(kept)} after NMS "
          f"(threshold={threshold})")
    return kept


# ---------------------------------------------------------------------------
# 5. ORB特徴量による回転・スケール不変な類似領域検出
# ---------------------------------------------------------------------------

def orb_similar_regions(binary: np.ndarray, template: np.ndarray,
                        min_matches: int = 6,
                        cluster_eps: float = 30.0) -> list:
    """
    ORB特徴で template に似た領域を探し、特徴点の塊(クラスタ)として返す。
    返り値: [(cx, cy, radius, n_matches), ...]
    """
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(binary, None)
    if des1 is None or des2 is None or len(kp2) < min_matches:
        print("[orb] insufficient features")
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(f"[orb] {len(good)} good matches")

    if len(good) < min_matches:
        return []

    # マッチした kp2 側の座標で簡易クラスタリング(DBSCAN相当の凝集)
    pts = np.array([kp2[m.trainIdx].pt for m in good])
    clusters = []  # list of [idx, ...]
    used = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        seed = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(pts)):
                if used[j]:
                    continue
                if any(np.linalg.norm(pts[j] - pts[k]) < cluster_eps for k in seed):
                    seed.append(j)
                    used[j] = True
                    changed = True
        clusters.append(seed)

    regions = []
    for cl in clusters:
        if len(cl) < min_matches:
            continue
        cpts = pts[cl]
        cx, cy = cpts.mean(axis=0)
        radius = max(cluster_eps, cpts.std(axis=0).mean() * 2 + 10)
        regions.append((float(cx), float(cy), float(radius), len(cl)))
    regions.sort(key=lambda r: -r[3])
    print(f"[orb] {len(regions)} clusters with >={min_matches} matches")
    return regions


# ---------------------------------------------------------------------------
# 6. 可視化
# ---------------------------------------------------------------------------

def visualize_results(depth: DepthMap, binary: np.ndarray,
                       template_roi: tuple,
                       template_matches: list, orb_regions: list,
                       output_path: str):
    """3枚並びの結果図を保存する。"""
    tx, ty, tw, th = template_roi
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(depth.image, cmap="gray")
    axes[0].add_patch(Rectangle((tx, ty), tw, th, fill=False,
                                  edgecolor="cyan", linewidth=2))
    axes[0].set_title("Depth map (Z-projection)\ncyan = template ROI")
    axes[0].axis("off")

    # 二値化 + テンプレートマッチ結果
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    for score, x, y, w, h, scale, angle in template_matches:
        color = (255, int(255 * (1 - score)), 0)  # スコア高→赤に近い
        cv2.rectangle(binary_rgb, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(binary_rgb, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2)
    axes[1].imshow(binary_rgb)
    axes[1].set_title(f"Binary + template matches ({len(template_matches)})")
    axes[1].axis("off")

    # ORB結果
    orb_rgb = cv2.cvtColor(depth.image, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(orb_rgb, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2)
    for cx, cy, r, n in orb_regions:
        cv2.circle(orb_rgb, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.putText(orb_rgb, f"{n}", (int(cx), int(cy)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    axes[2].imshow(orb_rgb)
    axes[2].set_title(f"ORB feature clusters ({len(orb_regions)})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] saved {output_path}")


# ---------------------------------------------------------------------------
# 7. ROI選択(GUIなし → 自動候補 or 引数指定)
# ---------------------------------------------------------------------------

def suggest_roi(binary: np.ndarray, size: int = 80) -> tuple:
    """
    自動でテンプレート候補を選ぶ:模様密度が最も高い size×size 領域。
    """
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    density = cv2.filter2D(binary.astype(np.float32) / 255, -1, kernel)
    # 端は避ける
    pad = size // 2
    density[:pad] = 0; density[-pad:] = 0
    density[:, :pad] = 0; density[:, -pad:] = 0
    cy, cx = np.unravel_index(np.argmax(density), density.shape)
    x, y = max(0, cx - size // 2), max(0, cy - size // 2)
    print(f"[roi] auto-suggested ROI at ({x},{y}) size={size}")
    return (x, y, size, size)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_pipeline(ply_path: str, output_dir: str,
                 resolution: float = 0.5,
                 window_size: int = 51, sauvola_k: float = 0.2,
                 template_size: int = 80,
                 roi: Optional[tuple] = None,
                 match_threshold: float = 0.55):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(ply_path))[0]

    pts = load_ply_points(ply_path)
    aligned = align_to_principal_plane(pts)
    depth = points_to_depth_map(aligned, resolution=resolution)

    cv2.imwrite(os.path.join(output_dir, f"{base}_01_depth.png"), depth.image)

    binary = local_binarize(depth.image, depth.mask,
                              window_size=window_size, k=sauvola_k)
    cv2.imwrite(os.path.join(output_dir, f"{base}_02_binary.png"), binary)

    if roi is None:
        roi = suggest_roi(binary, size=template_size)
    rx, ry, rw, rh = roi
    template = binary[ry:ry + rh, rx:rx + rw]
    cv2.imwrite(os.path.join(output_dir, f"{base}_03_template.png"), template)

    matches = multiscale_rotated_match(binary, template,
                                        threshold=match_threshold)
    orb_regs = orb_similar_regions(binary, template)

    out_viz = os.path.join(output_dir, f"{base}_04_result.png")
    visualize_results(depth, binary, roi, matches, orb_regs, out_viz)

    # CSV出力
    csv_path = os.path.join(output_dir, f"{base}_matches.csv")
    with open(csv_path, "w") as f:
        f.write("type,x,y,w_or_r,h_or_n,score_or_matches,scale,angle\n")
        for s, x, y, w, h, sc, ang in matches:
            f.write(f"template,{x},{y},{w},{h},{s:.3f},{sc},{ang}\n")
        for cx, cy, r, n in orb_regs:
            f.write(f"orb,{int(cx)},{int(cy)},{int(r)},{n},,,\n")
    print(f"[csv] saved {csv_path}")

    return {
        "depth": depth, "binary": binary, "template": template,
        "matches": matches, "orb": orb_regs,
        "output_viz": out_viz, "csv": csv_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply", help="入力PLYファイル(点群)")
    p.add_argument("--output_dir", default="./results", help="出力ディレクトリ")
    p.add_argument("--resolution", type=float, default=0.5,
                    help="深度マップ1ピクセルあたりのワールド単位(小さいほど高精細)")
    p.add_argument("--window_size", type=int, default=51,
                    help="Sauvola二値化の窓サイズ(奇数)")
    p.add_argument("--sauvola_k", type=float, default=0.2,
                    help="Sauvola二値化のkパラメータ")
    p.add_argument("--template_size", type=int, default=80,
                    help="自動ROIのサイズ(ピクセル)")
    p.add_argument("--roi", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="テンプレートROIを手動指定(指定なければ自動)")
    p.add_argument("--match_threshold", type=float, default=0.55,
                    help="正規化相関のしきい値(0..1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.ply):
        print(f"PLY not found: {args.ply}", file=sys.stderr)
        sys.exit(1)
    run_pipeline(
        ply_path=args.ply, output_dir=args.output_dir,
        resolution=args.resolution,
        window_size=args.window_size, sauvola_k=args.sauvola_k,
        template_size=args.template_size, roi=args.roi,
        match_threshold=args.match_threshold,
    )
