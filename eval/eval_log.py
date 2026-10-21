#!/usr/bin/env python3
"""detection_log.txt から mAP_frame と mAP_display を計算する。

IoU 0.5 を主指標、0.75 を参考値として両方出す。
--iou で単一の IoU だけ計算することもできる。

使い方:
  python eval_log.py detection_log.txt                    # IoU 0.5 (主) + 0.75 (参考)
  python eval_log.py detection_log.txt --iou 0.75         # IoU 0.75 のみ
  python eval_log.py detection_log.txt --mode frame       # mAP_frame のみ
  python eval_log.py detection_log.txt --scale-from 416x208  # 座標を 512x256 にスケール
  python eval_log.py --gen-dummy                          # テスト用ダミーログ生成

前提:
  - ground-truth/ に GT ファイル（frame_XXXXX.txt）が揃っていること
    （--gt-dir で指定可。なければダミー GT を生成してテスト可能）
  - eval/mAP/ に Cartucho mAP ツールがあること
"""

import argparse
import os
import random
import shutil
import subprocess
import sys

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------- ログの読み込み ----------

def parse_header(line):
    """ヘッダ行からパラメータを dict で返す。"""
    parts = line.lstrip("# ").split()
    params = {}
    i = 0
    while i < len(parts) - 1:
        params[parts[i]] = parts[i + 1]
        i += 2
    return params


def parse_log(path):
    """detection_log.txt を読み込み、(header, records) を返す。

    record: {"frame_id": int, "shown": bool, "dets": [{"cls": int, "score": float,
             "x1": float, "y1": float, "x2": float, "y2": float}, ...]}
    records は Show に届いた順。
    """
    header = {}
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = parse_header(line)
                continue
            parts = line.split()
            frame_id = int(parts[0])
            shown = parts[1] == "1"
            n_dets = int(parts[2])
            dets = []
            idx = 3
            for _ in range(n_dets):
                dets.append({
                    "cls": int(parts[idx]),
                    "score": float(parts[idx + 1]),
                    "x1": float(parts[idx + 2]),
                    "y1": float(parts[idx + 3]),
                    "x2": float(parts[idx + 4]),
                    "y2": float(parts[idx + 5]),
                })
                idx += 6
            records.append({"frame_id": frame_id, "shown": shown, "dets": dets})
    return header, records


def check_header(header, cfg):
    """ヘッダの CONF, NMS_THRESHOLD が既定値と一致するか検査する。"""
    expected_conf = str(cfg["student_defaults"]["CONF"])
    expected_nms = str(cfg["student_defaults"]["NMS_THRESHOLD"])
    warnings = []
    if "CONF" in header and header["CONF"] != expected_conf:
        warnings.append(
            f"WARNING: CONF={header['CONF']}（既定値 {expected_conf} と異なります）"
        )
    if "NMS_THRESHOLD" in header and header["NMS_THRESHOLD"] != expected_nms:
        warnings.append(
            f"WARNING: NMS_THRESHOLD={header['NMS_THRESHOLD']}"
            f"（既定値 {expected_nms} と異なります）"
        )
    return warnings


def validate_log(records, n_frames):
    """ログの整合性を検査する。

    - frame_id 0〜N-1 がちょうど 1 回ずつ記録されているか
    - shown=1 のレコードが 1 件以上あるか
    エラーがあれば表示して sys.exit する。
    """
    from collections import Counter
    fid_counts = Counter(r["frame_id"] for r in records)
    errors = []

    # 抜けの検査
    expected = set(range(n_frames))
    recorded = set(fid_counts.keys())
    missing = sorted(expected - recorded)
    if missing:
        if len(missing) <= 20:
            errors.append(f"ERROR: {len(missing)} フレームが記録されていません: {missing}")
        else:
            errors.append(
                f"ERROR: {len(missing)} フレームが記録されていません "
                f"(先頭: {missing[:10]}...)"
            )

    # 重複の検査
    duplicates = {fid: cnt for fid, cnt in fid_counts.items() if cnt > 1}
    if duplicates:
        dup_list = sorted(duplicates.items())
        if len(dup_list) <= 20:
            errors.append(f"ERROR: {len(dup_list)} フレームが複数回記録されています: {dup_list}")
        else:
            errors.append(
                f"ERROR: {len(dup_list)} フレームが複数回記録されています "
                f"(先頭: {dup_list[:10]}...)"
            )

    # 範囲外の検査
    out_of_range = sorted(fid for fid in fid_counts if fid < 0 or fid >= n_frames)
    if out_of_range:
        errors.append(f"ERROR: 範囲外の frame_id があります: {out_of_range[:10]}")

    # shown=1 の検査
    n_shown = sum(1 for r in records if r["shown"])
    if n_shown == 0:
        errors.append("ERROR: shown=1 のレコードが 1 件もありません")

    if errors:
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ログ検査: OK ({len(records)} レコード, shown={n_shown})")


# ---------- 座標スケール補正 ----------

def scale_coordinates(records, src_w, src_h, target_w, target_h):
    """ログの座標を (src_w, src_h) から (target_w, target_h) にスケールする。"""
    sx = target_w / src_w
    sy = target_h / src_h
    for rec in records:
        for d in rec["dets"]:
            d["x1"] *= sx
            d["x2"] *= sx
            d["y1"] *= sy
            d["y2"] *= sy
    return records


def auto_scale_if_needed(records, target_w, target_h):
    """座標の最大値から元の解像度を推定し、必要ならスケール補正する。

    x 座標の最大値で判定する（y はマスク領域の手前で止まるため最大値が
    画像高さに達しないことがある）。x の最大値 + 1 ≈ 元の幅。
    GT の幅と 5% 以上ずれていれば、幅から元の解像度を推定してスケール補正する。
    元の動画はアスペクト比を維持して縮小されている前提で、
    y のスケールも x と同じ倍率を使う。
    """
    all_x = []
    for rec in records:
        for d in rec["dets"]:
            all_x.extend([d["x1"], d["x2"]])

    if not all_x:
        return records, None

    max_x = max(all_x)
    est_w = round(max_x + 1)

    # GT 幅と近ければ補正不要
    if abs(est_w - target_w) <= target_w * 0.05:
        return records, None

    # 幅ベースでスケールを決定（アスペクト比維持の前提）
    scale = target_w / est_w
    est_h = round(target_h / scale)
    msg = (f"座標スケール補正（自動）: {est_w}x{est_h} → {target_w}x{target_h}"
           f" (x*{scale:.4f}, y*{scale:.4f})")
    records = scale_coordinates(records, est_w, est_h, target_w, target_h)
    return records, msg


# ---------- マスク処理 ----------

def is_masked(det, mask_cfg):
    """検出の中心がマスク領域内ならTrue。"""
    if mask_cfg["type"] == "hline":
        cy = (det["y1"] + det["y2"]) / 2
        return cy >= mask_cfg["y"]
    return False


def filter_masked(dets, mask_cfg):
    """マスク領域内の検出を除外する。"""
    return [d for d in dets if not is_masked(d, mask_cfg)]


# ---------- mAP_frame ----------

def build_frame_results(records, n_frames):
    """各 frame_id の検出結果を集める。全フレーム分の dict を返す。"""
    by_frame = {i: [] for i in range(n_frames)}
    seen_frames = set()
    for rec in records:
        fid = rec["frame_id"]
        if 0 <= fid < n_frames:
            by_frame[fid] = rec["dets"]
            seen_frames.add(fid)
    missing = n_frames - len(seen_frames)
    if missing > 0:
        print(f"WARNING: {missing} フレームがログにありません", file=sys.stderr)
    return by_frame


# ---------- mAP_display ----------

def build_held_results(records, n_frames):
    """shown=1 のレコードから hold 処理を行い、全フレームの検出結果を返す。

    CLAUDE.md の参照実装に準拠:
    - shown=1 を表示順に読む
    - 逆行（それまでの最大 frame_id 以下）は無効
    - 各フレーム j に、有効表示のうち frame_id <= j で最大のものの検出結果を割り当て
    """
    display_log = [
        (rec["frame_id"], rec["dets"])
        for rec in records if rec["shown"]
    ]

    valid = {}
    last = -1
    n_inversions = 0
    for fid, dets in display_log:
        if fid > last:
            valid[fid] = dets
            last = fid
        else:
            n_inversions += 1

    held = {}
    cur = []
    for j in range(n_frames):
        if j in valid:
            cur = valid[j]
        held[j] = cur

    stats = {
        "n_shown": len(display_log),
        "n_valid": len(valid),
        "n_inversions": n_inversions,
    }
    # 平均保持長: 有効表示間の平均フレーム間隔
    valid_ids = sorted(valid.keys())
    if len(valid_ids) > 1:
        gaps = [valid_ids[i + 1] - valid_ids[i] for i in range(len(valid_ids) - 1)]
        stats["avg_hold"] = sum(gaps) / len(gaps)
    else:
        stats["avg_hold"] = float(n_frames)

    return held, stats


# ---------- detection-results の書き出し ----------

def write_detection_results(results, outdir, type_to_class, mask_cfg, prefix, digits):
    """results: {frame_id: [det, ...]} を Cartucho 形式で書き出す。"""
    os.makedirs(outdir, exist_ok=True)
    for fid, dets in sorted(results.items()):
        fname = f"{prefix}{fid:0{digits}d}.txt"
        path = os.path.join(outdir, fname)
        filtered = filter_masked(dets, mask_cfg) if mask_cfg else dets
        with open(path, "w") as f:
            for d in filtered:
                cls_name = type_to_class[d["cls"]]
                f.write(
                    f"{cls_name} {d['score']:.6f} "
                    f"{d['x1']:.1f} {d['y1']:.1f} "
                    f"{d['x2']:.1f} {d['y2']:.1f}\n"
                )


# ---------- Cartucho mAP ツールの呼び出し ----------

def run_map(map_dir, gt_dir, det_dir, iou=0.5):
    """Cartucho mAP ツールを実行し、結果を返す。

    gt_dir, det_dir のファイルを mAP/input/ にコピーして実行する。
    iou: IoU しきい値 (Cartucho の MINOVERLAP)。
    """
    gt_input = os.path.join(map_dir, "input", "ground-truth")
    dr_input = os.path.join(map_dir, "input", "detection-results")

    # 既存ファイルをクリア
    for d in [gt_input, dr_input]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # GT ファイルをコピー
    for fname in os.listdir(gt_dir):
        if fname.endswith(".txt"):
            shutil.copy2(os.path.join(gt_dir, fname), gt_input)

    # detection-results をコピー
    for fname in os.listdir(det_dir):
        if fname.endswith(".txt"):
            shutil.copy2(os.path.join(det_dir, fname), dr_input)

    # output ディレクトリをクリア
    output_dir = os.path.join(map_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # main.py を実行（IoU は環境変数 MINOVERLAP で渡す）
    env = os.environ.copy()
    if iou != 0.5:
        env["MINOVERLAP"] = str(iou)
    result = subprocess.run(
        [sys.executable, "main.py", "-na", "-np", "-q"],
        cwd=map_dir,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"ERROR: mAP ツールがエラーで終了しました:\n{result.stderr}", file=sys.stderr)
        return None

    return parse_map_output(map_dir)


def parse_map_output(map_dir):
    """Cartucho mAP ツールの output/output.txt を読んで結果を返す。"""
    results_file = os.path.join(map_dir, "output", "output.txt")
    if not os.path.exists(results_file):
        return None

    results = {"per_class": {}, "mAP": 0.0}
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            # クラス別 AP の行: "0.00% = bicycle AP" 形式
            if " AP" in line and "=" in line:
                parts = line.split()
                # "X.XX% = classname AP" → parts[0]=percent, parts[2]=classname
                if len(parts) >= 4 and parts[1] == "=":
                    ap_str = parts[0].rstrip("%")
                    cls_name = parts[2]
                    try:
                        results["per_class"][cls_name] = float(ap_str)
                    except ValueError:
                        pass
            # mAP の行: "mAP = 0.00%"
            if line.startswith("mAP"):
                parts = line.split()
                for p in parts:
                    p = p.rstrip("%")
                    try:
                        results["mAP"] = float(p)
                        break
                    except ValueError:
                        pass
    return results


# ---------- 評価の実行 ----------

def eval_single_iou(records, n_frames, type_to_class, mask_cfg, prefix, digits,
                    map_dir, gt_dir, outdir, iou, mode):
    """指定 IoU で mAP_frame / mAP_display を計算し、結果 dict を返す。"""
    result = {}

    if mode in ("both", "frame"):
        frame_results = build_frame_results(records, n_frames)
        frame_det_dir = os.path.join(outdir, "det_frame")
        write_detection_results(
            frame_results, frame_det_dir, type_to_class, mask_cfg, prefix, digits,
        )
        r = run_map(map_dir, gt_dir, frame_det_dir, iou=iou)
        if r:
            result["mAP_frame"] = r

    if mode in ("both", "display"):
        held_results, stats = build_held_results(records, n_frames)
        display_det_dir = os.path.join(outdir, "det_display")
        write_detection_results(
            held_results, display_det_dir, type_to_class, mask_cfg, prefix, digits,
        )
        r = run_map(map_dir, gt_dir, display_det_dir, iou=iou)
        if r:
            result["mAP_display"] = r
        result["display_stats"] = stats

    return result


def plot_ap_dual(results_by_iou, primary_iou, header, logfile="", n_records=0):
    """IoU 0.5 と 0.75 のクラス別 AP 棒グラフを左右に並べて表示・保存する。"""
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib が無いためグラフを表示できません", file=sys.stderr)
        return

    ref_iou = 0.75 if primary_iou == 0.5 else 0.5
    iou_list = [primary_iou, ref_iou]
    iou_labels = {primary_iou: "primary", ref_iou: "ref"}

    # 両 IoU の mAP_display（なければ mAP_frame）を取得
    plot_data = []
    for iou in iou_list:
        r = results_by_iou.get(iou, {})
        if "mAP_display" in r:
            plot_data.append((iou, r["mAP_display"], ""))
        elif "mAP_frame" in r:
            plot_data.append((iou, r["mAP_frame"], " (frame)"))
        else:
            plot_data.append((iou, None, ""))

    if not any(d[1] for d in plot_data):
        return

    # ヘッダ情報
    model = header.get("MODEL", "")
    input_size = header.get("INPUT", "")
    show_every = header.get("SHOW_EVERY", "")
    fps = header.get("FPS", "")
    label = os.path.splitext(os.path.basename(logfile))[0] if logfile else ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for idx, (iou, results, extra) in enumerate(plot_data):
        ax = axes[idx]
        if results is None:
            ax.set_visible(False)
            continue

        per_class = results.get("per_class", {})
        classes = sorted(per_class.keys())
        aps = [per_class.get(c, 0) for c in classes]

        ap_car = per_class.get("car", 0)
        ap_person = per_class.get("person", 0)
        mAP2 = (ap_car + ap_person) / 2.0

        bars = ax.barh(classes, aps, color="royalblue")
        for bar, ap, cls in zip(bars, aps, classes):
            txt = f"{ap:.1f}%"
            if cls == "bicycle":
                txt += "  (n/a)"
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    txt, va="center", fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Average Precision (%)")
        ax.set_title(f"IoU {iou} ({iou_labels[iou]})\nmAP2 = {mAP2:.1f}%  (car+person)/2{extra}",
                     fontsize=10)

    # 上部にヘッダ情報
    info_parts = []
    if model:
        info_parts.append(f"Model: {model}")
    if input_size:
        info_parts.append(f"Input: {input_size}")
    ninfer = header.get("NINFER", "")
    cap = header.get("CAP", "")
    if ninfer:
        info_parts.append(f"NINFER: {ninfer}")
    if cap:
        info_parts.append(f"CAP: {cap}")
    if show_every:
        info_parts.append(f"SE: {show_every}")
    if fps:
        info_parts.append(f"FPS: {fps}")
    if n_records:
        info_parts.append(f"N: {n_records}")
    if label:
        info_parts.append(label)
    fig.suptitle("   ".join(info_parts), fontsize=9, color="gray", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # PNG 保存（ログと同じディレクトリに <ログ名>_map.png）
    if logfile:
        log_dir = os.path.dirname(os.path.abspath(logfile))
        png_name = os.path.splitext(os.path.basename(logfile))[0] + "_map.png"
        png_path = os.path.join(log_dir, png_name)
    else:
        png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "mAP_display.png")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close()
    print(f"\n棒グラフを保存しました: {png_path}")

    # ビューアで開く
    import subprocess as _sp
    import platform
    if platform.system() == "Linux":
        _sp.Popen(["xdg-open", png_path], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    elif platform.system() == "Darwin":
        _sp.Popen(["open", png_path])
    else:
        _sp.Popen(["start", png_path], shell=True)


def _mAP2(r):
    """per_class から mAP2 (car+person)/2 を計算する。"""
    pc = r.get("per_class", {})
    return (pc.get("car", 0) + pc.get("person", 0)) / 2.0


def print_results(results_by_iou, primary_iou, mode):
    """IoU ごとの結果を表示する。primary_iou を主指標として先に出す。"""
    iou_order = [primary_iou] + [k for k in sorted(results_by_iou) if k != primary_iou]

    for iou in iou_order:
        res = results_by_iou.get(iou)
        if not res:
            continue
        tag = "" if iou == primary_iou else " [ref]"

        if mode in ("both", "frame") and "mAP_frame" in res:
            r = res["mAP_frame"]
            print(f"\n=== mAP_frame (IoU={iou}){tag} ===")
            print(f"  mAP2 (car+person)/2 = {_mAP2(r):.2f}%")
            for cls, ap in sorted(r["per_class"].items()):
                print(f"    {cls}: AP = {ap:.2f}%")
            print(f"  mAP (3cls, ref) = {r['mAP']:.2f}%")

        if mode in ("both", "display") and "mAP_display" in res:
            r = res["mAP_display"]
            stats = res.get("display_stats", {})
            print(f"\n=== mAP_display (IoU={iou}){tag} ===")
            if stats:
                print(f"  valid: {stats['n_valid']}, "
                      f"inversions: {stats['n_inversions']}, "
                      f"avg hold: {stats['avg_hold']:.1f} frames")
            print(f"  mAP2 (car+person)/2 = {_mAP2(r):.2f}%")
            for cls, ap in sorted(r["per_class"].items()):
                print(f"    {cls}: AP = {ap:.2f}%")
            print(f"  mAP (3cls, ref) = {r['mAP']:.2f}%")

    # まとめ (mAP2 ベース)
    print(f"\n=== summary (mAP2) ===")
    header = "  {:>12s}".format("")
    for iou in iou_order:
        tag = "" if iou == primary_iou else " [ref]"
        header += f"  IoU={iou}{tag:>5s}"
    print(header)

    for key, label in [("mAP_frame", "mAP2_frame"), ("mAP_display", "mAP2_display")]:
        if not any(key in results_by_iou.get(i, {}) for i in iou_order):
            continue
        line = f"  {label:>12s}"
        for iou in iou_order:
            r = results_by_iou.get(iou, {}).get(key)
            if r:
                line += f"  {_mAP2(r):>11.2f}%"
            else:
                line += f"  {'---':>12s}"
        print(line)

    # loss (display - frame)
    if mode == "both":
        line = f"  {'loss':>12s}"
        for iou in iou_order:
            res = results_by_iou.get(iou, {})
            if "mAP_frame" in res and "mAP_display" in res:
                loss = _mAP2(res["mAP_display"]) - _mAP2(res["mAP_frame"])
                line += f"  {loss:>+11.2f}%"
            else:
                line += f"  {'---':>12s}"
        print(line)


# ---------- ダミーデータ生成 ----------

def gen_dummy_log(outpath, n_frames, show_every=2, n_infer=2):
    """テスト用のダミー detection_log.txt を生成する。"""
    random.seed(42)
    header = (
        f"# CONF 0.3 NMS_THRESHOLD 0.45 NINFER {n_infer} "
        f"CAP 16 SHOW_EVERY {show_every} Lbox_on 0 PROFILE 0"
    )

    frame_order = list(range(n_frames))
    for i in range(1, len(frame_order)):
        if random.random() < 0.03:
            frame_order[i - 1], frame_order[i] = frame_order[i], frame_order[i - 1]

    lines = [header]
    for fid in frame_order:
        shown = 1 if fid % show_every == 0 else 0
        n_dets = random.randint(0, 5)
        dets_str = []
        for _ in range(n_dets):
            cls = random.randint(0, 2)
            score = round(random.uniform(0.3, 0.99), 4)
            x1 = random.uniform(0, 400)
            y1 = random.uniform(0, 150)
            w = random.uniform(20, 100)
            h = random.uniform(15, 80)
            x2 = min(x1 + w, 511)
            y2 = min(y1 + h, 189)
            dets_str.append(f"{cls} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
        det_part = " ".join(dets_str)
        lines.append(f"{fid} {shown} {n_dets} {det_part}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"ダミーログ生成: {outpath} ({n_frames} フレーム, SHOW_EVERY={show_every})")


def gen_dummy_gt(outdir, n_frames, prefix, digits):
    """テスト用のダミー GT を生成する。"""
    random.seed(123)
    os.makedirs(outdir, exist_ok=True)
    classes = ["car", "person", "bicycle"]
    for i in range(n_frames):
        fname = f"{prefix}{i:0{digits}d}.txt"
        path = os.path.join(outdir, fname)
        n_objs = random.randint(0, 4)
        with open(path, "w") as f:
            for _ in range(n_objs):
                cls = random.choice(classes)
                x1 = random.uniform(0, 400)
                y1 = random.uniform(0, 150)
                w = random.uniform(20, 100)
                h = random.uniform(15, 80)
                x2 = min(x1 + w, 511)
                y2 = min(y1 + h, 189)
                f.write(f"{cls} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
    print(f"ダミー GT 生成: {outdir}/ ({n_frames} ファイル)")


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(
        description="detection_log.txt から mAP_frame と mAP_display を計算する"
    )
    parser.add_argument("logfile", nargs="?", help="detection_log.txt のパス")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
        help="設定ファイル (default: ../config.yaml)",
    )
    parser.add_argument(
        "--mode", choices=["both", "frame", "display"], default="both",
        help="計算する mAP の種類 (default: both)",
    )
    parser.add_argument("--gt-dir", default=None, help="GT ディレクトリ (default: ../ground-truth)")
    parser.add_argument("--map-dir", default=None, help="Cartucho mAP ツールのディレクトリ")
    parser.add_argument("--outdir", default=None, help="detection-results の出力先 (default: ./work)")
    parser.add_argument(
        "--iou", type=float, default=None,
        help="単一の IoU で計算する場合に指定。省略時は 0.5 (主) と 0.75 (参考) の両方を出す",
    )
    parser.add_argument(
        "--scale-from", default=None, metavar="WxH",
        help="ログの座標が WxH 空間の場合、GT の座標空間にスケール補正する "
             "(例: --scale-from 416x208)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="棒グラフのポップアップ表示を抑制する",
    )
    parser.add_argument(
        "--gen-dummy", action="store_true",
        help="テスト用ダミーログと GT を生成して評価する",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    n_frames = cfg["video"]["n_frames"]
    prefix = "frame_"
    digits = cfg["video"]["frame_pad"]
    mask_cfg = cfg["mask"]
    type_to_class = {int(k): v for k, v in cfg["classes"]["type_to_name"].items()}

    primary_iou = cfg.get("eval", {}).get("iou_threshold", 0.75)
    gt_dir = args.gt_dir or os.path.join(base, cfg["paths"]["ground_truth"])
    map_dir = args.map_dir or os.path.join(base, cfg["paths"]["map_tool"])
    outdir = args.outdir or os.path.join(eval_dir, "work")

    # --iou 指定時はその値だけ、省略時は主 (0.5) + 参考 (0.75)
    if args.iou is not None:
        iou_list = [args.iou]
        primary_iou = args.iou
    else:
        ref_iou = 0.5 if primary_iou != 0.5 else 0.75
        iou_list = [primary_iou, ref_iou]

    # ダミーモード
    if args.gen_dummy:
        os.makedirs(outdir, exist_ok=True)
        dummy_log = os.path.join(outdir, "dummy_log.txt")
        dummy_gt = os.path.join(outdir, "dummy_gt")
        gen_dummy_log(dummy_log, n_frames)
        gen_dummy_gt(dummy_gt, n_frames, prefix, digits)
        args.logfile = dummy_log
        gt_dir = dummy_gt
        print()

    if not args.logfile:
        parser.error("logfile が必要です（または --gen-dummy を指定）")

    if not os.path.exists(args.logfile):
        print(f"ERROR: ログファイルが見つかりません: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(gt_dir):
        print(f"ERROR: GT ディレクトリが見つかりません: {gt_dir}", file=sys.stderr)
        print("  --gen-dummy でダミーデータを使ってテストできます。", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(os.path.join(map_dir, "main.py")):
        print(f"ERROR: Cartucho mAP ツールが見つかりません: {map_dir}", file=sys.stderr)
        sys.exit(1)

    # ログ読み込み
    header, records = parse_log(args.logfile)
    print(f"ログ: {args.logfile}")
    print(f"  レコード数: {len(records)}, 期待フレーム数: {n_frames}")

    # 座標スケール補正
    target_w = cfg["video"]["width"]
    target_h = cfg["video"]["height"]
    if args.scale_from:
        m = args.scale_from.lower().split("x")
        if len(m) != 2:
            print(f"ERROR: --scale-from の形式が不正です: {args.scale_from} (WxH で指定)",
                  file=sys.stderr)
            sys.exit(1)
        src_w, src_h = float(m[0]), float(m[1])
        records = scale_coordinates(records, src_w, src_h, target_w, target_h)
        print(f"  座標スケール補正: {int(src_w)}x{int(src_h)} → {target_w}x{target_h}"
              f" (x*{target_w/src_w:.4f}, y*{target_h/src_h:.4f})")
    else:
        records, msg = auto_scale_if_needed(records, target_w, target_h)
        if msg:
            print(f"  {msg}")

    warnings = check_header(header, cfg)
    for w in warnings:
        print(f"  {w}", file=sys.stderr)

    # ログの整合性検査
    validate_log(records, n_frames)

    iou_str = ", ".join(f"{i}" + (" [主]" if i == primary_iou else " [参考]")
                        for i in iou_list)
    print(f"  IoU: {iou_str}")

    # 各 IoU で評価
    results_by_iou = {}
    for iou in iou_list:
        print(f"\n--- IoU={iou} を計算中 ---")
        results_by_iou[iou] = eval_single_iou(
            records, n_frames, type_to_class, mask_cfg, prefix, digits,
            map_dir, gt_dir, outdir, iou, args.mode,
        )

    # 結果表示
    print_results(results_by_iou, primary_iou, args.mode)

    # 両 IoU の mAP_display の棒グラフをポップアップ表示
    if not args.no_plot:
        plot_ap_dual(results_by_iou, primary_iou, header,
                     logfile=args.logfile, n_records=len(records))



if __name__ == "__main__":
    main()
