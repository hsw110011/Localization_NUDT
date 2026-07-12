#!/usr/bin/env python3
"""检查 dem_loc 粒子滤波每个粒子的 score/J_total/weight 分布。"""

import argparse
import csv
import math
import os


DEFAULT_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "particle_filter_debug",
        "pf_particles.csv",
    )
)


def _to_float(value, default=float("nan")):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _to_int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_particles(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return rows
        has_header = first and first[0] == "timestamp"
        if has_header:
            fieldnames = first
            required = {"timestamp", "frame_id", "particle_id", "x", "y", "yaw", "score", "weight"}
            missing = sorted(required - set(fieldnames))
            if missing:
                raise SystemExit(
                    "CSV 字段不对，当前脚本需要 pf_particles.csv。缺少字段: {}".format(
                        ", ".join(missing)
                    )
                )
            dict_reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in dict_reader:
                rows.append(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "frame_id": _to_int(row.get("frame_id", -1)),
                        "particle_id": _to_int(row.get("particle_id", -1)),
                        "x": _to_float(row.get("x")),
                        "y": _to_float(row.get("y")),
                        "yaw": _to_float(row.get("yaw")),
                        "score": _to_float(row.get("score")),
                        "j_total": _to_float(row.get("j_total")),
                        "weight": _to_float(row.get("weight"), 0.0),
                    }
                )
        else:
            for raw in [first] + list(reader):
                if len(raw) < 8:
                    continue
                rows.append(
                    {
                        "timestamp": raw[0],
                        "frame_id": _to_int(raw[1]),
                        "particle_id": _to_int(raw[2]),
                        "x": _to_float(raw[3]),
                        "y": _to_float(raw[4]),
                        "yaw": _to_float(raw[5]),
                        "score": _to_float(raw[6]),
                        "j_total": float("nan"),
                        "weight": _to_float(raw[7], 0.0),
                    }
                )
    return rows


def _group_frame_blocks(rows):
    """按文件写入顺序分组，避免多次运行追加时相同 frame_id 被混在一起。"""
    blocks = []
    current_rows = []
    current_frame_id = None

    for row in rows:
        frame_id = row["frame_id"]
        if frame_id < 0:
            continue
        if current_rows and frame_id != current_frame_id:
            blocks.append({"frame_id": current_frame_id, "rows": current_rows})
            current_rows = []
        current_frame_id = frame_id
        current_rows.append(row)

    if current_rows:
        blocks.append({"frame_id": current_frame_id, "rows": current_rows})
    return blocks


def _finite(values):
    return [v for v in values if math.isfinite(v)]


def _stats(values):
    vals = _finite(values)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return min(vals), sum(vals) / len(vals), max(vals)


def _weighted_mean(rows, field):
    total_w = sum(max(0.0, r["weight"]) for r in rows)
    if total_w <= 0.0:
        return float("nan")
    return sum(r[field] * max(0.0, r["weight"]) for r in rows) / total_w


def _std(values):
    vals = _finite(values)
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _fmt(value, precision=6):
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _summarize_frame(frame_rows, top_k, block_index, block_count):
    weights = [max(0.0, r["weight"]) for r in frame_rows]
    weight_sum = sum(weights)
    neff = 0.0 if weight_sum <= 0.0 else 1.0 / sum((w / weight_sum) ** 2 for w in weights)
    score_min, score_mean, score_max = _stats([r["score"] for r in frame_rows])
    j_min, j_mean, j_max = _stats([r["j_total"] for r in frame_rows])
    x_mean = _weighted_mean(frame_rows, "x")
    y_mean = _weighted_mean(frame_rows, "y")
    x_std = _std([r["x"] for r in frame_rows])
    y_std = _std([r["y"] for r in frame_rows])

    print(f"frame_block: {block_index + 1} / {block_count}")
    print(f"frame_id: {frame_rows[0]['frame_id']}")
    print(f"particles: {len(frame_rows)}")
    print(f"neff: {_fmt(neff, 3)} / {len(frame_rows)}")
    print(f"weight min/mean/max: {_fmt(min(weights), 8)} / {_fmt(weight_sum / len(weights), 8)} / {_fmt(max(weights), 8)}")
    print(f"score min/mean/max: {_fmt(score_min)} / {_fmt(score_mean)} / {_fmt(score_max)}")
    print(f"j_total min/mean/max: {_fmt(j_min)} / {_fmt(j_mean)} / {_fmt(j_max)}")
    print(f"weighted_xy: ({_fmt(x_mean, 3)}, {_fmt(y_mean, 3)})")
    print(f"unweighted_xy_std: x={_fmt(x_std, 3)} m, y={_fmt(y_std, 3)} m")

    print("\ntop particles by weight:")
    ranked = sorted(frame_rows, key=lambda r: r["weight"], reverse=True)
    print("particle_id,x,y,yaw,score,j_total,weight")
    for row in ranked[: max(1, top_k)]:
        print(
            "{pid},{x},{y},{yaw},{score},{j_total},{weight}".format(
                pid=row["particle_id"],
                x=_fmt(row["x"], 4),
                y=_fmt(row["y"], 4),
                yaw=_fmt(row["yaw"], 6),
                score=_fmt(row["score"], 8),
                j_total=_fmt(row["j_total"], 8),
                weight=_fmt(row["weight"], 10),
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="pf_particles.csv 路径")
    parser.add_argument("--frame", default="latest", help="帧号；默认 latest")
    parser.add_argument("--top-k", type=int, default=10, help="打印权重最高的粒子数")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"找不到粒子 CSV: {args.csv}")

    rows = _read_particles(args.csv)
    if not rows:
        raise SystemExit(
            "CSV 里没有粒子行，只有表头或空文件。请重新启动 dem_loc 节点并确认 "
            "pf_save_particle_csv/save_particle_csv 已生效。"
        )

    frame_blocks = _group_frame_blocks(rows)
    if not frame_blocks:
        raise SystemExit("CSV 中没有有效 frame_id")

    if args.frame == "latest":
        block_index = len(frame_blocks) - 1
    else:
        frame_id = _to_int(args.frame)
        matches = [
            idx for idx, block in enumerate(frame_blocks) if block["frame_id"] == frame_id
        ]
        if not matches:
            raise SystemExit(f"CSV 中没有 frame_id={args.frame}")
        block_index = matches[-1]

    print(f"csv: {args.csv}")
    print(f"frame_blocks: {len(frame_blocks)}, particle_rows: {len(rows)}")
    _summarize_frame(
        frame_blocks[block_index]["rows"],
        args.top_k,
        block_index,
        len(frame_blocks),
    )


if __name__ == "__main__":
    main()
