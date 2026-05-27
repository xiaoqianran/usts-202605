#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "batch_size",
    "mu",
    "unlabeled_batch_size",
    "lr",
    "seed",
    "epochs",
    "best_test_acc",
    "best_epoch",
    "time_seconds",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def collect_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        args_path = run_dir / "args.json"
        metrics_path = run_dir / "metrics.json"
        if not args_path.exists() or not metrics_path.exists():
            continue
        args = load_json(args_path)
        metrics = load_json(metrics_path)
        if args.get("mode") != "fixmatch" or metrics.get("method") != "fixmatch":
            continue
        batch_size = int(args["batch_size"])
        mu = int(args["mu"])
        rows.append(
            {
                "run": run_dir.name,
                "batch_size": batch_size,
                "mu": mu,
                "unlabeled_batch_size": batch_size * mu,
                "lr": args["lr"],
                "seed": args["seed"],
                "epochs": args["epochs"],
                "best_test_acc": metrics["best_test_acc"],
                "best_epoch": metrics["best_epoch"],
                "time_seconds": metrics.get("time_seconds", ""),
            }
        )
    return sorted(rows, key=lambda row: (row["batch_size"], row["mu"], row["lr"]))


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# FixMatch batch/mu 网格实验汇总",
        "",
        "| batch size | μ | 无标签 batch size | 学习率 | 最优测试精度 | 最优 epoch | 用时(s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {batch_size} | {mu} | {unlabeled_batch_size} | {lr} | {acc} | {best_epoch} | {time_seconds} |".format(
                **row,
                acc=pct(float(row["best_test_acc"])),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FixMatch batch/mu grid results.")
    parser.add_argument("--runs", required=True, help="Directory containing fixmatch_b*_mu* run folders.")
    parser.add_argument("--out", required=True, help="Output prefix or directory for CSV and Markdown summaries.")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    out = Path(args.out)
    if out.suffix:
        out.parent.mkdir(parents=True, exist_ok=True)
        csv_path = out.with_suffix(".csv")
        md_path = out.with_suffix(".md")
    else:
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "fixmatch_grid_summary.csv"
        md_path = out / "fixmatch_grid_summary.md"

    rows = collect_rows(runs_dir)
    if not rows:
        raise SystemExit(f"No completed runs found under {runs_dir}")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
