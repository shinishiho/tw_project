"""Render publication-ready plots from the locked full-test summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


RUN_GROUPS = {
    "visible": (
        "yolo_pretrained_visible",
        "yolo_finetuned_visible",
        "locate_anything_visible",
    ),
    "infrared": (
        "yolo_pretrained_infrared",
        "yolo_finetuned_infrared",
        "locate_anything_infrared",
    ),
}
RUN_LABELS = {
    "yolo_pretrained_visible": "YOLO pretrained",
    "yolo_pretrained_infrared": "YOLO pretrained",
    "yolo_finetuned_visible": "YOLO thermal-tuned",
    "yolo_finetuned_infrared": "YOLO thermal-tuned",
    "locate_anything_visible": "LocateAnything",
    "locate_anything_infrared": "LocateAnything",
}
SHORT_LABELS = {
    "yolo_pretrained_visible": "YOLO pre / visible",
    "yolo_pretrained_infrared": "YOLO pre / thermal",
    "yolo_finetuned_visible": "YOLO tuned / visible",
    "yolo_finetuned_infrared": "YOLO tuned / thermal",
    "locate_anything_visible": "LA / visible",
    "locate_anything_infrared": "LA / thermal",
}
RUN_STYLES = {
    "yolo_pretrained_visible": ("#0072B2", "o"),
    "yolo_pretrained_infrared": ("#0072B2", "o"),
    "yolo_finetuned_visible": ("#D55E00", "s"),
    "yolo_finetuned_infrared": ("#D55E00", "s"),
    "locate_anything_visible": ("#009E73", "^"),
    "locate_anything_infrared": ("#009E73", "^"),
}
COMPARISON_LABELS = {
    "visible_headline_yolo_pretrained_minus_locate_anything": (
        "Visible: YOLO pre - LA"
    ),
    "thermal_headline_yolo_finetuned_minus_locate_anything": (
        "Thermal: YOLO tuned - LA"
    ),
    "pretrained_modality_visible_minus_infrared": "YOLO pre: visible - thermal",
    "thermal_supervision_finetuned_minus_pretrained": (
        "Thermal: YOLO tuned - pre"
    ),
    "finetuned_modality_infrared_minus_visible": (
        "YOLO tuned: thermal - visible"
    ),
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "scripts/plot_full_results.py"},
    )
    plt.close(fig)


def plot_primary_f1(summary: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for axis_index, (ax, (modality, run_names)) in enumerate(
        zip(axes, RUN_GROUPS.items(), strict=True)
    ):
        for y, run_name in enumerate(reversed(run_names)):
            interval = summary["runs"][run_name]["metrics"]["iou_0.50"]["f1"]
            value = interval["estimate"]
            color, marker = RUN_STYLES[run_name]
            ax.errorbar(
                value,
                y,
                xerr=[[value - interval["low"]], [interval["high"] - value]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=7,
                linewidth=1.8,
            )
            place_left = value > 0.84
            ax.text(
                value - 0.025 if place_left else value + 0.025,
                y,
                f"{value:.3f}",
                va="center",
                ha="right" if place_left else "left",
            )
        ax.set_yticks(
            range(3),
            [RUN_LABELS[name] for name in reversed(run_names)]
            if axis_index == 0
            else ["", "", ""],
        )
        ax.set_title("Visible images" if modality == "visible" else "Thermal images")
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 at IoU 0.50")
    fig.suptitle("Detection accuracy with 95% paired bootstrap intervals", y=1.02)
    _save(fig, output_dir / "primary-f1.png")


def plot_stratified_f1(summary: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    strata = (
        (
            "brightness_quintile",
            ["0", "1", "2", "3", "4"],
            ["Q0", "Q1", "Q2", "Q3", "Q4"],
        ),
        (
            "crowd_bin",
            ["1", "2", "3-4", "5-8", "9+"],
            ["1", "2", "3-4", "5-8", "9+"],
        ),
    )
    for column, (modality, run_names) in enumerate(RUN_GROUPS.items()):
        for row, (attribute, keys, labels) in enumerate(strata):
            ax = axes[row, column]
            for run_name in run_names:
                values = [
                    summary["runs"][run_name]["strata_iou_0.50"][attribute][key][
                        "metrics"
                    ]["f1"]
                    for key in keys
                ]
                color, marker = RUN_STYLES[run_name]
                ax.plot(
                    labels,
                    values,
                    color=color,
                    marker=marker,
                    linewidth=1.8,
                    markersize=5,
                    label=RUN_LABELS[run_name],
                )
            ax.set_ylim(0, 1)
            ax.set_title("Visible images" if modality == "visible" else "Thermal images")
            ax.set_ylabel("F1 at IoU 0.50" if column == 0 else "")
            ax.set_xlabel(
                "Visible-image brightness quintile"
                if row == 0
                else "People per image"
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=RUN_STYLES[name][0],
            marker=RUN_STYLES[name][1],
            label=RUN_LABELS[name],
        )
        for name in RUN_GROUPS["visible"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Accuracy across dataset-only brightness and crowd strata", y=1.08)
    fig.tight_layout()
    _save(fig, output_dir / "stratified-f1.png")


def plot_paired_differences(summary: dict, output_dir: Path) -> None:
    comparisons = summary["paired_differences_left_minus_right"]
    names = list(COMPARISON_LABELS)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    offsets = {"iou_0.50": 0.13, "iou_0.75": -0.13}
    styles = {
        "iou_0.50": ("#0072B2", "o", "IoU 0.50"),
        "iou_0.75": ("#D55E00", "s", "IoU 0.75"),
    }
    for threshold, offset in offsets.items():
        color, marker, label = styles[threshold]
        for index, name in enumerate(reversed(names)):
            interval = comparisons[name][threshold]["f1"]
            value = interval["estimate"]
            ax.errorbar(
                value,
                index + offset,
                xerr=[[value - interval["low"]], [interval["high"] - value]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=6,
                linewidth=1.7,
                label=label if index == 0 else None,
            )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(
        range(len(names)), [COMPARISON_LABELS[name] for name in reversed(names)]
    )
    ax.set_xlabel("Paired F1 difference (left minus right)")
    ax.set_title("Paired model and modality effects with 95% intervals")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "paired-f1-differences.png")


def plot_accuracy_efficiency(summary: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    offsets = {
        "yolo_pretrained_visible": (8, 8),
        "yolo_pretrained_infrared": (8, -16),
        "yolo_finetuned_visible": (8, 8),
        "yolo_finetuned_infrared": (8, -16),
        "locate_anything_visible": (-8, -16),
        "locate_anything_infrared": (-8, 8),
    }
    for run_name, run in summary["runs"].items():
        latency = run["efficiency"]["median_latency_ms"]
        f1 = run["metrics"]["iou_0.50"]["f1"]["estimate"]
        color, marker = RUN_STYLES[run_name]
        visible = run_name.endswith("_visible")
        ax.scatter(
            latency,
            f1,
            s=75,
            marker=marker,
            facecolor=color if visible else "white",
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        x_offset, y_offset = offsets[run_name]
        ax.annotate(
            SHORT_LABELS[run_name],
            (latency, f1),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
        )
    ax.set_xscale("log")
    ax.set_xlim(4, 2_500)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Warm median batch-1 latency (ms, log scale)")
    ax.set_ylabel("F1 at IoU 0.50")
    ax.set_title("Accuracy-latency trade-off")
    modality_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            label="Visible",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="Thermal",
        ),
    ]
    ax.legend(handles=modality_handles, loc="lower right")
    _save(fig, output_dir / "accuracy-latency.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/full/full_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    if summary.get("test_images") != 3_463 or len(summary.get("runs", {})) != 6:
        raise ValueError("expected the locked six-run, 3,463-image summary")

    _configure_style()
    plot_primary_f1(summary, args.output_dir)
    plot_stratified_f1(summary, args.output_dir)
    plot_paired_differences(summary, args.output_dir)
    plot_accuracy_efficiency(summary, args.output_dir)
    print(f"Wrote four plots to {args.output_dir}")


if __name__ == "__main__":
    main()
