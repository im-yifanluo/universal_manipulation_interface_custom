#!/usr/bin/env python3
"""Visualize policy predictions and measured EEF motion from inference JSONL logs.

By default this compares the four recent successful turn-knob runs.  The newest
run is shown first so the inferred hardware-skip sequence is 2, 1, 2, 3.

Example:
    conda run -n umi python visualize_inference_trajectory.py
    conda run -n umi python visualize_inference_trajectory.py 48 45 --show
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EPISODES = (48, 45, 43, 40)
DEFAULT_LOG_ROOT = Path(__file__).resolve().parent / "data" / "turn_knob_ros_logs"
DEFAULT_OUTPUT = DEFAULT_LOG_ROOT / "recent_successful_trajectories.png"


@dataclass
class Episode:
    episode_id: int
    records: list[dict]
    measured_eef: np.ndarray
    action_chunks: list[np.ndarray]
    stitched_target: np.ndarray
    hardware_skip: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episodes",
        metavar="EPISODE",
        type=int,
        nargs="*",
        default=list(DEFAULT_EPISODES),
        help="episode IDs to plot (default, newest first: 48 45 43 40)",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"episode directory (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output PNG (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--inference-latency-steps",
        type=int,
        default=1,
        help="model steps skipped for inference latency (default: 1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the interactive Matplotlib window",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing inference log: {path}")

    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

    if len(records) < 2:
        raise ValueError(f"Need at least two predictions to visualize {path}")
    return records


def infer_hardware_skip(records: Sequence[dict], inference_latency_steps: int) -> int:
    """Infer HARDWARE_LATENCY_STEPS from each non-initial stitched trajectory."""
    inferred = []
    for record in records[1:]:
        action_steps = int(record["selected_action_steps"])
        interpolation_factor = float(record["interpolation_factor"])
        skip = action_steps - inference_latency_steps - record["traj_len"] / interpolation_factor
        rounded = int(round(skip))
        if not np.isclose(skip, rounded):
            raise ValueError(f"Non-integral hardware skip inferred from record: {skip}")
        inferred.append(rounded)

    if len(set(inferred)) != 1:
        raise ValueError(f"Hardware skip changed within one episode: {sorted(set(inferred))}")
    return inferred[0]


def interpolate_position(start: np.ndarray, end: np.ndarray, factor: int) -> list[np.ndarray]:
    """Match inference_umi.interpolate_between_actions for position columns."""
    return [start + (end - start) * i / factor for i in range(1, factor + 1)]


def reconstruct_stitched_target(
    records: Sequence[dict], hardware_skip: int, inference_latency_steps: int
) -> np.ndarray:
    """Reconstruct the published target path from the logged policy chunks."""
    chunks = [np.asarray(record["action_chunk_pos"], dtype=float) for record in records]
    factor = int(round(float(records[0]["interpolation_factor"])))

    target = [chunks[0][0]]
    for start, end in zip(chunks[0][:-1], chunks[0][1:]):
        target.extend(interpolate_position(start, end, factor))

    entry_index = inference_latency_steps + hardware_skip
    for previous, current in zip(chunks[:-1], chunks[1:]):
        if entry_index >= len(current):
            raise ValueError(
                f"Stitch entry index {entry_index} is outside an action chunk of length {len(current)}"
            )
        target.extend(interpolate_position(previous[-1], current[entry_index], factor))
        for start, end in zip(current[entry_index:-1], current[entry_index + 1 :]):
            target.extend(interpolate_position(start, end, factor))

    return np.asarray(target)


def load_episode(logs_root: Path, episode_id: int, inference_latency_steps: int) -> Episode:
    records = read_records(logs_root / str(episode_id) / "prediction_debug.jsonl")
    hardware_skip = infer_hardware_skip(records, inference_latency_steps)
    action_chunks = [np.asarray(record["action_chunk_pos"], dtype=float) for record in records]
    return Episode(
        episode_id=episode_id,
        records=records,
        measured_eef=np.asarray([record["eef_pos"] for record in records], dtype=float),
        action_chunks=action_chunks,
        stitched_target=reconstruct_stitched_target(
            records, hardware_skip, inference_latency_steps
        ),
        hardware_skip=hardware_skip,
    )


def set_equal_limits(axes, episodes: Sequence[Episode]) -> None:
    all_points = np.concatenate(
        [
            points
            for episode in episodes
            for points in (episode.measured_eef, *episode.action_chunks)
        ],
        axis=0,
    )
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max((upper - lower).max() / 2.0 * 1.08, 1e-3)
    for ax in axes:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))


def plot_episodes(episodes: Sequence[Episode], output: Path, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    is_single = len(episodes) == 1
    columns = 2 if len(episodes) > 1 else 1
    rows = int(np.ceil(len(episodes) / columns))
    figure_size = (11.0, 9.5) if is_single else (7.3 * columns, 6.2 * rows)
    figure = plt.figure(figsize=figure_size, constrained_layout=True)
    axes = [figure.add_subplot(rows, columns, i + 1, projection="3d") for i in range(len(episodes))]
    max_chunks = max(len(episode.action_chunks) for episode in episodes)
    # A discrete, high-contrast map makes neighboring action chunks easy to
    # distinguish even where their predicted paths overlap.
    cmap = plt.get_cmap("turbo", max_chunks)

    for ax, episode in zip(axes, episodes):
        for index, chunk in enumerate(episode.action_chunks):
            color = cmap(index)
            ax.plot(
                *chunk.T,
                color=color,
                linewidth=1.8 if is_single else 1.1,
                alpha=0.78 if is_single else 0.45,
            )
            ax.scatter(
                *chunk.T,
                color=color,
                s=8 if is_single else 5,
                alpha=0.48 if is_single else 0.28,
            )
            # Make the first target of every connected action chunk explicit.
            ax.scatter(
                *chunk[0],
                color=color,
                marker="o",
                s=62 if is_single else 32,
                edgecolor="#111111",
                linewidth=1.0,
                alpha=1.0,
                depthshade=False,
                zorder=5,
            )

        actual = episode.measured_eef
        ax.plot(
            *actual.T,
            color="#202020",
            linewidth=3.2 if is_single else 2.3,
            marker="o",
            markersize=4.0 if is_single else 2.8,
            alpha=0.38,
        )
        ax.scatter(*actual[0], color="#2ca02c", marker="o", s=90 if is_single else 65, depthshade=False)
        ax.scatter(*actual[-1], color="#d62728", marker="*", s=150 if is_single else 110, depthshade=False)
        ax.set_title(
            f"Episode {episode.episode_id}  |  hardware skip {episode.hardware_skip}\n"
            f"{len(episode.records)} predictions, final counter {episode.records[-1]['counter']}"
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, alpha=0.25)

        if is_single:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import BoundaryNorm

            boundaries = np.arange(len(episode.action_chunks) + 1) - 0.5
            colorbar = figure.colorbar(
                ScalarMappable(
                    norm=BoundaryNorm(boundaries, cmap.N),
                    cmap=cmap,
                ),
                ax=ax,
                boundaries=boundaries,
                ticks=np.arange(len(episode.action_chunks)),
                shrink=0.68,
                pad=0.08,
            )
            colorbar.set_label("Action chunk index")

    set_equal_limits(axes, episodes)
    title = (
        f"Inference trajectory — Episode {episodes[0].episode_id}\n"
        "Each policy action chunk has a different color"
        if is_single
        else "Recent successful inference trajectories\n"
        "Each policy action chunk has a different color"
    )
    figure.suptitle(title, fontsize=18 if is_single else 16)
    legend = [
        Line2D([0], [0], color="#202020", lw=2.3, marker="o", markersize=4, alpha=0.38, label="Measured EEF (at inference times)"),
        Line2D([0], [0], color=cmap(max_chunks // 2), lw=1.8, alpha=0.8, label="Policy action chunks"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(max_chunks // 2), markeredgecolor="#111111", markersize=7, label="Action chunk start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728", markersize=10, label="End"),
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=3 if is_single else 5,
        frameon=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved {output}")
    for episode in episodes:
        print(
            f"Episode {episode.episode_id}: hardware_skip={episode.hardware_skip}, "
            f"predictions={len(episode.records)}, target_points={len(episode.stitched_target)}"
        )
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.episodes:
        raise SystemExit("Specify at least one episode")
    episodes = [
        load_episode(args.logs_root, episode_id, args.inference_latency_steps)
        for episode_id in args.episodes
    ]
    plot_episodes(episodes, args.output, args.show)


if __name__ == "__main__":
    main()
