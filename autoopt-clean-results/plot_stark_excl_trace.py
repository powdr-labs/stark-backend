#!/usr/bin/env python3
"""Plot STARK proving time excluding trace as an SVG line chart.

Reads result.csv/results.csv, keeps APC 0/100/300, and writes an SVG plot with
one line per APC setting across the ordered optimization iterations.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXPECTED_APCS = (0, 100, 300)
SERIES_COLORS = {
    0: "#1d4ed8",
    100: "#ea580c",
    300: "#059669",
}


def resolve_default_input() -> Path:
    candidates = [
        HERE / "result.csv",
        HERE / "results.csv",
        REPO / "result.csv",
        REPO / "results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find result.csv or results.csv in the repository root "
        f"or {HERE}"
    )


def load_rows(path: Path) -> tuple[list[str], dict[str, str], dict[int, list[float]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    steps: list[str] = []
    labels: dict[str, str] = {}
    values_by_apc: dict[int, dict[str, float]] = {apc: {} for apc in EXPECTED_APCS}

    for row in rows:
        apc = int(row["apc"])
        if apc not in EXPECTED_APCS:
            continue
        step = row["step"]
        if step not in steps:
            steps.append(step)
        labels[step] = row["label"]
        values_by_apc[apc][step] = float(row["stark_excl_trace_ms"])

    missing: list[str] = []
    for step in steps:
        for apc in EXPECTED_APCS:
            if step not in values_by_apc[apc]:
                missing.append(f"{step} APC={apc}")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing stark_excl_trace_ms values for: {joined}")

    ordered_values = {
        apc: [values_by_apc[apc][step] for step in steps] for apc in EXPECTED_APCS
    }
    return steps, labels, ordered_values


def nice_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0:
        return 100.0
    raw = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    if residual <= 1:
        nice = 1
    elif residual <= 2:
        nice = 2
    elif residual <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def format_ms(value: float) -> str:
    return f"{int(round(value)):,} ms"


def path_from_points(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(
        f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}"
        for i, (x, y) in enumerate(points)
    )


def make_svg(
    steps: list[str],
    labels: dict[str, str],
    values_by_apc: dict[int, list[float]],
    input_path: Path,
) -> str:
    columns = 2
    rows_per_col = math.ceil(len(steps) / columns)
    key_h = max(212, 72 + rows_per_col * 30)

    width = 1400
    height = 980
    margin_left = 110
    margin_right = 50
    margin_top = 100
    margin_bottom = 92 + key_h + 24
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for series in values_by_apc.values() for value in series]
    y_max_raw = max(all_values)
    padding = max(60.0, y_max_raw * 0.08)
    y_axis_min = 0.0
    y_axis_max = y_max_raw + padding
    tick_step = nice_step(y_axis_max - y_axis_min)
    y_tick_min = 0.0
    y_tick_max = math.ceil(y_axis_max / tick_step) * tick_step
    y_span = y_tick_max - y_tick_min

    def x_pos(index: int) -> float:
        if len(steps) == 1:
            return margin_left + plot_width / 2
        return margin_left + (index * plot_width) / (len(steps) - 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1 - (value - y_tick_min) / y_span)

    y_ticks: list[float] = []
    tick = y_tick_min
    while tick <= y_tick_max + (tick_step * 0.5):
        y_ticks.append(tick)
        tick += tick_step

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title subtitle">'
    )
    lines.append(
        '<style>'
        'text{fill:#0f172a;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}'
        '.muted{fill:#475569}'
        '.grid{stroke:#cbd5e1;stroke-width:1}'
        '.axis{stroke:#334155;stroke-width:1.5}'
        '.plot-frame{fill:#ffffff;stroke:#94a3b8;stroke-width:1}'
        '.legend-box{fill:#ffffff;stroke:#cbd5e1;stroke-width:1}'
        '.key-box{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1}'
        '</style>'
    )
    lines.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    lines.append('<title id="title">STARK proving time excluding trace by iteration</title>')
    lines.append(
        '<desc id="subtitle">'
        f'Generated from {escape(str(input_path))}. '
        'Each line shows STARK proving time excluding trace for a fixed APC value.'
        '</desc>'
    )

    lines.append(
        '<text x="110" y="48" font-size="30" font-weight="700">'
        'STARK Proving Time Excluding Trace'
        "</text>"
    )

    plot_x = margin_left
    plot_y = margin_top
    lines.append(
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" '
        'rx="8" class="plot-frame"/>'
    )

    for tick_value in y_ticks:
        y = y_pos(tick_value)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" '
            f'y2="{y:.2f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{margin_left - 16}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="14" class="muted">{format_ms(tick_value)}</text>'
        )

    for idx in range(len(steps)):
        x = x_pos(idx)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{margin_top + plot_height}" class="grid" stroke-dasharray="4 6"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 26}" text-anchor="middle" '
            f'font-size="14" class="muted">{idx:02d}</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" class="axis"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_height}" class="axis"/>'
    )
    lines.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{margin_top + plot_height + 60}" '
        'text-anchor="middle" font-size="18" font-weight="600">Iteration</text>'
    )
    lines.append(
        f'<text x="34" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 34 {margin_top + plot_height / 2:.2f})" '
        'text-anchor="middle" font-size="18" font-weight="600">'
        'STARK proving time excl. trace (ms)'
        "</text>"
    )

    for apc in EXPECTED_APCS:
        points = [(x_pos(idx), y_pos(value)) for idx, value in enumerate(values_by_apc[apc])]
        color = SERIES_COLORS[apc]
        lines.append(
            f'<path d="{path_from_points(points)}" fill="none" stroke="{color}" '
            'stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for idx, value in enumerate(values_by_apc[apc]):
            cx, cy = points[idx]
            lines.append(
                f'<g><title>Iteration {idx:02d}, APC={apc}, {format_ms(value)}</title>'
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.5" fill="{color}" '
                'stroke="#ffffff" stroke-width="1.5"/></g>'
            )

    legend_x = margin_left + plot_width - 210
    legend_y = margin_top + 24
    legend_w = 180
    legend_h = 110
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" '
        'rx="8" class="legend-box"/>'
    )
    lines.append(
        f'<text x="{legend_x + 16}" y="{legend_y + 28}" font-size="16" font-weight="700">Legend</text>'
    )
    for row, apc in enumerate(EXPECTED_APCS):
        y = legend_y + 54 + row * 24
        color = SERIES_COLORS[apc]
        lines.append(
            f'<line x1="{legend_x + 16}" y1="{y}" x2="{legend_x + 48}" y2="{y}" '
            f'stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>'
        )
        lines.append(
            f'<circle cx="{legend_x + 32}" cy="{y}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{legend_x + 58}" y="{y + 5}" font-size="15">APC={apc}</text>'
        )

    key_x = margin_left
    key_y = margin_top + plot_height + 92
    key_w = plot_width
    lines.append(
        f'<rect x="{key_x}" y="{key_y}" width="{key_w}" height="{key_h}" rx="8" class="key-box"/>'
    )
    lines.append(
        f'<text x="{key_x + 16}" y="{key_y + 30}" font-size="18" font-weight="700">Experiments</text>'
    )

    col_width = key_w / columns
    for idx, step in enumerate(steps):
        col = idx // rows_per_col
        row = idx % rows_per_col
        entry_x = key_x + 16 + col * col_width
        entry_y = key_y + 62 + row * 30
        label = labels[step]
        lines.append(
            f'<text x="{entry_x:.2f}" y="{entry_y:.2f}" font-size="14">'
            f'<tspan font-weight="700">{idx:02d}</tspan>'
            f'<tspan class="muted"> {escape(label)}</tspan>'
            "</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=resolve_default_input(),
        help="Path to result.csv/results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "stark_excl_trace_by_iteration.svg",
        help="Destination SVG path",
    )
    args = parser.parse_args()

    steps, labels, values_by_apc = load_rows(args.input)
    svg = make_svg(steps, labels, values_by_apc, args.input.resolve())
    args.output.write_text(svg)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
