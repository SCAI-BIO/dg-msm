import logging
import string
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

PLOT_STYLE = "seaborn-v0_8-whitegrid"

TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 11
AXIS_LABEL_FONTWEIGHT = "bold"
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 9

HIGHLIGHT_LABELS = {"OS", "Baseline", "Death"}
HIGHLIGHT_COLOR = "#BC3C29"

TERMINAL_STATES = ("Death",)

KM_SCORE_CONFIG = ("KM Similarity Score", "KM Similarity", "#4C78A8")

AJ_CONFIG = ("AJ Similarity Score", "AJ Similarity", "#4C78A8")

OTHER_SCORE_CONFIGS = [
    ("Distribution similarity score", "Distribution Similarity", "#4C78A8"),
    ("Correlation score", "Correlation Similarity", "#4C78A8"),
]

MULTI_MODEL_PALETTE = [
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#4E79A7",
]


def plot_syndat_summary_cv(
    scores_long: pd.DataFrame,
    title: str  = None,
    figsize: tuple[float, float] = (16.0, 4.8),
    dpi: int = 300,
    show_points: bool = False,
    show: bool = True,
    score_ylim: tuple[float, float] = (50, 100),
    save_path: str  = None,
    model_order: list[str]  = None,
    model_colors: dict[str, str]  = None,
    include_km_similarity: bool = False,
    state_order: Sequence[str]  = None,
) -> plt.Figure:
    _validate_required_columns(
        scores_long,
        required={"fold", "metric", "value"},
        df_name="scores_long",
    )

    plt.style.use(PLOT_STYLE)

    multi_model = _has_multiple_models(scores_long)
    models = _extract_model_order(scores_long, model_order) if multi_model else []
    resolved_model_colors = (
        _resolve_model_colors(models, model_colors) if multi_model else {}
    )

    # Assemble score panels in the desired order:
    # KM scores (optional) -> AJ (if present) -> Distribution -> Correlation
    score_configs = []

    if include_km_similarity:
        score_configs.append(KM_SCORE_CONFIG)

    if _has_metric(scores_long, AJ_CONFIG[0]):
        score_configs.append(AJ_CONFIG)

    score_configs.extend(OTHER_SCORE_CONFIGS)

    n_panels = len(score_configs)
    width_ratios = [1.0] * n_panels

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"width_ratios": width_ratios},
    )

    axes = np.atleast_1d(axes)

    letters = string.ascii_lowercase

    for i, (ax, (metric_name, base_title, panel_color)) in enumerate(
        zip(axes, score_configs)
    ):
        panel_title = f"{letters[i]}) {base_title}"

        if multi_model:
            _plot_score_box_panel_multi(
                ax=ax,
                scores_long=scores_long,
                metric_name=metric_name,
                panel_title=panel_title,
                show_points=show_points,
                ylim=score_ylim,
                show_ylabel=(i == 0),
                model_order=models,
                model_colors=resolved_model_colors,
                state_order=state_order,
            )
        else:
            _plot_score_box_panel(
                ax=ax,
                scores_long=scores_long,
                metric_name=metric_name,
                panel_title=panel_title,
                box_color=panel_color,
                show_points=show_points,
                ylim=score_ylim,
                show_ylabel=(i == 0),
                state_order=state_order,
            )

    show_figure_model_legend = multi_model and bool(models)

    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold", y=0.995)

    if show_figure_model_legend:
        _add_model_legend_figure(
            fig=fig,
            models=models,
            model_colors=resolved_model_colors,
            y=0.955 if title else 0.985,
        )

    if title and show_figure_model_legend:
        tight_top = 0.86
    elif title or show_figure_model_legend:
        tight_top = 0.92
    else:
        tight_top = 0.98

    fig.tight_layout(rect=[0, 0, 1, tight_top])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        logger.info("Syndat CV summary figure saved to %s", save_path)

    if show:
        plt.show()

    return fig


def _has_metric(scores_long: pd.DataFrame, metric_name: str) -> bool:
    if "metric" not in scores_long.columns:
        return False
    return bool((scores_long["metric"] == metric_name).any())


def _has_multiple_models(scores_long: pd.DataFrame) -> bool:
    if "model" not in scores_long.columns:
        return False

    vals = scores_long["model"].dropna().astype(str).unique()
    return len(vals) > 1


def _extract_model_order(
    scores_long: pd.DataFrame,
    model_order: list[str]  = None,
) -> list[str]:
    if "model" not in scores_long.columns:
        return []

    present = [str(x) for x in pd.unique(scores_long["model"].dropna())]

    if model_order is None:
        return present

    ordered = [m for m in model_order if m in present]
    remaining = [m for m in present if m not in ordered]
    return ordered + remaining


def _resolve_model_colors(
    models: list[str],
    model_colors: dict[str, str]  = None,
) -> dict[str, str]:
    out = {}
    if model_colors is not None:
        out.update(model_colors)

    for i, model in enumerate(models):
        if model not in out:
            out[model] = MULTI_MODEL_PALETTE[i % len(MULTI_MODEL_PALETTE)]

    return out


def _plot_score_box_panel(
    ax: plt.Axes,
    scores_long: pd.DataFrame,
    metric_name: str,
    panel_title: str,
    box_color: str,
    show_points: bool,
    ylim: tuple[float, float],
    show_ylabel: bool,
    state_order: Sequence[str]  = None,
) -> None:
    labels, grouped = _prepare_score_groups_ordered(
        scores_long=scores_long,
        metric_name=metric_name,
        state_order=state_order,
    )

    if not labels:
        ax.set_visible(False)
        return

    positions = np.arange(1, len(grouped) + 1)

    bp = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
        whiskerprops={"color": box_color, "linewidth": 1.0},
        capprops={"color": box_color, "linewidth": 1.0},
        boxprops={"edgecolor": box_color, "linewidth": 1.0},
    )

    for i, (label, patch) in enumerate(zip(labels, bp["boxes"])):
        color = _label_color(label, box_color)

        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.18)

        for whisker in bp["whiskers"][2 * i : 2 * i + 2]:
            whisker.set_color(color)

        for cap in bp["caps"][2 * i : 2 * i + 2]:
            cap.set_color(color)

    if show_points:
        rng = np.random.default_rng(42)
        for pos, label, vals in zip(positions, labels, grouped):
            point_color = HIGHLIGHT_COLOR if _is_highlight_label(label) else "0.35"
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(
                pos + jitter,
                vals,
                s=20,
                color=point_color,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

    for pos, label, vals in zip(positions, labels, grouped):
        color = _label_color(label, box_color)
        q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])

        ax.vlines(pos, q25, q75, color=color, linewidth=3.0, zorder=4)
        ax.scatter(
            pos,
            med,
            s=34,
            marker="D",
            color=color,
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )

    ax.axhline(100, color="0.5", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_title(panel_title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel(
        "Similarity score" if show_ylabel else "",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight=AXIS_LABEL_FONTWEIGHT,
    )
    ax.set_ylim(*ylim)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=TICK_LABEL_FONTSIZE)

    for tick, label in zip(ax.get_xticklabels(), labels):
        if _is_highlight_label(label):
            tick.set_color(HIGHLIGHT_COLOR)
            tick.set_fontweight("bold")

    _style_axis(ax)


def _plot_score_box_panel_multi(
    ax: plt.Axes,
    scores_long: pd.DataFrame,
    metric_name: str,
    panel_title: str,
    show_points: bool,
    ylim: tuple[float, float],
    show_ylabel: bool,
    model_order: list[str],
    model_colors: dict[str, str],
    state_order: Sequence[str]  = None,
) -> None:
    labels, models, grouped = _prepare_score_groups_multi_ordered(
        scores_long=scores_long,
        metric_name=metric_name,
        model_order=model_order,
        state_order=state_order,
    )

    if not labels or not models:
        ax.set_visible(False)
        return

    base_positions = np.arange(1, len(labels) + 1, dtype=float)
    n_models = len(models)

    if n_models == 1:
        box_width = 0.50
        offsets = np.array([0.0])
    else:
        box_width = min(0.24, 0.60 / n_models)
        inner_gap = box_width * 0.15

        cluster_width = n_models * box_width + (n_models - 1) * inner_gap

        offsets = (
            np.arange(n_models, dtype=float) * (box_width + inner_gap)
            - cluster_width / 2
            + box_width / 2
        )

    datasets = []
    positions = []
    colors = []

    for label_idx, label in enumerate(labels):
        for model_idx, model in enumerate(models):
            vals = grouped.get((label, model))
            if vals is None or len(vals) == 0:
                continue

            datasets.append(vals)
            positions.append(base_positions[label_idx] + offsets[model_idx])
            colors.append(model_colors[model])

    if not datasets:
        ax.set_visible(False)
        return

    bp = ax.boxplot(
        datasets,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "0.35", "linewidth": 1.0},
        capprops={"color": "0.35", "linewidth": 1.0},
        boxprops={"edgecolor": "0.35", "linewidth": 1.0},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.22)

    for i, color in enumerate(colors):
        for whisker in bp["whiskers"][2 * i : 2 * i + 2]:
            whisker.set_color(color)
        for cap in bp["caps"][2 * i : 2 * i + 2]:
            cap.set_color(color)

    if show_points:
        rng = np.random.default_rng(42)
        for pos, vals, color in zip(positions, datasets, colors):
            jitter = rng.uniform(-box_width * 0.22, box_width * 0.22, size=len(vals))
            ax.scatter(
                pos + jitter,
                vals,
                s=18,
                color=color,
                alpha=0.65,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )

    for pos, vals, color in zip(positions, datasets, colors):
        q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])

        ax.vlines(pos, q25, q75, color=color, linewidth=2.8, zorder=4)
        ax.scatter(
            pos,
            med,
            s=30,
            marker="D",
            color=color,
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )

    ax.axhline(100, color="0.5", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_title(panel_title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_ylabel(
        "Similarity score" if show_ylabel else "",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight=AXIS_LABEL_FONTWEIGHT,
    )
    ax.set_ylim(*ylim)

    ax.set_xticks(base_positions)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=TICK_LABEL_FONTSIZE)

    for tick, label in zip(ax.get_xticklabels(), labels):
        if _is_highlight_label(label):
            tick.set_color(HIGHLIGHT_COLOR)
            tick.set_fontweight("bold")

    _style_axis(ax)


def _prepare_score_groups_ordered(
    scores_long: pd.DataFrame,
    metric_name: str,
    state_order: Sequence[str]  = None,
) -> tuple[list[str], list[np.ndarray]]:
    sub = scores_long.loc[scores_long["metric"] == metric_name].copy()
    if sub.empty:
        return [], []

    if "state_name" not in sub.columns:
        sub["state_name"] = np.nan
    if "visit" not in sub.columns:
        sub["visit"] = np.nan

    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"])

    if sub.empty:
        return [], []

    sub["plot_label"] = sub.apply(
        lambda row: _make_plot_label(
            state_name=row.get("state_name", np.nan),
            visit=row.get("visit", np.nan),
            for_km=(metric_name == "KM Similarity Score"),
        ),
        axis=1,
    )

    sub = sub.dropna(subset=["plot_label"])
    if sub.empty:
        return [], []

    label_order = _build_label_order(
        sub=sub,
        metric_name=metric_name,
        state_order=state_order,
    )

    labels: list[str] = []
    grouped: list[np.ndarray] = []

    for label in label_order:
        vals = sub.loc[sub["plot_label"] == label, "value"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        labels.append(label)
        grouped.append(vals)

    return labels, grouped


def _prepare_score_groups_multi_ordered(
    scores_long: pd.DataFrame,
    metric_name: str,
    model_order: list[str]  = None,
    state_order: Sequence[str]  = None,
) -> tuple[list[str], list[str], dict[tuple[str, str], np.ndarray]]:
    sub = scores_long.loc[scores_long["metric"] == metric_name].copy()
    if sub.empty:
        return [], [], {}

    if "model" not in sub.columns:
        sub["model"] = "synthetic"
    if "state_name" not in sub.columns:
        sub["state_name"] = np.nan
    if "visit" not in sub.columns:
        sub["visit"] = np.nan

    sub["model"] = sub["model"].astype(str)
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value", "model"])

    if sub.empty:
        return [], [], {}

    sub["plot_label"] = sub.apply(
        lambda row: _make_plot_label(
            state_name=row.get("state_name", np.nan),
            visit=row.get("visit", np.nan),
            for_km=(metric_name == "KM Similarity Score"),
        ),
        axis=1,
    )

    sub = sub.dropna(subset=["plot_label"])
    if sub.empty:
        return [], [], {}

    label_order = _build_label_order(
        sub=sub,
        metric_name=metric_name,
        state_order=state_order,
    )

    labels: list[str] = []
    for label in label_order:
        vals = sub.loc[sub["plot_label"] == label, "value"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            labels.append(label)

    models_present = [str(x) for x in pd.unique(sub["model"])]
    if model_order is None:
        models = models_present
    else:
        models = [m for m in model_order if m in models_present]
        models += [m for m in models_present if m not in models]

    grouped: dict[tuple[str, str], np.ndarray] = {}

    for label in labels:
        for model in models:
            vals = sub.loc[
                (sub["plot_label"] == label) & (sub["model"] == model),
                "value",
            ].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]

            if len(vals) > 0:
                grouped[(label, model)] = vals

    return labels, models, grouped


def _build_label_order(
    sub: pd.DataFrame,
    metric_name: str,
    state_order: Sequence[str]  = None,
) -> list[str]:
    meta = (
        sub.groupby("plot_label", sort=False, dropna=False)
        .agg(
            state_name=("state_name", "first"),
            visit=("visit", "first"),
        )
        .reset_index()
    )

    present_labels = meta["plot_label"].tolist()

    if present_labels == ["Overall"]:
        return present_labels

    by_state: dict[str, list[tuple[Any, str]]] = {}
    terminal_by_state: dict[str, list[tuple[Any, str]]] = {}

    for row in meta.itertuples(index=False):
        label = str(row.plot_label)
        state_name = "Overall" if pd.isna(row.state_name) else str(row.state_name)
        visit = row.visit

        if metric_name != "KM Similarity Score" and label == "OS":
            continue

        if state_name in {"Baseline", "OS", "Overall"}:
            continue

        if state_name in TERMINAL_STATES:
            terminal_by_state.setdefault(state_name, []).append((visit, label))
        else:
            by_state.setdefault(state_name, []).append((visit, label))

    ordered_states = _ordered_nonterminal_states(
        states_present=list(by_state.keys()),
        state_order=state_order,
    )

    labels: list[str] = []

    if metric_name != "KM Similarity Score" and "Baseline" in present_labels:
        labels.append("Baseline")

    for state in ordered_states:
        pairs = by_state.get(state, [])
        for _, label in sorted(pairs, key=lambda x: _visit_sort_key(x[0], x[1])):
            if label not in labels:
                labels.append(label)

    for state in TERMINAL_STATES:
        pairs = terminal_by_state.get(state, [])
        for _, label in sorted(pairs, key=lambda x: _visit_sort_key(x[0], x[1])):
            if label not in labels:
                labels.append(label)

    if metric_name == "KM Similarity Score" and "OS" in present_labels:
        labels.append("OS")

    for label in present_labels:
        if metric_name != "KM Similarity Score" and label == "OS":
            continue
        if label not in labels:
            labels.append(label)

    return labels


def _ordered_nonterminal_states(
    states_present: Sequence[str],
    state_order: Sequence[str] = None,
) -> list[str]:
    special = {"Baseline", "OS", "Overall"}

    ordered: list[str] = []
    seen: set[str] = set()

    for state in (state_order or []):
        if state in states_present and state not in special and state not in seen:
            ordered.append(state)
            seen.add(state)

    for state in states_present:
        if state not in special and state not in seen:
            ordered.append(state)
            seen.add(state)

    return ordered


def _visit_sort_key(visit: Any, label: str) -> tuple[int, Any, str]:
    if pd.isna(visit):
        return (0, -1, label)

    try:
        return (1, int(visit), label)
    except Exception:
        return (2, str(visit), label)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


def _is_highlight_label(label: str) -> bool:
    if label in HIGHLIGHT_LABELS:
        return True

    for state in TERMINAL_STATES:
        if label == state or label.startswith(f"{state} ("):
            return True

    return False


def _label_color(label: str, default_color: str) -> str:
    return HIGHLIGHT_COLOR if _is_highlight_label(label) else default_color


def _validate_required_columns(
    df: pd.DataFrame,
    required: set[str],
    df_name: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def _make_plot_label(state_name, visit, for_km: bool) -> str:
    state_name = "Overall" if pd.isna(state_name) else str(state_name)

    if for_km and state_name in {"Baseline", "OS"}:
        return "OS"

    if pd.isna(visit):
        return state_name

    try:
        visit_int = int(visit)
    except Exception:
        return f"{state_name} ({visit})"

    if visit_int == 0 or state_name == "Baseline":
        return state_name

    return f"{state_name} (v{visit_int})"


def _add_model_legend_figure(
    fig: plt.Figure,
    models: list[str],
    model_colors: dict[str, str],
    y: float = 0.98,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=model_colors[m],
            lw=2.0,
            linestyle="-",
            label=m,
        )
        for m in models
    ]

    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=min(len(handles), 4),
            frameon=True,
            framealpha=0.9,
            fontsize=LEGEND_FONTSIZE,
        )