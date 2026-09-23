import logging
import pandas as pd
import syndat


from .aj_divergence import aj_divergence_per_state

logger = logging.getLogger(__name__)


def syndat_scores(
    real_msp,
    synthetic_msp,
    time_scale: float = 1.0,
    n_points: int = 500,
    stratify_by_visit: bool = False,
    min_rows_per_group: int = 30,
    n_conditioning_visits: int = 0,
) -> pd.DataFrame:
    if n_conditioning_visits < 0:
        raise ValueError("n_conditioning_visits must be >= 0")
    if min_rows_per_group < 1:
        raise ValueError("min_rows_per_group must be >= 1")
    if time_scale <= 0:
        raise ValueError("time_scale must be > 0")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    real_msp.assert_same_metadata(synthetic_msp)

    state_features = real_msp.list_state_features()
    baseline_features = real_msp.list_baseline_features()

    real_states = real_msp.states.copy()
    synthetic_states = synthetic_msp.states.copy()

    real_baseline = real_msp.baseline[baseline_features].copy()
    synthetic_baseline = synthetic_msp.baseline[baseline_features].copy()

    baseline_cat_features = set(
        real_msp.list_baseline_features(categories={"cat", "bin", "disc", "thresh"})
    )
    state_cat_features = set(
        real_msp.list_state_features(categories={"cat", "bin", "disc", "thresh"})
    )

    # Treat first n_conditioning_visits visits as observed, not predicted
    skip_baseline = n_conditioning_visits > 0
    visit_cutoff = n_conditioning_visits

    if visit_cutoff > 0:
        real_states_cov = real_states[real_states["visit"] > visit_cutoff].copy()
        synthetic_states_cov = synthetic_states[synthetic_states["visit"] > visit_cutoff].copy()
    else:
        real_states_cov = real_states
        synthetic_states_cov = synthetic_states

    rows: list[dict] = []

    if not skip_baseline:
        real_bl = real_baseline.copy()
        synth_bl = synthetic_baseline.copy()

        _cast_categoricals(real_bl, baseline_cat_features)
        _cast_categoricals(synth_bl, baseline_cat_features)

        baseline_distribution_score = syndat.scores.distribution(real_bl, synth_bl)
        baseline_correlation_score = syndat.scores.correlation(real_bl, synth_bl)

        rows.append(
            {
                "state_id": 0,
                "state_name": "Baseline",
                "visit": 0,
                "Distribution similarity score": baseline_distribution_score,
                "Correlation score": baseline_correlation_score,
            }
        )

    for state_id in real_msp.nonterminal_state_indices:
        state_name = real_msp.state_names[state_id - 1]

        real_state = real_states_cov[real_states_cov["state"] == state_id]
        synthetic_state = synthetic_states_cov[synthetic_states_cov["state"] == state_id]

        if real_state.empty or synthetic_state.empty:
            logger.warning(
                "syndat_scores: skipping state_id=%s (%s): %s.",
                state_id,
                state_name,
                _empty_reason(real_state, synthetic_state),
            )
            continue

        if stratify_by_visit:
            visits = sorted(
                set(real_state["visit"].unique()) | set(synthetic_state["visit"].unique())
            )
        else:
            visits = [None]

        for visit in visits:
            if visit is None:
                real_sub = real_state
                synth_sub = synthetic_state
                visit_label = 0
            else:
                real_sub = real_state[real_state["visit"] == visit]
                synth_sub = synthetic_state[synthetic_state["visit"] == visit]
                visit_label = int(visit)

            if len(real_sub) < min_rows_per_group or len(synth_sub) < min_rows_per_group:
                logger.info(
                    "syndat_scores: skipping (%s, visit=%s): insufficient rows "
                    "(real=%d, synthetic=%d, min=%d).",
                    state_name,
                    "all" if visit is None else visit,
                    len(real_sub),
                    len(synth_sub),
                    min_rows_per_group,
                )
                continue

            score_row = _compute_state_scores(
                real_sub=real_sub,
                synth_sub=synth_sub,
                state_features=state_features,
                state_cat_features=state_cat_features,
                state_id=state_id,
                state_name=state_name,
                visit=visit_label,
            )
            if score_row is not None:
                rows.append(score_row)

    score_columns = [
        "state_id",
        "state_name",
        "visit",
        "Distribution similarity score",
        "Correlation score",
    ]
    scores_per_state = pd.DataFrame(rows, columns=score_columns)

    if stratify_by_visit:
        scores_per_state = scores_per_state.set_index(["state_name", "visit"])
    else:
        scores_per_state = scores_per_state.set_index("state_name").drop(columns="visit")

    aj_scores = aj_divergence_per_state(
        real_msp,
        synthetic_msp,
        time_scale=time_scale,
        n_points=n_points,
    )

    if stratify_by_visit:
        scores_per_state = scores_per_state.join(aj_scores, on="state_name", how="outer")
    else:
        scores_per_state = scores_per_state.join(aj_scores, how="outer")

    return scores_per_state


def _cast_categoricals(df: pd.DataFrame, cat_features: set[str]) -> None:
    """Cast categorical feature columns to pandas Categorical dtype in-place."""
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _empty_reason(real_state: pd.DataFrame, synthetic_state: pd.DataFrame) -> str:
    """Return a human-readable reason for why a state slice is empty."""
    if real_state.empty and synthetic_state.empty:
        return "no real or synthetic rows"
    elif real_state.empty:
        return "no real rows"
    return "no synthetic rows"


def _compute_state_scores(
    real_sub: pd.DataFrame,
    synth_sub: pd.DataFrame,
    state_features: list[str],
    state_cat_features: set[str],
    state_id: int,
    state_name: str,
    visit: int,
):
    """
    Compute distribution + correlation scores for one (state, visit) slice.

    Returns None if the slice is degenerate or syndat raises.
    """
    real_sel = real_sub[state_features].copy()
    synthetic_sel = synth_sub[state_features].copy()

    # Cast categorical state features
    _cast_categoricals(real_sel, state_cat_features)
    _cast_categoricals(synthetic_sel, state_cat_features)

    # Filter degenerate columns (<=1 unique non-NaN value in either dataset)
    valid_cols = [
        col for col in state_features
        if real_sel[col].dropna().nunique() > 1
        and synthetic_sel[col].dropna().nunique() > 1
    ]

    if not valid_cols:
        logger.warning(
            "syndat_scores: skipping (%s, visit=%d): no non-degenerate features.",
            state_name, visit,
        )
        return None

    real_sel = real_sel[valid_cols]
    synthetic_sel = synthetic_sel[valid_cols]

    try:
        distribution_score = syndat.scores.distribution(real_sel, synthetic_sel)
        correlation_score = syndat.scores.correlation(real_sel, synthetic_sel)
    except ValueError as e:
        logger.warning(
            "syndat_scores: skipping (%s, visit=%d): syndat raised %r.",
            state_name, visit, e,
        )
        return None

    return {
        "state_id": state_id,
        "state_name": state_name,
        "visit": visit,
        "Distribution similarity score": distribution_score,
        "Correlation score": correlation_score,
    }