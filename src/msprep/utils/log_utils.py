import pandas as pd
import logging

logger = logging.getLogger(__name__)


def log_df_change(
    step: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    key_col: str = "person_id",
) -> None:
    """Log how many rows/keys were removed by a step."""
    n_before = len(before)
    n_after = len(after)

    if key_col in before.columns and key_col in after.columns:
        before_ids = set(before[key_col])
        after_ids = set(after[key_col])
        removed_ids = before_ids - after_ids
        n_removed_ids = len(removed_ids)
    else:
        removed_ids = set()
        n_removed_ids = 0
    if n_removed_ids != 0:
        logger.info(
            "%s: rows %d -> %d (difference=%d), unique %s removed=%d",
            step,
            n_before,
            n_after,
            n_after - n_before,
            key_col,
            n_removed_ids,
        )
        
def _filter_df(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    key_col: str = "person_id",
) -> pd.DataFrame:
    """Apply a boolean mask with logging of exclusions (OMOP side)."""
    before = df
    after = df[mask].copy()
    log_df_change(reason, before, after, key_col=key_col)
    return after
