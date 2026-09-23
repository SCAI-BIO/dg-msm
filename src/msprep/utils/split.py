from __future__ import annotations

from typing import Tuple, Union, Optional

import logging
from sklearn.model_selection import train_test_split

from ..multistateprep import MultiStatePrep

logger = logging.getLogger(__name__)


MSP_SPLITS = Union[
    Tuple[MultiStatePrep, MultiStatePrep, MultiStatePrep], 
    Tuple[MultiStatePrep, MultiStatePrep]
    ]

def msp_split_by_max_state(
    msp: MultiStatePrep,
    test_size: float = 0.2,
    val_size: Optional[float] = 0.2,
    stratify_by_state: bool = True,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    ) -> MSP_SPLITS:
    """
    Split a MultiStatePrep into train(/val)/test by person_id.

    If stratify_by_state=True, splitting is stratified by each person's
    maximum state (using the `state` column in msp._states).

    Parameters
    ----------
    msp :
        Source MultiStatePrep instance.
    test_size :
        Fraction of *all persons* in the test split. Must be in (0, 1).
    val_size :
        Fraction of *all persons* in the validation split. If 0.0 or None,
        no validation split is created and only (train, test) is returned.
        If not None, must be in [0, 1) and satisfy test_size + val_size < 1.
    stratify_by_state :
        If True, stratify splits by each person's max state value.
    random_state :
        Random seed for reproducible splits. Passed to `train_test_split`.
    shuffle :
        Whether to shuffle person_ids before splitting. Passed to
        `train_test_split`.


   ### Returns
    If val_size is None or 0.0:
        (msp_train, msp_test)
    Else:
        (msp_train, msp_val, msp_test)
    """
    msp._ensure_built()

    # ---------------------------
    # Validate sizes
    # ---------------------------
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    do_val = val_size not in (None, 0.0)

    if do_val:
        if not (0.0 <= val_size < 1.0):
            raise ValueError("val_size must be in [0, 1) when not None.")
        if test_size + val_size >= 1.0:
            raise ValueError(
                "test_size + val_size must be < 1 when val_size is not None."
            )
        # Adjusted val fraction on the remaining (non-test) persons
        val_size_adjusted = val_size / (1.0 - test_size)
    else:
        val_size_adjusted = None  # unused
        logger.info(
            "msp_split_by_max_state: val_size=%r -> no validation split; "
            "will return only (train, test).",
            val_size,
        )

    # ---------------------------
    # Build per-person table
    # ---------------------------
    states = msp._states

    if stratify_by_state:
        # One row per person with their max state (MultiStatePrep guarantees 'state')
        per_person = (
            states.groupby("person_id", as_index=False)["state"]
            .max()
            .rename(columns={"state": "strata"})
        )
    else:
        # One row per person, with dummy strata (no stratification)
        per_person = states[["person_id"]].drop_duplicates().copy()
        per_person["strata"] = 0  # single class, unused when stratify=False

    if per_person.empty:
        raise ValueError("No persons available for splitting.")

    n_persons = len(per_person)
    if n_persons < 2:
        raise ValueError(
            f"Need at least 2 persons to split, but found {n_persons}."
        )

    # Basic logging of population and strata
    strata_counts = per_person["strata"].value_counts().to_dict()
    logger.info(
        "msp_split_by_max_state: total persons=%d, stratify_by_state=%s, "
        "strata distribution=%s",
        n_persons,
        stratify_by_state,
        strata_counts,
    )

    # ---------------------------
    # First split: train+val vs test
    # ---------------------------
    stratify_arg = per_person["strata"] if stratify_by_state else None

    trainval_ids_df, test_ids_df = train_test_split(
        per_person[["person_id"]],
        test_size=test_size,
        stratify=stratify_arg,
        random_state=random_state,
        shuffle=shuffle,
    )

    n_trainval = len(trainval_ids_df)
    n_test = len(test_ids_df)
    logger.info(
        "msp_split_by_max_state: test split -> n_trainval=%d, n_test=%d "
        "(test_size=%.3f, frac_test=%.3f)",
        n_trainval,
        n_test,
        test_size,
        n_test / n_persons,
    )

    # ---------------------------
    # Second split: train vs val (if requested)
    # ---------------------------
    if do_val:
        per_person_trainval = per_person[
            per_person.person_id.isin(trainval_ids_df.person_id)
        ]
        if per_person_trainval.empty:
            raise ValueError("No persons left for train/val after test split.")

        stratify_arg_tv = per_person_trainval["strata"] if stratify_by_state else None

        train_ids_df, val_ids_df = train_test_split(
            per_person_trainval[["person_id"]],
            test_size=val_size_adjusted,
            stratify=stratify_arg_tv,
            random_state=random_state,
            shuffle=shuffle,
        )

        n_train = len(train_ids_df)
        n_val = len(val_ids_df)
        logger.info(
            "msp_split_by_max_state: val split -> n_train=%d, n_val=%d "
            "(target val_size=%.3f, frac_val=%.3f)",
            n_train,
            n_val,
            val_size,
            n_val / n_persons,
        )
    else:
        train_ids_df = trainval_ids_df
        val_ids_df = None
        n_train = len(train_ids_df)
        logger.info(
            "msp_split_by_max_state: no validation split -> n_train=%d, n_test=%d",
            n_train,
            n_test,
        )

    train_ids = train_ids_df["person_id"].tolist()
    test_ids = test_ids_df["person_id"].tolist()
    val_ids = val_ids_df["person_id"].tolist() if val_ids_df is not None else []

    # Sanity logging: check disjointness and coverage
    overlap_train_test = set(train_ids) & set(test_ids)
    overlap_train_val = set(train_ids) & set(val_ids)
    overlap_val_test = set(val_ids) & set(test_ids)
    if overlap_train_test or overlap_train_val or overlap_val_test:
        logger.warning(
            "msp_split_by_max_state: overlaps detected between splits: "
            "train∩test=%s, train∩val=%s, val∩test=%s",
            overlap_train_test,
            overlap_train_val,
            overlap_val_test,
        )


    # ---------------------------
    # Build new MultiStatePrep objects via subset_persons
    # ---------------------------
    msp_train = msp.subset_persons(
        person_ids=train_ids,
        dataset_name=f"{msp.dataset_name}_train",
    )
    msp_test = msp.subset_persons(
        person_ids=test_ids,
        dataset_name=f"{msp.dataset_name}_test",
    )

    if do_val:
        msp_val = msp.subset_persons(
            person_ids=val_ids,
            dataset_name=f"{msp.dataset_name}_val",
        )
        return msp_train, msp_val, msp_test

    return msp_train, msp_test