# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. Config
# ============================================================
#
# seq,strength
# ACGT...,5.123
# CGTA...,7.456
#
DATA_PATH = (
    "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/merged_dataset63468.csv"

)



OUT_ROOT = (
    "/data/stu1/wrc3_pycharm_project/"
    "PromoDGDE_main/Data/SC/"
    "wrc_regression_63468_5fold"
)



SEQ_COL = "seq"



STRENGTH_COL = "strength"



NUM_BINS = 5



N_SPLITS = 5



SEED = 42


# ============================================================
# 2. Load data
# ============================================================


df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=[
        "seq",
        "strength"
    ]
)


print(
    "=" * 100
)

print(
    "[Regression 5-Fold Preprocessing]"
)

print(
    f"DATA_PATH : {DATA_PATH}"
)

print(
    f"OUT_ROOT  : {OUT_ROOT}"
)

print(
    f"N_SPLITS  : {N_SPLITS}"
)

print(
    f"NUM_BINS  : {NUM_BINS}"
)

print(
    f"SEED      : {SEED}"
)

print(
    "=" * 100
)


# ============================================================
# 3. Basic data cleaning
# ============================================================

before_n = len(df)



df = df.dropna(
    how="all"
).copy()



df["seq"] = (
    df["seq"]
    .astype(str)
    .str.strip()
    .str.upper()
)



df["strength"] = pd.to_numeric(
    df["strength"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "seq",
        "strength"
    ]
).reset_index(
    drop=True
)


after_n = len(df)


print(
    f"\nOriginal samples : {before_n}"
)

print(
    f"Valid samples    : {after_n}"
)


# ============================================================
# 4. Check expected sample number
# ============================================================

print(
    f"\nTotal samples: {len(df)}"
)


if len(df) != 63468:

    print(
        f"[Warning] Expected 63468 samples, "
        f"but found {len(df)}."
    )


# ============================================================
# 5. Create 5 equal-frequency strength bins
# ============================================================
#
#
# Bin 0 : lowest strength
# Bin 1
# Bin 2
# Bin 3
# Bin 4 : highest strength
#
#
# ============================================================


# ------------------------------------------------------------
#
#
# ------------------------------------------------------------

sorted_idx = np.argsort(
    df[STRENGTH_COL].values,
    kind="mergesort"
)


n_samples = len(df)


# ============================================================
#
#
#
# ============================================================

base_bin_size = (
    n_samples // NUM_BINS
)


bin_sizes = [
    base_bin_size
    for _ in range(
        NUM_BINS - 1
    )
]


last_bin_size = (
    n_samples
    - base_bin_size
    * (
        NUM_BINS - 1
    )
)


bin_sizes.append(
    last_bin_size
)


print(
    "\nPlanned bin sizes:"
)

for i, size in enumerate(
        bin_sizes
):

    print(
        f"Bin {i}: {size}"
    )


# ============================================================
# Assign bins
# ============================================================

bin_label = np.zeros(
    n_samples,
    dtype=np.int64
)


start = 0


for bin_id, size in enumerate(
        bin_sizes
):

    end = (
        start
        + size
    )


    indices_this_bin = sorted_idx[
        start:end
    ]


    bin_label[
        indices_this_bin
    ] = bin_id


    start = end


df["bin"] = bin_label


# ============================================================
# 6. Check global bin distribution
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "Global Bin Distribution"
)

print(
    "=" * 100
)


global_counts = (
    df["bin"]
    .value_counts()
    .sort_index()
)


for bin_id in range(
        NUM_BINS
):

    count = int(
        global_counts.loc[
            bin_id
        ]
    )


    ratio = (
        count
        / len(df)
    )


    subset = df[
        df["bin"] == bin_id
    ]


    print(
        f"Bin {bin_id} | "
        f"N={count:6d} | "
        f"Ratio={ratio:.6f} | "
        f"Strength range="
        f"[{subset[STRENGTH_COL].min():.6f}, "
        f"{subset[STRENGTH_COL].max():.6f}]"
    )


# ============================================================
# 7. Overall strength statistics
# ============================================================

print(
    "\nOverall strength statistics:"
)

print(
    f"Mean : "
    f"{df[STRENGTH_COL].mean():.6f}"
)

print(
    f"Std  : "
    f"{df[STRENGTH_COL].std():.6f}"
)

print(
    f"Min  : "
    f"{df[STRENGTH_COL].min():.6f}"
)

print(
    f"Max  : "
    f"{df[STRENGTH_COL].max():.6f}"
)


# ============================================================
# 8. Create output root
# ============================================================

os.makedirs(
    OUT_ROOT,
    exist_ok=True
)


# ============================================================
# Save full dataset with bin labels
# ============================================================

full_data_path = os.path.join(
    OUT_ROOT,
    "full_dataset_with_bins.csv"
)


df.to_csv(
    full_data_path,
    index=False
)


print(
    f"\nSaved full dataset with bins -> "
    f"{full_data_path}"
)


# ============================================================
# 9. Stratified 5-Fold
# ============================================================
#
#
#
#     df["bin"]
#
#
#     df["strength"]
# ============================================================

skf = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=SEED

)


all_indices = np.arange(
    len(df)
)


fold_summary_list = []


# ============================================================
# 10. Generate folds
# ============================================================

for fold_idx, (
        train_idx,
        val_idx
) in enumerate(

    skf.split(
        all_indices,
        df["bin"].values
    ),

    start=1

):


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Fold {fold_idx}"
    )

    print(
        "=" * 100
    )


    # ========================================================
    # Split dataframe
    # ========================================================

    train_df = (
        df.iloc[
            train_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    val_df = (
        df.iloc[
            val_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Output directory
    # ========================================================

    fold_dir = os.path.join(

        OUT_ROOT,

        f"fold{fold_idx}"

    )


    os.makedirs(

        fold_dir,

        exist_ok=True

    )


    # ========================================================
    # Save
    # ========================================================
    #
    #
    # seq
    # strength
    # bin
    #
    #
    # seq + strength
    # ========================================================

    train_path = os.path.join(

        fold_dir,

        "train.csv"

    )


    val_path = os.path.join(

        fold_dir,

        "val.csv"

    )


    train_df.to_csv(

        train_path,

        index=False

    )


    val_df.to_csv(

        val_path,

        index=False

    )


    # ========================================================
    # Basic counts
    # ========================================================

    print(
        f"Train samples : "
        f"{len(train_df)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_df)}"
    )


    print(
        f"Train ratio   : "
        f"{len(train_df) / len(df):.6f}"
    )

    print(
        f"Val ratio     : "
        f"{len(val_df) / len(df):.6f}"
    )


    # ========================================================
    # Train bin distribution
    # ========================================================

    print(
        "\nTrain bin distribution:"
    )


    train_bin_counts = (
        train_df["bin"]
        .value_counts()
        .sort_index()
    )


    for bin_id in range(
        NUM_BINS
    ):


        count = int(
            train_bin_counts.loc[
                bin_id
            ]
        )


        ratio = (
            count
            / len(train_df)
        )


        print(
            f"  Bin {bin_id}: "
            f"N={count:6d} | "
            f"Ratio={ratio:.6f}"
        )


    # ========================================================
    # Validation bin distribution
    # ========================================================

    print(
        "\nValidation bin distribution:"
    )


    val_bin_counts = (
        val_df["bin"]
        .value_counts()
        .sort_index()
    )


    for bin_id in range(
        NUM_BINS
    ):


        count = int(
            val_bin_counts.loc[
                bin_id
            ]
        )


        ratio = (
            count
            / len(val_df)
        )


        print(
            f"  Bin {bin_id}: "
            f"N={count:6d} | "
            f"Ratio={ratio:.6f}"
        )


    # ========================================================
    # Continuous strength statistics
    # ========================================================

    train_mean = (
        train_df[
            STRENGTH_COL
        ].mean()
    )


    train_std = (
        train_df[
            STRENGTH_COL
        ].std()
    )


    train_min = (
        train_df[
            STRENGTH_COL
        ].min()
    )


    train_max = (
        train_df[
            STRENGTH_COL
        ].max()
    )


    val_mean = (
        val_df[
            STRENGTH_COL
        ].mean()
    )


    val_std = (
        val_df[
            STRENGTH_COL
        ].std()
    )


    val_min = (
        val_df[
            STRENGTH_COL
        ].min()
    )


    val_max = (
        val_df[
            STRENGTH_COL
        ].max()
    )


    print(
        "\nContinuous strength statistics:"
    )


    print(
        "Train | "
        f"Mean={train_mean:.6f} | "
        f"Std={train_std:.6f} | "
        f"Min={train_min:.6f} | "
        f"Max={train_max:.6f}"
    )


    print(
        "Val   | "
        f"Mean={val_mean:.6f} | "
        f"Std={val_std:.6f} | "
        f"Min={val_min:.6f} | "
        f"Max={val_max:.6f}"
    )


    # ========================================================
    # Difference between train / val statistics
    # ========================================================

    print(
        "\nTrain-Val difference:"
    )


    print(
        f"Mean diff = "
        f"{abs(train_mean - val_mean):.6f}"
    )


    print(
        f"Std diff  = "
        f"{abs(train_std - val_std):.6f}"
    )


    # ========================================================
    # Check overlap
    # ========================================================

    overlap = np.intersect1d(
        train_idx,
        val_idx
    )


    if len(overlap) != 0:

        raise RuntimeError(

            f"Fold {fold_idx} has "
            f"{len(overlap)} overlapping samples!"

        )


    print(
        "\nTrain / Val overlap: 0"
    )


    # ========================================================
    # Save fold distribution report
    # ========================================================

    report_rows = []


    for bin_id in range(
        NUM_BINS
    ):


        tr_count = int(
            train_bin_counts.loc[
                bin_id
            ]
        )


        va_count = int(
            val_bin_counts.loc[
                bin_id
            ]
        )


        report_rows.append({

            "Fold": fold_idx,

            "Bin": bin_id,

            "Train_Count": tr_count,

            "Train_Ratio": (
                tr_count
                / len(train_df)
            ),

            "Val_Count": va_count,

            "Val_Ratio": (
                va_count
                / len(val_df)
            )

        })


    report_df = pd.DataFrame(
        report_rows
    )


    report_path = os.path.join(

        fold_dir,

        "bin_distribution.csv"

    )


    report_df.to_csv(

        report_path,

        index=False

    )


    # ========================================================
    # Fold summary
    # ========================================================

    fold_summary_list.append({

        "Fold": fold_idx,

        "Train_N": len(
            train_df
        ),

        "Val_N": len(
            val_df
        ),

        "Train_Mean": train_mean,

        "Train_Std": train_std,

        "Train_Min": train_min,

        "Train_Max": train_max,

        "Val_Mean": val_mean,

        "Val_Std": val_std,

        "Val_Min": val_min,

        "Val_Max": val_max,

        "Mean_Diff": abs(
            train_mean
            - val_mean
        ),

        "Std_Diff": abs(
            train_std
            - val_std
        )

    })


    print(
        f"\nSaved:"
    )

    print(
        f"  {train_path}"
    )

    print(
        f"  {val_path}"
    )

    print(
        f"  {report_path}"
    )


# ============================================================
# 11. Save overall fold summary
# ============================================================

fold_summary_df = pd.DataFrame(
    fold_summary_list
)


fold_summary_path = os.path.join(

    OUT_ROOT,

    "five_fold_summary.csv"

)


fold_summary_df.to_csv(

    fold_summary_path,

    index=False

)


# ============================================================
# 12. Verify each sample appears in validation exactly once
# ============================================================

val_seen = np.zeros(
    len(df),
    dtype=np.int64
)


for _, val_idx in skf.split(

        all_indices,

        df["bin"].values

):

    val_seen[
        val_idx
    ] += 1


unique_seen, seen_counts = np.unique(

    val_seen,

    return_counts=True

)


print(
    "\n"
    + "=" * 100
)

print(
    "Final Validation Coverage Check"
)

print(
    "=" * 100
)


print(
    dict(
        zip(
            unique_seen.tolist(),
            seen_counts.tolist()
        )
    )
)


if not np.all(
        val_seen == 1
):

    raise RuntimeError(

        "Some samples were not used exactly once "
        "as validation samples."

    )


print(
    "Every sample appears exactly once "
    "in the validation sets."
)


# ============================================================
# 13. Finish
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "[DONE] Regression stratified 5-fold preprocessing completed."
)

print(
    "=" * 100
)


print(
    f"\nFull dataset:"
)

print(
    full_data_path
)


print(
    f"\n5-fold summary:"
)

print(
    fold_summary_path
)


print(
    f"\nFold folders:"
)


for fold_idx in range(
        1,
        N_SPLITS + 1
):

    print(
        os.path.join(
            OUT_ROOT,
            f"fold{fold_idx}"
        )
    )