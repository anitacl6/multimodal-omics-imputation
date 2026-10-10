"""
Multimodal Omics Missing-Data Imputation Benchmark
===================================================

Portfolio version of a graduate biostatistics practicum analysis.

Goal
----
Evaluate missing-value imputation methods for high-dimensional plasma RNA data
and assess whether adding metabolomics or synovial RNA predictors improves
imputation accuracy.

Methods
-------
- Mean imputation
- Median imputation
- K-nearest-neighbors imputation
- IterativeImputer (single iterative chained-regression imputation)
- ChainImputer (neural-network iterative imputation)

Evaluation
----------
Artificial MCAR missingness is injected into plasma features only. Models are
evaluated only on masked plasma cells using MAE and RMSE across repeated seeds.

Data
----
Public multimodal osteoarthritis datasets are loaded directly from:
https://github.com/divya031090/DeepLearning_KOA

Third-party method
------------------
The ChainImputer implementation below is adapted from:
https://github.com/KhrTim/ChainImputer
MIT License, Copyright (c) 2025 KhrTim.
See THIRD_PARTY_NOTICES.md in this repository.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_REPO_RAW = (
    "https://raw.githubusercontent.com/divya031090/"
    "DeepLearning_KOA/main"
)

DATA_URLS = {
    "metabolites": f"{DATA_REPO_RAW}/metabolites.csv",
    "plasma": f"{DATA_REPO_RAW}/RNA_plasma.csv",
    "synovial": f"{DATA_REPO_RAW}/RNA_synovial.csv",
    "urine": f"{DATA_REPO_RAW}/RNA_urine.csv",
}

PREDICTOR_SETS = {
    "plasma_only": (),
    "plasma+metabolites": ("metabolites",),
    "plasma+synovial": ("synovial",),
}

BEST_PARAMS = {
    "plasma_only": {
        "chain": dict(
            max_epochs=35,
            lr=0.0007,
            hidden_dim=176,
            imputation_scheme="ascending",
        ),
        "knn": dict(n_neighbors=11, weights="distance"),
        "iterative": dict(
            max_iter=20,
            imputation_order="ascending",
            n_nearest_features=200,
            initial_strategy="mean",
            tol=1e-3,
        ),
    },
    "plasma+metabolites": {
        "chain": dict(
            max_epochs=29,
            lr=0.0007,
            hidden_dim=156,
            imputation_scheme="ascending",
        ),
        "knn": dict(n_neighbors=15, weights="distance"),
        "iterative": dict(
            max_iter=10,
            imputation_order="ascending",
            n_nearest_features=150,
            initial_strategy="mean",
            tol=1e-3,
        ),
    },
    "plasma+synovial": {
        "chain": dict(
            max_epochs=9,
            lr=0.0010,
            hidden_dim=144,
            imputation_scheme="ascending",
        ),
        "knn": dict(n_neighbors=17, weights="distance"),
        "iterative": dict(
            max_iter=20,
            imputation_order="ascending",
            n_nearest_features=150,
            initial_strategy="mean",
            tol=1e-3,
        ),
    },
}


@dataclass
class PreparedData:
    model_input: np.ndarray
    plasma_truth: np.ndarray
    missing_mask: np.ndarray
    n_predictor_cols: int


def load_public_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    filenames = {
        "metabolites": "metabolites.csv",
        "plasma": "RNA_plasma.csv",
        "synovial": "RNA_synovial.csv",
        "urine": "RNA_urine.csv",
    }

    data: dict[str, pd.DataFrame] = {}

    for name, filename in filenames.items():
        source = (
            Path(data_dir) / filename
            if data_dir is not None
            else DATA_URLS[name]
        )
        data[name] = pd.read_csv(source)

    for df in data.values():
        id_col = get_id_column(df)
        if id_col is not None:
            df.sort_values(id_col, inplace=True)
            df.reset_index(drop=True, inplace=True)

    validate_row_alignment(data)
    return data


def get_id_column(df: pd.DataFrame) -> str | None:
    matches = [col for col in df.columns if col.upper() == "ID"]
    return matches[0] if matches else None


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.upper() != "ID"]


def validate_row_alignment(data: dict[str, pd.DataFrame]) -> None:
    plasma = data["plasma"]
    plasma_id = get_id_column(plasma)

    for name, df in data.items():
        if len(df) != len(plasma):
            raise ValueError(
                f"{name} contains {len(df)} rows; plasma contains "
                f"{len(plasma)} rows."
            )

        other_id = get_id_column(df)
        if plasma_id is not None and other_id is not None:
            left = plasma[plasma_id].reset_index(drop=True)
            right = df[other_id].reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(
                    f"Participant ID order in {name} does not match plasma."
                )


def inject_mcar(
    df: pd.DataFrame,
    missing_percentage: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 <= missing_percentage <= 100:
        raise ValueError("missing_percentage must be between 0 and 100.")

    out = df.copy(deep=True)
    n_rows, n_cols = out.shape
    n_cells = n_rows * n_cols
    n_missing = round(n_cells * missing_percentage / 100)

    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(n_cells, size=n_missing, replace=False)

    row_idx, col_idx = np.unravel_index(flat_indices, (n_rows, n_cols))
    for i, j in zip(row_idx, col_idx):
        out.iat[i, j] = np.nan

    return out


def standardize_complete_predictor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = feature_columns(out)

    means = out[cols].mean()
    sds = out[cols].std().replace(0, 1).fillna(1)

    out[cols] = (out[cols] - means) / sds
    return out


def prepare_data_for_seed(
    data: dict[str, pd.DataFrame],
    seed: int,
    missing_percentage: float,
    predictor_sets: Sequence[str],
) -> PreparedData:
    predictor_sets = tuple(predictor_sets)
    valid_predictors = {"metabolites", "synovial", "urine"}

    invalid = set(predictor_sets) - valid_predictors
    if invalid:
        raise ValueError(f"Invalid predictor modalities: {sorted(invalid)}.")

    if len(set(predictor_sets)) != len(predictor_sets):
        raise ValueError("predictor_sets contains duplicate modality names.")

    plasma = data["plasma"]
    plasma_cols = feature_columns(plasma)

    plasma_missing = plasma.copy()
    plasma_missing[plasma_cols] = inject_mcar(
        plasma[plasma_cols],
        missing_percentage=missing_percentage,
        seed=seed,
    )

    means = plasma_missing[plasma_cols].mean(skipna=True)
    sds = (
        plasma_missing[plasma_cols]
        .std(skipna=True)
        .replace(0, 1)
        .fillna(1)
    )

    plasma_missing_std = plasma_missing.copy()
    plasma_truth_std = plasma.copy()

    plasma_missing_std[plasma_cols] = (
        plasma_missing[plasma_cols] - means
    ) / sds

    plasma_truth_std[plasma_cols] = (
        plasma[plasma_cols] - means
    ) / sds

    predictor_arrays = {}
    for name in valid_predictors:
        standardized = standardize_complete_predictor(data[name])
        predictor_arrays[name] = standardized[
            feature_columns(standardized)
        ].to_numpy(dtype=np.float32)

    plasma_missing_array = plasma_missing_std[
        plasma_cols
    ].to_numpy(dtype=np.float32)

    plasma_truth_array = plasma_truth_std[
        plasma_cols
    ].to_numpy(dtype=np.float32)

    selected_arrays = [predictor_arrays[name] for name in predictor_sets]
    n_predictor_cols = sum(arr.shape[1] for arr in selected_arrays)

    if selected_arrays:
        model_input = np.concatenate(
            selected_arrays + [plasma_missing_array],
            axis=1,
        )
    else:
        model_input = plasma_missing_array.copy()

    return PreparedData(
        model_input=model_input,
        plasma_truth=plasma_truth_array,
        missing_mask=np.isnan(plasma_missing_array),
        n_predictor_cols=n_predictor_cols,
    )


def extract_plasma_block(
    full_matrix: np.ndarray,
    prepared: PreparedData,
) -> np.ndarray:
    n_plasma = prepared.plasma_truth.shape[1]
    start = prepared.n_predictor_cols
    stop = start + n_plasma

    plasma = np.asarray(full_matrix)[:, start:stop]

    if plasma.shape != prepared.plasma_truth.shape:
        raise ValueError(
            f"Expected plasma shape {prepared.plasma_truth.shape}, "
            f"got {plasma.shape}."
        )

    return plasma


class ChainImputer:
    """
    Neural-network iterative imputer adapted from KhrTim/ChainImputer.

    Source:
    https://github.com/KhrTim/ChainImputer/blob/main/update.py

    License:
    MIT License, Copyright (c) 2025 KhrTim.

    See THIRD_PARTY_NOTICES.md for the required MIT notice.
    """

    def __init__(
        self,
        max_epochs: int = 10,
        hidden_dim: int = 64,
        lr: float = 0.001,
        imputation_scheme: str = "ascending",
    ):
        self.max_epochs = max_epochs
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.imputation_scheme = imputation_scheme
        self.models: dict[int, nn.Module] = {}
        self.feature_sets: dict[int, list[int]] = {}
        self.imputation_order: list[int] = []
        self.prefilled_values: dict[int, float | None] = {}

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32).copy()
        features = list(range(data.shape[1]))

        if self.imputation_scheme == "ascending":
            features.sort(key=lambda f: np.isnan(data[:, f]).sum())
        elif self.imputation_scheme == "descending":
            features.sort(
                key=lambda f: np.isnan(data[:, f]).sum(),
                reverse=True,
            )
        elif self.imputation_scheme == "random":
            random.shuffle(features)
        else:
            raise ValueError(
                "imputation_scheme must be ascending, descending, or random."
            )

        usable: list[int] = []

        for feature in features:
            if not np.isnan(data[:, feature]).any():
                usable.append(feature)
                self.prefilled_values[feature] = None

        if not usable:
            feature = features[0]
            mean_value = float(np.nanmean(data[:, feature]))
            missing = np.isnan(data[:, feature])
            data[missing, feature] = mean_value
            usable.append(feature)
            self.prefilled_values[feature] = mean_value

        for feature in features:
            if feature in usable:
                continue

            target = data[:, feature]
            observed = ~np.isnan(target)

            if observed.sum() == 0:
                continue

            x_train = data[observed][:, usable]
            y_train = target[observed]

            model = nn.Sequential(
                nn.Linear(len(usable), self.hidden_dim),
                nn.Sigmoid(),
                nn.Linear(self.hidden_dim, 1),
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=1e-4,
            )
            criterion = nn.MSELoss()

            x_tensor = torch.tensor(x_train, dtype=torch.float32)
            y_tensor = torch.tensor(
                y_train,
                dtype=torch.float32,
            ).unsqueeze(1)

            for _ in range(self.max_epochs):
                model.train()
                optimizer.zero_grad()
                prediction = model(x_tensor)
                loss = criterion(prediction, y_tensor)
                loss.backward()
                optimizer.step()

            self.models[feature] = model
            self.feature_sets[feature] = list(usable)
            self.imputation_order.append(feature)

            missing = np.isnan(target)
            if missing.any():
                x_missing = torch.tensor(
                    data[missing][:, usable],
                    dtype=torch.float32,
                )
                model.eval()
                with torch.no_grad():
                    prediction = model(x_missing).squeeze(1).cpu().numpy()
                data[missing, feature] = prediction

            usable.append(feature)

        return data


def impute_matrix(
    method: str,
    prepared: PreparedData,
    params: dict,
    seed: int,
) -> np.ndarray:
    x = prepared.model_input.copy()

    if method == "mean":
        return SimpleImputer(strategy="mean").fit_transform(x)

    if method == "median":
        return SimpleImputer(strategy="median").fit_transform(x)

    if method == "knn":
        return KNNImputer(**params).fit_transform(x)

    if method == "iterative":
        return IterativeImputer(
            random_state=seed,
            sample_posterior=False,
            **params,
        ).fit_transform(x)

    if method == "chain":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return ChainImputer(**params).fit_transform(x)

    raise ValueError(f"Unknown method: {method}")


def score_masked_plasma(
    imputed_full_matrix: np.ndarray,
    prepared: PreparedData,
) -> tuple[float, float]:
    plasma_imputed = extract_plasma_block(imputed_full_matrix, prepared)

    y_true = prepared.plasma_truth[prepared.missing_mask]
    y_pred = plasma_imputed[prepared.missing_mask]

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if y_true.size == 0:
        raise ValueError("No valid masked plasma cells are available to score.")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def evaluate_method(
    data: dict[str, pd.DataFrame],
    method: str,
    predictor_name: str,
    missing_percentage: float,
    seeds: Iterable[int],
) -> pd.DataFrame:
    predictor_sets = PREDICTOR_SETS[predictor_name]

    if method in {"mean", "median"}:
        params = {}
    else:
        params = BEST_PARAMS[predictor_name][method]

    rows = []

    for seed in seeds:
        print(
            f"method={method:9s} | predictors={predictor_name:20s} | "
            f"missing={missing_percentage:>4.0f}% | seed={seed}"
        )

        prepared = prepare_data_for_seed(
            data=data,
            seed=seed,
            missing_percentage=missing_percentage,
            predictor_sets=predictor_sets,
        )

        imputed = impute_matrix(
            method=method,
            prepared=prepared,
            params=params,
            seed=seed,
        )

        rmse, mae = score_masked_plasma(imputed, prepared)

        rows.append(
            {
                "method": method,
                "predictors": predictor_name,
                "missing_percentage": missing_percentage,
                "seed": seed,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

    return pd.DataFrame(rows)


def run_benchmark(
    data: dict[str, pd.DataFrame],
    methods: Sequence[str],
    predictor_names: Sequence[str],
    missing_percentages: Sequence[float],
    seeds: Iterable[int],
) -> pd.DataFrame:
    all_results = []

    for missing_percentage, predictor_name, method in product(
        missing_percentages,
        predictor_names,
        methods,
    ):
        result = evaluate_method(
            data=data,
            method=method,
            predictor_name=predictor_name,
            missing_percentage=missing_percentage,
            seeds=seeds,
        )
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


def summarize_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_results
        .groupby(
            ["method", "predictors", "missing_percentage"],
            as_index=False,
        )
        .agg(
            MAE=("MAE", "mean"),
            MAE_SD=("MAE", "std"),
            RMSE=("RMSE", "mean"),
            RMSE_SD=("RMSE", "std"),
        )
        .sort_values(
            ["missing_percentage", "predictors", "MAE", "RMSE"]
        )
        .reset_index(drop=True)
    )


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> None:
    sd_col = f"{metric}_SD"

    fig, ax = plt.subplots(figsize=(10, 6))

    for (method, predictors), group in summary.groupby(
        ["method", "predictors"]
    ):
        group = group.sort_values("missing_percentage")
        ax.errorbar(
            group["missing_percentage"],
            group[metric],
            yerr=group[sd_col],
            marker="o",
            capsize=3,
            label=f"{method} | {predictors}",
        )

    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across missingness levels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def tune_knn(
    data: dict[str, pd.DataFrame],
    predictor_name: str,
    k_values: Sequence[int],
    weights_values: Sequence[str],
    seeds: Iterable[int],
    missing_percentage: float = 10,
) -> pd.DataFrame:
    rows = []
    predictor_sets = PREDICTOR_SETS[predictor_name]

    for k, weights in product(k_values, weights_values):
        seed_scores = []

        for seed in seeds:
            prepared = prepare_data_for_seed(
                data,
                seed,
                missing_percentage,
                predictor_sets,
            )
            imputed = KNNImputer(
                n_neighbors=k,
                weights=weights,
            ).fit_transform(prepared.model_input.copy())

            rmse, mae = score_masked_plasma(imputed, prepared)
            seed_scores.append((rmse, mae))

        scores = np.asarray(seed_scores)
        rows.append(
            {
                "predictors": predictor_name,
                "k": k,
                "weights": weights,
                "RMSE": scores[:, 0].mean(),
                "MAE": scores[:, 1].mean(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark imputation methods on public multimodal omics data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Optional local directory containing the four CSV files. "
            "If omitted, data are read from the public source GitHub repository."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a fast demonstration using two seeds, 10%% missingness, "
            "and plasma-only predictors."
        ),
    )
    parser.add_argument(
        "--skip-chain",
        action="store_true",
        help="Skip ChainImputer for a faster run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_public_data(args.data_dir)

    methods = ["mean", "median", "knn", "iterative", "chain"]
    if args.skip_chain:
        methods.remove("chain")

    if args.quick:
        missing_percentages = [10]
        predictor_names = ["plasma_only"]
        seeds = range(11, 13)
    else:
        missing_percentages = [10, 30, 50]
        predictor_names = list(PREDICTOR_SETS)
        seeds = range(11, 31)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    seed_results = run_benchmark(
        data=data,
        methods=methods,
        predictor_names=predictor_names,
        missing_percentages=missing_percentages,
        seeds=seeds,
    )

    summary = summarize_results(seed_results)

    seed_results.to_csv(
        results_dir / "seed_level_results.csv",
        index=False,
    )
    summary.to_csv(
        results_dir / "summary_results.csv",
        index=False,
    )

    plot_metric(
        summary,
        metric="MAE",
        output_path=figures_dir / "mae_by_missingness.png",
    )
    plot_metric(
        summary,
        metric="RMSE",
        output_path=figures_dir / "rmse_by_missingness.png",
    )

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved results to results/ and figures/.")


if __name__ == "__main__":
    main()
