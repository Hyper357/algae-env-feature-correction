from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
    )


def correct_features_linear(
    x_train: np.ndarray,
    x_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map features to the median environment state of the training fold.

    For each feature j:
        g_j(z) = beta0_j + beta1_j * z
        x_corr = x - (g_j(z) - g_j(z0))

    All parameters are fitted on the training fold only.
    """
    z_train_2d = z_train.reshape(-1, 1)
    z_test_2d = z_test.reshape(-1, 1)
    z0 = float(np.median(z_train))

    model = LinearRegression().fit(z_train_2d, x_train)
    drift_train = model.predict(z_train_2d) - model.predict(np.array([[z0]]))
    drift_test = model.predict(z_test_2d) - model.predict(np.array([[z0]]))
    return x_train - drift_train, x_test - drift_test


def loso_predict(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    mode: str = "baseline",
) -> np.ndarray:
    loo = LeaveOneOut()
    pred = np.full_like(y, np.nan, dtype=float)

    for train_idx, test_idx in loo.split(x):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr = y[train_idx]

        if mode == "corrected":
            if z is None:
                raise ValueError("z is required for corrected mode")
            x_tr, x_te = correct_features_linear(
                x_tr, x_te, z[train_idx], z[test_idx]
            )
        elif mode == "concat":
            if z is None:
                raise ValueError("z is required for concat mode")
            x_tr = np.column_stack([x_tr, z[train_idx]])
            x_te = np.column_stack([x_te, z[test_idx]])
        elif mode != "baseline":
            raise ValueError(f"Unknown mode: {mode}")

        scaler = StandardScaler().fit(x_tr)
        x_tr_s = scaler.transform(x_tr)
        x_te_s = scaler.transform(x_te)

        predictor = RidgeCV(alphas=np.logspace(-4, 4, 41)).fit(x_tr_s, y_tr)
        pred[test_idx] = predictor.predict(x_te_s)

    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--target", default="chlorophyll")
    parser.add_argument("--id", default="station_id")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata)
    df = metadata.merge(features, on=args.id, how="inner")

    feature_cols = [c for c in features.columns if c != args.id]
    required = [args.target, args.env] + feature_cols
    df = df.dropna(subset=required).reset_index(drop=True)

    x = df[feature_cols].to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=float)
    z = df[args.env].to_numpy(dtype=float)

    predictions = {
        "baseline": loso_predict(x, y, mode="baseline"),
        "concat_diagnostic": loso_predict(x, y, z=z, mode="concat"),
        "explicit_correction": loso_predict(x, y, z=z, mode="corrected"),
    }

    for name, y_pred in predictions.items():
        m = _metrics(y, y_pred)
        print(
            f"{name:22s} MAE={m.mae:.6g}  RMSE={m.rmse:.6g}  R2={m.r2:.6g}"
        )

    base = _metrics(y, predictions["baseline"])
    corr = _metrics(y, predictions["explicit_correction"])
    improvement = (base.mae - corr.mae) / base.mae if base.mae else np.nan
    print(f"relative_MAE_improvement={improvement:.2%}")


if __name__ == "__main__":
    main()
