"""PyTorch utilities for the SciVer MLP router validation run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from router_model_lib import (
    build_preprocessor,
    candidate_metadata,
    candidate_sort_key,
    group_kfold_splits,
    normalize_model_names,
    prediction_frame,
    routing_metrics,
    target_columns_for,
)

try:  # pragma: no cover - exercised in environments with torch installed
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - keeps importable for dependency diagnostics
    torch = None
    nn = None

_TorchModuleBase = nn.Module if nn is not None else object


DTYPE_CHOICES = ("float32", "float64")
DEVICE_CHOICES = ("auto", "cpu", "mps")


@dataclass(frozen=True)
class TorchRuntime:
    requested_device: str
    requested_dtype: str
    effective_device: str
    torch_dtype_name: str
    torch_version: str
    mps_available: bool
    mps_built: bool
    fallback_reason: str

    @property
    def torch_dtype(self) -> Any:
        require_torch()
        return torch.float32 if self.torch_dtype_name == "float32" else torch.float64

    @property
    def device(self) -> Any:
        require_torch()
        return torch.device(self.effective_device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "requested_dtype": self.requested_dtype,
            "effective_device": self.effective_device,
            "torch_dtype": self.torch_dtype_name,
            "torch_version": self.torch_version,
            "mps_available": self.mps_available,
            "mps_built": self.mps_built,
            "fallback_reason": self.fallback_reason,
        }


def require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for the torch MLP router. Install project requirements with "
            "`python -m pip install -r requirements.txt`."
        )


def resolve_runtime(requested_device: str = "auto", requested_dtype: str = "float32") -> TorchRuntime:
    require_torch()
    if requested_device not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device {requested_device!r}; expected one of {DEVICE_CHOICES}.")
    if requested_dtype not in DTYPE_CHOICES:
        raise ValueError(f"Unknown dtype {requested_dtype!r}; expected one of {DTYPE_CHOICES}.")

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    fallback_reason = ""
    if requested_device == "auto":
        effective_device = "mps" if mps_available else "cpu"
    elif requested_device == "mps":
        if not mps_available:
            raise RuntimeError("Requested --device mps, but PyTorch reports MPS is unavailable.")
        effective_device = "mps"
    else:
        effective_device = "cpu"

    if effective_device == "mps" and requested_dtype == "float64":
        effective_device = "cpu"
        fallback_reason = "PyTorch MPS does not support float64 consistently; using CPU for float64."

    return TorchRuntime(
        requested_device=requested_device,
        requested_dtype=requested_dtype,
        effective_device=effective_device,
        torch_dtype_name=requested_dtype,
        torch_version=str(torch.__version__),
        mps_available=mps_available,
        mps_built=mps_built,
        fallback_reason=fallback_reason,
    )


def parameter_count(input_dim: int, hidden_layer_sizes: tuple[int, ...], output_dim: int) -> int:
    total = 0
    previous = int(input_dim)
    for width in hidden_layer_sizes:
        total += previous * int(width) + int(width)
        previous = int(width)
    total += previous * int(output_dim) + int(output_dim)
    return int(total)


def candidate_grid(
    seed: int,
    input_dim: int,
    output_dim: int,
    candidate_limit: int | None = None,
) -> list[dict[str, Any]]:
    layer_options = [
        (8,),
        (16,),
        (32,),
        (64,),
        (8, 8),
        (16, 16),
        (32, 16),
        (32, 32),
        (64, 32),
    ]
    candidates: list[dict[str, Any]] = []
    order = 0
    for layers in layer_options:
        for activation in ["relu", "tanh"]:
            for alpha in [0.0001, 0.001, 0.01, 0.1]:
                order += 1
                params = parameter_count(input_dim, layers, output_dim)
                layer_label = "x".join(str(width) for width in layers)
                candidates.append(
                    {
                        "kind": "torch_mlp",
                        "hidden_layer_sizes": layers,
                        "hidden_layer_count": len(layers),
                        "max_hidden_width": max(layers),
                        "activation": activation,
                        "alpha": alpha,
                        "optimizer": "torch_lbfgs",
                        "seed": seed,
                        "input_dim": int(input_dim),
                        "output_dim": int(output_dim),
                        "parameter_count": params,
                        "complexity_score": float(params),
                        "complexity_order": order,
                        "complexity_label": f"torch mlp {layer_label} {activation} a={alpha:g}",
                    }
                )
    if candidate_limit is not None:
        if candidate_limit < 1:
            raise ValueError("--candidate-limit must be positive when provided.")
        return candidates[:candidate_limit]
    return candidates


def candidate_id(candidate: dict[str, Any]) -> str:
    layers = "x".join(str(width) for width in candidate["hidden_layer_sizes"])
    return f"mlp_layers={layers}_act={candidate['activation']}_alpha={candidate['alpha']}"


class TorchMLP(_TorchModuleBase):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        output_dim: int,
        activation: str,
    ) -> None:
        require_torch()
        super().__init__()
        activation_cls = nn.ReLU if activation == "relu" else nn.Tanh
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_layer_sizes:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(activation_cls())
            previous = int(width)
        layers.append(nn.Linear(previous, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.network(x)


def set_torch_seed(seed: int) -> None:
    require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - not expected on local Mac, harmless elsewhere
        torch.cuda.manual_seed_all(seed)


def encoded_feature_count(
    train_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> int:
    preprocessor = build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)
    transformed = preprocessor.fit_transform(train_df[numeric_columns + categorical_columns])
    return int(transformed.shape[1])


def to_numpy_float(array: Any) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array)


def fit_torch_model(
    fit_df: pd.DataFrame,
    candidate: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    runtime: TorchRuntime,
    target_columns: list[str],
    max_iter: int,
    tolerance: float,
) -> tuple[Any, TorchMLP, float]:
    require_torch()
    set_torch_seed(int(candidate["seed"]))
    preprocessor = build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)
    x_np = to_numpy_float(preprocessor.fit_transform(fit_df[numeric_columns + categorical_columns]))
    y_np = fit_df[target_columns].to_numpy()
    x = torch.as_tensor(x_np, dtype=runtime.torch_dtype, device=runtime.device)
    y = torch.as_tensor(y_np, dtype=runtime.torch_dtype, device=runtime.device)
    model = TorchMLP(
        input_dim=x.shape[1],
        hidden_layer_sizes=tuple(int(width) for width in candidate["hidden_layer_sizes"]),
        output_dim=y.shape[1],
        activation=str(candidate["activation"]),
    ).to(device=runtime.device, dtype=runtime.torch_dtype)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=float(tolerance),
        tolerance_change=float(tolerance),
        line_search_fn="strong_wolfe",
    )
    mse_loss = nn.MSELoss()
    alpha = float(candidate["alpha"])
    n_samples = max(1, int(x.shape[0]))
    started = time.perf_counter()

    def closure() -> Any:
        optimizer.zero_grad()
        predictions = model(x)
        loss = mse_loss(predictions, y)
        if alpha:
            l2 = torch.zeros((), dtype=runtime.torch_dtype, device=runtime.device)
            for name, parameter in model.named_parameters():
                if "weight" in name:
                    l2 = l2 + torch.sum(parameter * parameter)
            loss = loss + (alpha * l2 / (2.0 * n_samples))
        loss.backward()
        return loss

    optimizer.step(closure)
    elapsed = time.perf_counter() - started
    return preprocessor, model, elapsed


def predict_torch_model(
    preprocessor: Any,
    model: TorchMLP,
    eval_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    runtime: TorchRuntime,
) -> np.ndarray:
    require_torch()
    x_np = to_numpy_float(preprocessor.transform(eval_df[numeric_columns + categorical_columns]))
    x = torch.as_tensor(x_np, dtype=runtime.torch_dtype, device=runtime.device)
    model.eval()
    with torch.no_grad():
        predictions = model(x).detach().cpu().numpy()
    return predictions


def fit_predict_torch(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    split_name: str,
    runtime: TorchRuntime,
    model_names: list[str] | tuple[str, ...] | None,
    max_iter: int,
    tolerance: float,
) -> tuple[Any, TorchMLP, pd.DataFrame, dict[str, Any], float]:
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    preprocessor, model, elapsed = fit_torch_model(
        fit_df,
        candidate,
        numeric_columns,
        categorical_columns,
        runtime,
        target_columns,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    predictions = predict_torch_model(preprocessor, model, eval_df, numeric_columns, categorical_columns, runtime)
    frame = prediction_frame(eval_df, predictions, split_name, model_names=names)
    metrics = routing_metrics(frame, names)
    return preprocessor, model, frame, metrics, elapsed


def evaluate_candidate_cv_torch(
    train_df: pd.DataFrame,
    candidate: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    cv_folds: int,
    runtime: TorchRuntime,
    model_names: list[str] | tuple[str, ...] | None,
    max_iter: int,
    tolerance: float,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    names = normalize_model_names(model_names)
    splits = group_kfold_splits(train_df, "paper_id", cv_folds)
    fold_frames: list[pd.DataFrame] = []
    elapsed_total = 0.0
    for fold_index, (fit_indices, holdout_indices) in enumerate(splits, start=1):
        fold_fit = train_df.iloc[fit_indices].copy()
        fold_eval = train_df.iloc[holdout_indices].copy()
        _, _, predictions, _, elapsed = fit_predict_torch(
            fold_fit,
            fold_eval,
            candidate,
            numeric_columns,
            categorical_columns,
            "cv",
            runtime,
            names,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        predictions["fold"] = fold_index
        fold_frames.append(predictions)
        elapsed_total += elapsed
    predictions_df = pd.concat(fold_frames, ignore_index=True)
    metrics = routing_metrics(predictions_df, names)
    metrics.update(candidate_metadata(candidate, candidate_id(candidate)))
    metrics["cv_folds"] = len(splits)
    metrics["fit_seconds"] = round(float(elapsed_total), 6)
    return metrics, predictions_df, elapsed_total


def evaluate_overfitting_curve_torch(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    candidates: list[dict[str, Any]],
    numeric_columns: list[str],
    categorical_columns: list[str],
    runtime: TorchRuntime,
    model_names: list[str] | tuple[str, ...] | None,
    max_iter: int,
    tolerance: float,
) -> pd.DataFrame:
    names = normalize_model_names(model_names)
    rows: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates, start=1):
        preprocessor, model, train_frame, train_metrics, elapsed = fit_predict_torch(
            train_df,
            train_df,
            candidate,
            numeric_columns,
            categorical_columns,
            "overfit_train",
            runtime,
            names,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        validation_predictions = predict_torch_model(
            preprocessor,
            model,
            validation_df,
            numeric_columns,
            categorical_columns,
            runtime,
        )
        validation_frame = prediction_frame(validation_df, validation_predictions, "overfit_validation", model_names=names)
        validation_metrics = routing_metrics(validation_frame, names)
        row = candidate_metadata(candidate, candidate_id(candidate))
        row.update(
            {
                "complexity_order": int(candidate.get("complexity_order", order)),
                "train_top1_hit_rate": train_metrics["top1_hit_rate"],
                "validation_top1_hit_rate": validation_metrics["top1_hit_rate"],
                "top1_hit_rate_gap": round(train_metrics["top1_hit_rate"] - validation_metrics["top1_hit_rate"], 6),
                "train_mean_regret": train_metrics["mean_regret"],
                "validation_mean_regret": validation_metrics["mean_regret"],
                "mean_regret_gap": round(validation_metrics["mean_regret"] - train_metrics["mean_regret"], 6),
                "fit_seconds": round(float(elapsed), 6),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["complexity_order", "candidate_id"])


def save_torch_bundle(
    path: Path,
    preprocessor: Any,
    model: TorchMLP,
    candidate: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_names: list[str],
    runtime: TorchRuntime,
) -> None:
    require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "preprocessor": preprocessor,
            "state_dict": model.cpu().state_dict(),
            "candidate": candidate,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "model_names": model_names,
            "runtime": runtime.to_dict(),
        },
        path,
    )
