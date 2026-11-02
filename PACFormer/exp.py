import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from PACFormer.modules.PACFormer import Model
from PACFormer.modules.pangu_baseline import PanguBaseline
from src.metrics import CapacityMetricAccumulator
from src.tools import adjust_learning_rate


class _OptimizerBundle:
    """Expose multiple optimizers through the interface used by training."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {
            "optimizer_types": [
                optimizer.__class__.__name__ for optimizer in self.optimizers
            ],
            "optimizer_states": [
                optimizer.state_dict() for optimizer in self.optimizers
            ],
        }


_MUON_EXCLUDED_NAMES = (
    "enc_embedding",
    "value_embedding",
    "patch_embed",
    "w_pos",
    ".head.",
    "out_nwp",
    "out_pv_corr",
    "out_fused",
    "output_projection",
    "nwp_branch.et2e",
    "nwp_branch.token_to_sensor",
)


def _use_muon(name, parameter):
    """Select only 2D hidden-layer matrices, as required by torch.optim.Muon."""

    lowered = name.lower()
    return parameter.ndim == 2 and not any(
        excluded in lowered for excluded in _MUON_EXCLUDED_NAMES
    )


def _parameter_groups(named_parameters, predicate=None):
    grouped = {"pv": [], "vit": []}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if predicate is not None and not predicate(name, parameter):
            continue
        role = "vit" if name.startswith("nwp_branch.") else "pv"
        grouped[role].append(parameter)
    return [
        {"params": parameters, "group_name": role}
        for role, parameters in grouped.items()
        if parameters
    ]


def _build_optimizer(args, model):
    named_parameters = list(model.named_parameters())
    if args.optimizer == "adam":
        groups = _parameter_groups(named_parameters)
        for group in groups:
            group["lr"] = args.learning_rate
        return torch.optim.Adam(groups, weight_decay=args.weight_decay)

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "--optimizer muon requires a PyTorch build that provides torch.optim.Muon "
            "(the pinned torch==2.9.0 build does)."
        )

    muon_groups = _parameter_groups(named_parameters, _use_muon)
    adamw_groups = _parameter_groups(
        named_parameters, lambda name, parameter: not _use_muon(name, parameter)
    )
    if not muon_groups:
        raise ValueError("No eligible 2D hidden-layer parameters were found for Muon.")
    for group in [*muon_groups, *adamw_groups]:
        group["lr"] = args.learning_rate

    muon = torch.optim.Muon(
        muon_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        adjust_lr_fn=args.muon_adjust_lr,
    )
    optimizers = [muon]
    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW(
                adamw_groups,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        )

    muon_count = sum(
        parameter.numel() for group in muon_groups for parameter in group["params"]
    )
    adamw_count = sum(
        parameter.numel() for group in adamw_groups for parameter in group["params"]
    )
    print(
        "Optimizer: Muon+AdamW "
        f"| Muon parameters={muon_count:,} | AdamW parameters={adamw_count:,}"
    )
    return _OptimizerBundle(optimizers)


def build_model(args):
    model_class = PanguBaseline if args.model == "pangu" else Model
    return model_class(args).float().to(args.device)


def _move_batch(batch, device):
    return tuple(tensor.float().to(device, non_blocking=True) for tensor in batch)


def _forward(model, batch, device):
    x_enc, target, x_mark_enc, x_dec, x_mark_dec, nwp = _move_batch(batch, device)
    prediction = model(x_enc, x_mark_enc, x_dec, x_mark_dec, nwp)
    return prediction, target


def _criterion(name):
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError("loss must be either 'mse' or 'mae'.")


def _serializable_args(args):
    values = {}
    for key, value in vars(args).items():
        if isinstance(value, torch.device):
            values[key] = str(value)
        elif isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
        else:
            values[key] = value
    return values


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    pv_scaler,
    site_capacity,
    metric_threshold=1e-3,
):
    model.eval()
    accumulator = CapacityMetricAccumulator(site_capacity)
    for batch in loader:
        prediction, target = _forward(model, batch, device)
        prediction_scaled = prediction.cpu().numpy()
        target_scaled = target.cpu().numpy()
        prediction_inverse = pv_scaler.inverse_transform(prediction_scaled)
        target_inverse = pv_scaler.inverse_transform(target_scaled)
        valid_mask = target_inverse > metric_threshold
        accumulator.update(prediction_inverse, target_inverse, valid_mask)
    return accumulator.compute()


def train(args, model, data):
    run_dir = Path(args.checkpoints) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    criterion = _criterion(args.loss)
    optimizer = _build_optimizer(args, model)

    best_nmae = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.train_epochs + 1):
        epoch_learning_rates = {
            group["group_name"]: group["lr"] for group in optimizer.param_groups
        }
        model.train()
        losses = []
        for batch in data.train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = _forward(model, batch, args.device)
            loss = criterion(prediction, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(
            model,
            data.val_loader,
            args.device,
            data.pv_scaler,
            data.site_capacity,
            args.metric_threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "pv_learning_rate": epoch_learning_rates.get("pv"),
            "nwp_learning_rate": epoch_learning_rates.get("vit"),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        with (run_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
        print(
            f"Epoch {epoch:03d} | train_loss={epoch_record['train_loss']:.6f} "
            f"| val_nMAE={val_metrics['nMAE_percent']:.6f}% "
            f"| val_nRMSE={val_metrics['nRMSE_percent']:.6f}%"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_nMAE_percent": min(
                best_nmae, val_metrics["nMAE_percent"]
            ),
            "config": _serializable_args(args),
            "site_columns": data.site_columns,
            "nwp_variables": data.nwp_variables,
            "pv_mean": data.pv_scaler.mean,
            "pv_std": data.pv_scaler.std,
            "nwp_mean": data.nwp_scaler.mean,
            "nwp_std": data.nwp_scaler.std,
            "site_capacity": data.site_capacity,
            "capacity_column": data.capacity_column,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if val_metrics["nMAE_percent"] < best_nmae:
            best_nmae = val_metrics["nMAE_percent"]
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1

        next_learning_rates = adjust_learning_rate(optimizer, epoch + 1, args)
        rates = " | ".join(
            f"{name}={value:.6g}" for name, value in next_learning_rates.items()
        )
        print(f"Next LR | {rates}")

        if args.early_stopping and stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    return run_dir / "best.pt"


def test(args, model, data, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(
        model,
        data.test_loader,
        args.device,
        data.pv_scaler,
        data.site_capacity,
        args.metric_threshold,
    )

    output_dir = checkpoint_path.parent
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(
        f"Test | nMAE={metrics['nMAE_percent']:.6f}% "
        f"| nRMSE={metrics['nRMSE_percent']:.6f}% "
        f"| regional nMAE={metrics['regional_total_nMAE_percent']:.6f}% "
        f"| regional nRMSE={metrics['regional_total_nRMSE_percent']:.6f}%"
    )
    return metrics
