import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from PACFormer.modules.PACFormer import Model
from PACFormer.modules.pangu_baseline import PanguBaseline
from src.metrics import CapacityMetricAccumulator
from src.tools import adjust_learning_rate


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
    pv_parameters, nwp_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("nwp_branch."):
            nwp_parameters.append(parameter)
        else:
            pv_parameters.append(parameter)
    parameter_groups = []
    if pv_parameters:
        parameter_groups.append(
            {"params": pv_parameters, "lr": args.learning_rate, "group_name": "pv"}
        )
    if nwp_parameters:
        parameter_groups.append(
            {"params": nwp_parameters, "lr": args.learning_rate, "group_name": "vit"}
        )
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=args.weight_decay)

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
