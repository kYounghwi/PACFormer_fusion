import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from PACFormer.modules.PACFormer import Model
from src.metrics import SiteMetricAccumulator
from src.tools import adjust_learning_rate


def build_model(args):
    return Model(args).float().to(args.device)


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
    metric_threshold=1e-3,
    collect_outputs=False,
):
    model.eval()
    predictions = [] if collect_outputs else None
    targets = [] if collect_outputs else None
    accumulator = None
    for batch in loader:
        prediction, target = _forward(model, batch, device)
        prediction_scaled = prediction.cpu().numpy()
        target_scaled = target.cpu().numpy()
        target_inverse = pv_scaler.inverse_transform(target_scaled)
        valid_mask = target_inverse > metric_threshold

        if accumulator is None:
            accumulator = SiteMetricAccumulator(target_scaled.shape[-1])
        accumulator.update(prediction_scaled, target_scaled, valid_mask)

        if collect_outputs:
            predictions.append(pv_scaler.inverse_transform(prediction_scaled))
            targets.append(target_inverse)

    if accumulator is None:
        raise ValueError("Cannot evaluate an empty data loader.")
    metrics, site_metrics = accumulator.compute()
    prediction = np.concatenate(predictions, axis=0) if collect_outputs else None
    target = np.concatenate(targets, axis=0) if collect_outputs else None
    return metrics, prediction, target, site_metrics


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
    optimizer = torch.optim.Adam(
        [
            {"params": pv_parameters, "lr": args.learning_rate, "group_name": "pv"},
            {
                "params": nwp_parameters,
                "lr": args.learning_rate,
                "group_name": "vit",
            },
        ],
        weight_decay=args.weight_decay,
    )

    best_mae = float("inf")
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

        val_metrics, _, _, _ = evaluate(
            model,
            data.val_loader,
            args.device,
            data.pv_scaler,
            args.metric_threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "pv_learning_rate": epoch_learning_rates["pv"],
            "nwp_learning_rate": epoch_learning_rates["vit"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        with (run_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
        print(
            f"Epoch {epoch:03d} | train_loss={epoch_record['train_loss']:.6f} "
            f"| val_mae={val_metrics['mae']:.6f} | val_rmse={val_metrics['rmse']:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": min(best_mae, val_metrics["mae"]),
            "config": _serializable_args(args),
            "site_columns": data.site_columns,
            "nwp_variables": data.nwp_variables,
            "pv_mean": data.pv_scaler.mean,
            "pv_std": data.pv_scaler.std,
            "nwp_mean": data.nwp_scaler.mean,
            "nwp_std": data.nwp_scaler.std,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1

        next_learning_rates = adjust_learning_rate(optimizer, epoch + 1, args)
        print(
            f"Next LR | pv={next_learning_rates['pv']:.6g} "
            f"| nwp={next_learning_rates['vit']:.6g}"
        )

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
    metrics, prediction, target, site_metrics = evaluate(
        model,
        data.test_loader,
        args.device,
        data.pv_scaler,
        args.metric_threshold,
        collect_outputs=True,
    )

    output_dir = checkpoint_path.parent
    np.save(output_dir / "test_prediction.npy", prediction)
    np.save(output_dir / "test_target.npy", target)
    np.savez(output_dir / "test_site_metrics.npz", **site_metrics)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    site_records = []
    for index, site_name in enumerate(data.site_columns):
        site_records.append(
            {
                "site": site_name,
                "mae": None
                if np.isnan(site_metrics["mae"][index])
                else float(site_metrics["mae"][index]),
                "mse": None
                if np.isnan(site_metrics["mse"][index])
                else float(site_metrics["mse"][index]),
                "rmse": None
                if np.isnan(site_metrics["rmse"][index])
                else float(site_metrics["rmse"][index]),
                "valid_count": int(site_metrics["valid_count"][index]),
                "valid_fraction": float(site_metrics["valid_fraction"][index]),
            }
        )
    with (output_dir / "test_site_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(site_records, stream, indent=2)
    print(
        f"Test | MAE={metrics['mae']:.6f} | MSE={metrics['mse']:.6f} "
        f"| RMSE={metrics['rmse']:.6f}"
    )
    return metrics
