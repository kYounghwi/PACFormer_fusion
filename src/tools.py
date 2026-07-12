import math


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "type1":
        learning_rate = max(args.learning_rate * (0.8 ** (epoch - 1)), 2e-6)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    elif args.lradj == "type2":
        schedule = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
        if epoch in schedule:
            for group in optimizer.param_groups:
                group["lr"] = schedule[epoch]
    elif args.lradj == "type3":
        schedule = {
            15: 5e-5,
            30: 1e-5,
            60: 5e-6,
            90: 1e-6,
            100: 5e-7,
            120: 1e-7,
            150: 5e-8,
        }
        if epoch in schedule:
            for group in optimizer.param_groups:
                group["lr"] = schedule[epoch]
    elif args.lradj == "cosine":
        warmup_epochs = 8
        total_epochs = 150
        if epoch <= warmup_epochs:
            learning_rate = args.learning_rate * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            learning_rate = 2e-6 + 0.5 * (args.learning_rate - 2e-6) * (
                1 + math.cos(math.pi * progress)
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    elif args.lradj == "dual":
        pv_learning_rate = max(args.learning_rate * (0.92 ** (epoch - 1)), 2e-6)
        warmup_epochs = 1
        if epoch <= warmup_epochs:
            nwp_learning_rate = args.learning_rate * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(
                1, args.total_epochs - warmup_epochs
            )
            nwp_learning_rate = 4e-6 + 0.5 * (args.learning_rate - 4e-6) * (
                1 + math.cos(math.pi * progress)
            )

        for group in optimizer.param_groups:
            group["lr"] = (
                nwp_learning_rate
                if group.get("group_name", "pv") == "vit"
                else pv_learning_rate
            )
    else:
        raise ValueError(f"Unknown learning-rate schedule: {args.lradj}")

    return {
        group.get("group_name", "pv"): group["lr"] for group in optimizer.param_groups
    }
