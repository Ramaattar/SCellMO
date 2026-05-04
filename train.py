import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from models import TransformerUserModel
from training import train_one_epoch, validate, WarmupLR
from training.utils import log
from data import CellSequenceDataset


def main_worker(rank: int, world_size: int, cfg: dict):
    print(f"Starting process {rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    with open(cfg["data"]["vocab_path"]) as f:
        vocab = json.load(f)
    with open(cfg["data"]["area_vocab_path"]) as f:
        area_vocab = json.load(f)

    mcfg = cfg["model"]
    model = TransformerUserModel(
        vocab_size=len(vocab),
        area_vocab_size=len(area_vocab),
        embedding_dim=mcfg["embedding_dim"],
        user_embedding_dim=mcfg["user_embedding_dim"],
        area_embedding_dim=mcfg["area_embedding_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        nhead=mcfg["nhead"],
        max_seq_len=mcfg["max_seq_len"],
        num_users=mcfg["num_users"],
        dropout=mcfg["dropout"],
    ).to(device)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True,
    )

    tcfg = cfg["training"]
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )

    warmup_steps = tcfg["warmup_steps"]
    warmup_scheduler = WarmupLR(optimizer, warmup_steps=warmup_steps)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=tcfg["scheduler_factor"],
        patience=tcfg["scheduler_patience"],
    )

    train_dataset = CellSequenceDataset(cfg["data"]["train_path"])
    val_dataset = CellSequenceDataset(cfg["data"]["val_path"])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=tcfg["batch_size"],
        sampler=train_sampler, num_workers=tcfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tcfg["batch_size"],
        sampler=val_sampler, num_workers=tcfg["num_workers"], pin_memory=True,
    )

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_val_area_loss = float("inf")
    use_area_loss = True
    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(tcfg["epochs"]):
        train_sampler.set_epoch(epoch)

        train_loss_cell, train_loss_area, _, train_loss, train_acc_cell, train_acc_area, _ = (
            train_one_epoch(
                model, train_loader, loss_fn, optimizer, device,
                rank=rank, world_size=world_size,
                use_area_loss=use_area_loss,
                area_loss_weight=tcfg["area_loss_weight"],
                max_grad_norm=tcfg["max_grad_norm"],
            )
        )

        val_loss, val_cell_loss, val_area_loss, _, area_acc, _, val_acc = validate(
            model, val_loader, loss_fn, device,
            rank=rank, world_size=world_size,
            use_area_loss=use_area_loss,
        )

        use_area_loss = val_area_loss <= prev_val_area_loss
        prev_val_area_loss = val_area_loss

        warmup_scheduler.step()
        if epoch >= warmup_steps:
            plateau_scheduler.step(val_cell_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr <= tcfg["lr_min_threshold"]:
            if rank == 0:
                log(f"LR reached {current_lr:.1e}. Early stopping at epoch {epoch + 1}.", output_dir)
            break

        if rank == 0:
            log(f"Epoch {epoch + 1}/{tcfg['epochs']}", output_dir)
            log(f"  Train — Loss: {train_loss:.4f}  Cell Loss: {train_loss_cell:.4f}  "
                f"Area Loss: {train_loss_area:.4f}", output_dir)
            log(f"  Train — Cell Acc: {train_acc_cell:.4f}  Area Acc: {train_acc_area:.4f}", output_dir)
            log(f"  Val   — Loss: {val_loss:.4f}  Cell Loss: {val_cell_loss:.4f}  "
                f"Area Loss: {val_area_loss:.4f}", output_dir)
            log(f"  Val   — Area Acc: {area_acc:.4f}  "
                + "  ".join(f"Acc@{k}: {val_acc[f'top{k}']:.4f}" for k in [1, 3, 5, 7, 10]),
                output_dir)
            log(f"  LR: {current_lr:.6f}\n", output_dir)

            if val_acc["top1"] > best_val_acc:
                best_val_acc = val_acc["top1"]
                torch.save(model.module.state_dict(), output_dir / "best_acc_model.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), output_dir / "best_loss_model.pth")

    if rank == 0:
        torch.save(model.module.state_dict(), output_dir / "final_model.pth")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Mobility Transformer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # DDP environment
    dcfg = cfg["distributed"]
    os.environ["MASTER_ADDR"] = dcfg["master_addr"]
    os.environ["MASTER_PORT"] = dcfg["master_port"]
    os.environ["CUDA_VISIBLE_DEVICES"] = dcfg["gpus"]

    world_size = len(dcfg["gpus"].split(","))
    mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size)


if __name__ == "__main__":
    main()
