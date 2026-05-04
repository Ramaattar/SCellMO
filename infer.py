import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models import TransformerUserModel
from data import CellSequenceDataset


def strip_module_prefix(state_dict: dict) -> dict:
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def run_inference(cfg: dict):

    # --- Config ---
    icfg = cfg["inference"]
    device = torch.device(icfg["device"] if torch.cuda.is_available() else "cpu")
    topk = tuple(icfg["topk"])

    output_dir = Path(cfg["output"]["dir"]) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Vocab ---
    with open(cfg["data"]["vocab_path"]) as f:
        vocab = json.load(f)
    with open(cfg["data"]["area_vocab_path"]) as f:
        area_vocab = json.load(f)

    # --- Dataset ---
    test_ds = CellSequenceDataset(cfg["data"]["test_path"])
    test_loader = DataLoader(
        test_ds,
        batch_size=icfg["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # --- Model ---
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

    ckpt = torch.load(icfg["checkpoint"], map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    try:
        model.load_state_dict(ckpt, strict=False)
    except RuntimeError:
        model.load_state_dict(strip_module_prefix(ckpt))
    model.eval()

    counts_topk = {k: 0 for k in topk}
    reciprocal_ranks = []
    results = []
    total = 0
    total_time_sec = 0.0

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            (
                inputs, targets, user_ids, area, target_area, input_hours,
            ) = [t.to(device) for t in batch]

            starter.record()
            log_cell, log_area = model(
                inputs, area, user_ids, input_hours,
            )
            ender.record()
            torch.cuda.synchronize()
            total_time_sec += starter.elapsed_time(ender) / 1000

            logp_last = log_cell[:, -1, :]
            max_k = max(topk)
            _, topk_idx = torch.topk(logp_last, k=max_k, dim=1)
            for k in topk:
                counts_topk[k] += topk_idx[:, :k].eq(targets.view(-1, 1)).any(dim=1).sum().item()

            sorted_indices = torch.argsort(logp_last, dim=1, descending=True)
            matches = sorted_indices.eq(targets.view(-1, 1))
            rank_positions = torch.argmax(matches.int(), dim=1)
            rr = 1.0 / (rank_positions.float() + 1.0)
            reciprocal_ranks.append(rr.cpu().numpy())

            for i in range(inputs.size(0)):
                results.append({
                    "target": int(targets[i].item()),
                    "top1": int(topk_idx[i, 0].item()),
                    "rr": float(rr[i].item()),
                    "user_id": int(user_ids[i].item()),
                })

            total += inputs.size(0)

    # --- Summary ---
    topk_acc = {k: counts_topk[k] / total for k in topk}
    mrr = float(np.concatenate(reciprocal_ranks).mean())
    throughput = total / total_time_sec if total_time_sec > 0 else 0

  print(f"Samples: {total}")
    print(f"Throughput: {throughput:.1f} samples/sec")
    print(f"MRR: {mrr:.4f}")
    for k in topk:
        print(f"  Top-{k} Acc: {topk_acc[k]:.4f}")

    metrics = {"total_samples": total, "mrr": mrr, "topk_acc": topk_acc}

    with open(output_dir / "inference_results.pkl", "wb") as f:
        pickle.dump({"metrics": metrics, "per_sample": results}, f)

    pd.DataFrame(results).to_csv(output_dir / "inference_summary.csv", index=False)

    with open(output_dir / "inference_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)


if __name__ == "__main__":
    main()
