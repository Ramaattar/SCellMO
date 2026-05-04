import torch
import torch.distributed as dist
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    rank=0,
    world_size=1,
    use_area_loss=True,
    area_loss_weight=0.3,
    max_grad_norm=1.0,
):
    
    model.train()
    total_loss = 0.0
    total_loss_cell = 0.0
    total_loss_area = 0.0
    correct_cell = 0
    correct_area = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training Epoch", disable=(rank != 0)):
        (
            inputs, targets, user_ids, area, target_area, input_hours,
        ) = [t.to(device) for t in batch]

        optimizer.zero_grad()

        log_cell, log_area = model(
            inputs, area, user_ids, input_hours,
        )

        loss_cell = loss_fn(log_cell[:, -1, :], targets)
        loss_area = loss_fn(log_area[:, -1, :], target_area)

        loss = loss_cell + area_loss_weight * loss_area if use_area_loss else loss_cell
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss_cell += loss_cell.item()
        total_loss_area += loss_area.item()
        total_loss += loss.item()
        correct_cell += (log_cell[:, -1, :].argmax(dim=1) == targets).sum().item()
        correct_area += (log_area[:, -1, :].argmax(dim=1) == target_area).sum().item()
        total += targets.size(0)

    metrics = torch.tensor(
        [total_loss, correct_cell, correct_area, 0, total,
         total_loss_cell, total_loss_area, 0],
        dtype=torch.float32, device=device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    (
        total_loss, correct_cell, correct_area, _, total,
        total_loss_cell, total_loss_area, _,
    ) = metrics.tolist()

    n_batches = len(dataloader) * world_size
    avg_loss_cell = total_loss_cell / n_batches
    avg_loss_area = total_loss_area / n_batches
    avg_loss = total_loss / n_batches
    acc_cell = correct_cell / total if total > 0 else 0
    acc_area = correct_area / total if total > 0 else 0

    return avg_loss_cell, avg_loss_area, 0, avg_loss, acc_cell, acc_area, 0


def validate(
    model,
    dataloader,
    loss_fn,
    device,
    topk=(1, 3, 5, 7, 10),
    rank=0,
    world_size=1,
    use_area_loss=True,
):
    model.eval()
    total_loss = 0.0
    total_loss_cell = 0.0
    total_loss_area = 0.0
    correct_area = 0
    total = 0
    total_area = 0
    topk_hits = {k: 0 for k in topk}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=(rank != 0)):
            (
                inputs, targets, user_ids, area, target_area, input_hours,
            ) = [t.to(device) for t in batch]

            log_cell, log_area = model(
                inputs, area, user_ids, input_hours,
            )

            loss_cell = loss_fn(log_cell[:, -1, :], targets)
            loss_area = loss_fn(log_area[:, -1, :], target_area)

            total_loss += loss_cell.item()
            total_loss_cell += loss_cell.item()
            total_loss_area += loss_area.item()

            pred_area = log_area[:, -1, :].argmax(dim=1)
            correct_area += (pred_area == target_area).sum().item()

            _, topk_preds = torch.topk(log_cell[:, -1, :], max(topk), dim=1)
            for k in topk:
                topk_hits[k] += topk_preds[:, :k].eq(targets.view(-1, 1)).any(dim=1).sum().item()

            total += targets.size(0)
            total_area += target_area.size(0)

    metrics_list = [total_loss, total_loss_cell, total_loss_area, correct_area, total, total_area]
    metrics_list += [topk_hits[k] for k in topk]
    metrics = torch.tensor(metrics_list, dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics = metrics.tolist()

    total_loss = metrics[0]
    total_loss_cell = metrics[1]
    total_loss_area = metrics[2]
    correct_area = metrics[3]
    total = int(metrics[4])
    total_area = int(metrics[5])
    idx = 6
    for k in topk:
        topk_hits[k] = metrics[idx]
        idx += 1

    n_batches = len(dataloader) * world_size
    avg_loss = total_loss / n_batches
    avg_loss_cell = total_loss_cell / n_batches
    avg_loss_area = total_loss_area / n_batches
    area_acc = correct_area / total_area if total_area > 0 else 0
    accs = {f"top{k}": topk_hits[k] / total for k in topk}

    return avg_loss, avg_loss_cell, avg_loss_area, 0, area_acc, 0, accs

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        return [group["lr"] for group in self.optimizer.param_groups]
