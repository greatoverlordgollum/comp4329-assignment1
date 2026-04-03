"""
train_utils.py — Low-level training utilities used by train().
"""

import os

import numpy as np
import torch
from tqdm import tqdm


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]


def train_single_epoch(model, optimizer, scheduler, data_iter,
                       steps, grad_clip, loss_fn, device,
                       global_step: int = 0,
                       warmup_steps: int = 0,
                       accumulate_grad_steps: int = 4,
                       ema=None) -> float:
    """
    Run one block of `steps` training iterations consuming from `data_iter`.
    Returns the mean loss over this block.
    """
    model.train()
    loss_list = []

    base_lrs = [group.get("base_lr", group["lr"]) for group in optimizer.param_groups]
    optimizer.zero_grad(set_to_none=True)

    for i in tqdm(range(steps), total=steps):
        # Optional linear warmup to avoid unstable early updates.
        if warmup_steps > 0:
            step_num = global_step + i + 1
            if step_num <= warmup_steps:
                warm = step_num / float(warmup_steps)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * warm

        accumulated_loss = 0.0
        for _ in range(accumulate_grad_steps):
            Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
            Cwid, Ccid = Cwid.to(device), Ccid.to(device)
            Qwid, Qcid = Qwid.to(device), Qcid.to(device)
            y1, y2     = y1.to(device),   y2.to(device)

            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            loss   = loss_fn(p1, p2, y1, y2)
            
            # Normalize loss for accumulation
            loss = loss / accumulate_grad_steps
            loss.backward()
            accumulated_loss += float(loss.item())

        loss_list.append(accumulated_loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        if ema is not None:
            ema.update(model)

        optimizer.zero_grad(set_to_none=True)
        
        # Only step the scheduler if we are past the warmup phase
        if warmup_steps == 0 or (global_step + i + 1) > warmup_steps:
            scheduler.step()

    mean_loss = float(np.mean(loss_list))
    print(f"STEP {global_step + steps:8d}  loss {mean_loss:8f}\n")
    return mean_loss


def save_checkpoint(save_dir, ckpt_name, model, optimizer, scheduler,
                    step, best_f1, best_em, config):
    """Save model, optimizer, scheduler state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step":            step,
        "best_f1":         best_f1,
        "best_em":         best_em,
        "config":          config,
    }
    torch.save(payload, os.path.join(save_dir, ckpt_name))
