"""Цикл обучения/валидации, общий для baseline, основной модели и эксперимента с водяным знаком."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (вероятности N×C, истинные метки N)."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(ys)


def _param_groups(model, kind, tcfg):
    """Для fine-tuning: у «головы» шаг обучения больше, чем у предобученного «тела»."""
    if kind == "baseline_cnn":
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": tcfg["lr"]}]
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return [{"params": body, "lr": tcfg["lr_backbone"]}, {"params": head, "lr": tcfg["lr_head"]}]


def fit(model, kind, train_ds, val_ds, tcfg, device, out_dir, num_workers=2, log_every=50):
    """
    Обучение с косинусным расписанием и выбором лучшей эпохи по Macro-F1 на валидации
    (тест в выборе модели НЕ участвует).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pin = device.type == "cuda"
    tl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=num_workers,
                    pin_memory=pin, drop_last=True, persistent_workers=num_workers > 0)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin)

    opt = torch.optim.AdamW(_param_groups(model, kind, tcfg), weight_decay=tcfg["weight_decay"])
    epochs = tcfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=epochs * len(tl), pct_start=0.15
    )
    crit = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))

    history, best_f1, best_state = [], -1.0, None
    model.to(device)
    for ep in range(1, epochs + 1):
        model.train()
        t0, run_loss, n = time.perf_counter(), 0.0, 0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
            if log_every and (it + 1) % log_every == 0:
                print(f"  ep{ep} it{it + 1}/{len(tl)} loss={run_loss / n:.3f}", flush=True)
        p, yv = predict_proba(model, vl, device)
        pred = p.argmax(1)
        rec = dict(epoch=ep, train_loss=run_loss / n, val_acc=accuracy_score(yv, pred),
                   val_macro_f1=f1_score(yv, pred, average="macro"), epoch_time_s=time.perf_counter() - t0)
        history.append(rec)
        print(f"[ep {ep}/{epochs}] loss={rec['train_loss']:.3f} val_acc={rec['val_acc']:.4f} "
              f"val_F1={rec['val_macro_f1']:.4f} ({rec['epoch_time_s']:.0f}s)", flush=True)
        if rec["val_macro_f1"] > best_f1:
            best_f1 = rec["val_macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history
