"""Обучение SRCNN (×2 или ×4) с валидацией по PSNR/SSIM.

  python train.py --scale 2                 # ~10–15 мин на RTX 3050 Ti (20 000 итераций)
  python train.py --scale 4
  python train.py --scale 4 --fast          # быстрый прогон (~3–5 мин) для проверки
  python train.py --scale 4 --resume        # продолжить с последнего чекпоинта

Результаты:
  checkpoints/srcnn_x{S}.pth        — лучшая модель по PSNR на валидации (её использует upscale.py)
  checkpoints/srcnn_x{S}_last.pth   — последнее состояние (для --resume)
  runs/srcnn_x{S}/train_log.csv     — loss, lr, PSNR/SSIM на валидации по итерациям
  runs/srcnn_x{S}/curves.png        — графики обучения
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from srlib.data import SRPatchDataset
from srlib.imgproc import bgr2y, imread, imresize, modcrop, to_image, to_tensor
from srlib.metrics import psnr, ssim
from srlib.models import SRCNN, count_params

ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "train_y.npy"))
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits.json"))
    ap.add_argument("--iters", type=int, default=20000, help="число итераций (батчей)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patch", type=int, default=96, help="размер HR-патча")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate слоёв 1–2")
    ap.add_argument("--lr-last", type=float, default=1e-4, help="learning rate последнего слоя (в 10 раз меньше, как в статье)")
    ap.add_argument("--n1", type=int, default=64)
    ap.add_argument("--n2", type=int, default=32)
    ap.add_argument("--f1", type=int, default=9)
    ap.add_argument("--f2", type=int, default=5)
    ap.add_argument("--f3", type=int, default=5)
    ap.add_argument("--residual", action="store_true", help="глобальная skip-связь (учить только остаток)")
    ap.add_argument("--init", choices=["kaiming", "paper"], default="kaiming")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true", help="отключить mixed precision на GPU")
    ap.add_argument("--fast", action="store_true", help="короткое обучение для проверки: 3000 итераций")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    args = ap.parse_args()
    if args.fast:
        args.iters, args.val_every = 3000, 500
    return args


def load_val(paths, scale):
    """HR (Y, modcrop) → LR → bicubic↑ : вход сети и эталон для валидации."""
    items = []
    for p in paths:
        hr = modcrop(np.clip(bgr2y(imread(p)).round(), 0, 255).astype(np.uint8), scale)
        h, w = hr.shape
        lr = imresize(hr, (w // scale, h // scale))
        items.append((Path(p).name, imresize(lr, (w, h)), hr))
    return items


@torch.no_grad()
def validate(net, items, scale, device, amp):
    net.eval()
    ps, ss = [], []
    for _, inp, hr in items:
        x = to_tensor(inp).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            y = net(x)
        sr = np.clip(to_image(y.float()).round(), 0, 255)
        ps.append(psnr(sr, hr, scale))
        ss.append(ssim(sr, hr, scale))
    net.train()
    return float(np.mean(ps)), float(np.mean(ss))


def plot_curves(log_csv, out_png, bicubic_psnr):
    from srlib import plotstyle
    from srlib.plotstyle import COLORS, INK2, plt

    plotstyle.apply()
    rows = list(csv.DictReader(open(log_csv, encoding="utf-8")))
    it = np.array([int(r["iter"]) for r in rows])
    loss = np.array([float(r["train_loss"]) for r in rows])
    vit = [int(r["iter"]) for r in rows if r["val_psnr"]]
    vp = [float(r["val_psnr"]) for r in rows if r["val_psnr"]]
    vs = [float(r["val_ssim"]) for r in rows if r["val_ssim"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    k = max(len(loss) // 40, 1)
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    ax[0].plot(it, loss, color=COLORS["srcnn"], alpha=0.25, lw=1)
    ax[0].plot(it[k - 1:], smooth, color=COLORS["srcnn"], lw=2)
    ax[0].set(title="Функция потерь MSE (train)", xlabel="итерация", yscale="log")
    ax[1].plot(vit, vp, "o-", color=COLORS["srcnn"], label="SRCNN")
    ax[1].axhline(bicubic_psnr, color=COLORS["bicubic"], ls="--", lw=1.5, label="Бикубическая")
    ax[1].annotate(f"{max(vp):.2f} дБ", (vit[int(np.argmax(vp))], max(vp)), textcoords="offset points",
                   xytext=(0, 8), ha="center", color=INK2, fontsize=9)
    ax[1].set(title="PSNR на валидации (Y), дБ", xlabel="итерация")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    s = args.scale
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "runs" / f"srcnn_x{s}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / f"srcnn_x{s}.pth", out_dir / f"srcnn_x{s}_last.pth"

    # ---------------- данные
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    ds = SRPatchDataset(args.data, s, args.patch, length=args.iters * args.batch, seed=args.seed)
    workers = args.workers if device.type == "cuda" else min(args.workers, 2)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=workers, pin_memory=device.type == "cuda",
                    drop_last=True, persistent_workers=workers > 0)
    val_items = load_val(splits["val"], s)
    bic_p = float(np.mean([psnr(i, h, s) for _, i, h in val_items]))
    bic_s = float(np.mean([ssim(i, h, s) for _, i, h in val_items]))

    # ---------------- модель, оптимизатор (Adam; последний слой с меньшим lr, как в статье SRCNN)
    net = SRCNN(args.n1, args.n2, args.f1, args.f2, args.f3, residual=args.residual, init=args.init).to(device)
    last_ids = {id(p) for p in net.last_layer_params()}
    opt = torch.optim.Adam([
        {"params": [p for p in net.parameters() if id(p) not in last_ids], "lr": args.lr},
        {"params": net.last_layer_params(), "lr": args.lr_last},
    ], betas=(0.9, 0.999))
    # косинусное затухание lr до 1% от начального
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_it, best = 0, -1.0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_it, best = ck["iter"], ck.get("best_psnr", -1.0)
        print(f"Продолжаю с итерации {start_it}")

    print(f"Устройство: {device} (AMP: {amp}) | SRCNN {args.f1}-{args.f2}-{args.f3}, "
          f"{args.n1}/{args.n2} фильтров, {count_params(net):,} параметров | x{s}")
    print(f"Train: {ds.n} фрагментов | Val: {len(val_items)} изобр. | бикубическая на val: "
          f"PSNR {bic_p:.2f} дБ, SSIM {bic_s:.4f}")

    log_csv = run_dir / "train_log.csv"
    new_log = not (args.resume and log_csv.exists())
    logf = open(log_csv, "w" if new_log else "a", newline="", encoding="utf-8")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["iter", "train_loss", "train_psnr", "lr", "val_psnr", "val_ssim", "elapsed_s"])

    def save(path, it, best_psnr, full=False):
        ck = {"model": "srcnn", "scale": s, "config": net.config, "state_dict": net.state_dict(),
              "iter": it, "best_psnr": best_psnr, "bicubic_psnr": bic_p, "args": vars(args)}
        if full:
            ck.update(optimizer=opt.state_dict(), scheduler=sched.state_dict())
        torch.save(ck, path)

    net.train()
    it, t0, run_loss, n_loss = start_it, time.time(), 0.0, 0
    log_every = max(args.val_every // 10, 10)
    data_iter = iter(dl)
    while it < args.iters:
        try:
            inp, hr = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            inp, hr = next(data_iter)
        inp, hr = inp.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = F.mse_loss(net(inp).float(), hr)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        it += 1
        run_loss += loss.item()
        n_loss += 1

        if it % log_every == 0 or it == args.iters:
            avg = run_loss / n_loss
            vp = vs = ""
            if it % args.val_every == 0 or it == args.iters:
                vp, vs = validate(net, val_items, s, device, amp)
                mark = ""
                if vp > best:
                    best = vp
                    save(best_path, it, best)
                    mark = "  ← лучшая"
                print(f"[{it:6d}/{args.iters}] loss {avg:.5f} | val PSNR {vp:.2f} дБ "
                      f"({vp - bic_p:+.2f} к бикубич.) SSIM {vs:.4f} | {time.time() - t0:.0f} c{mark}")
                save(last_path, it, best, full=True)
            writer.writerow([it, f"{avg:.6f}", f"{10 * math.log10(1 / max(avg, 1e-12)):.3f}",
                             f"{opt.param_groups[0]['lr']:.2e}",
                             f"{vp:.4f}" if vp != "" else "", f"{vs:.5f}" if vs != "" else "",
                             f"{time.time() - t0:.1f}"])
            logf.flush()
            run_loss, n_loss = 0.0, 0
    logf.close()
    plot_curves(log_csv, run_dir / "curves.png", bic_p)
    summary = {"scale": s, "best_val_psnr": best, "bicubic_val_psnr": bic_p, "bicubic_val_ssim": bic_s,
               "iters": args.iters, "minutes": (time.time() - t0) / 60, "device": str(device)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово. Лучший PSNR на валидации: {best:.2f} дБ (бикубическая {bic_p:.2f} дБ) → {best_path}")


if __name__ == "__main__":
    main()
