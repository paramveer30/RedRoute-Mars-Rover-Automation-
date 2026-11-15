"""Fit a scalar softmax temperature on validation logits and report ECE/NLL (val fit only)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from world.paths import CALIBRATION, LOGITS_DIR, RUNS_DIR

N_BINS = 15


def flatten(logits: np.ndarray, labels: np.ndarray):
    """(N,4,H,W), (N,H,W) -> (M,4) float32 logits and (M,) labels over non-255 pixels."""
    lg = np.moveaxis(np.asarray(logits), 1, -1).reshape(-1, logits.shape[1]).astype(np.float32)
    lb = np.asarray(labels).reshape(-1)
    keep = lb != 255
    return lg[keep], lb[keep].astype(np.int64)


def nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    lg = torch.from_numpy(np.asarray(logits, np.float64)) / T
    return float(torch.nn.functional.cross_entropy(lg, torch.from_numpy(labels)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    lg = torch.from_numpy(np.asarray(logits, np.float64))
    lb = torch.from_numpy(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, tolerance_grad=1e-10,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def _conf_correct(logits, labels, T):
    p = torch.softmax(torch.from_numpy(np.asarray(logits, np.float64)) / T, dim=1).numpy()
    return p.max(1), p.argmax(1) == labels


def reliability(logits, labels, T=1.0, n_bins=N_BINS):
    conf, ok = _conf_correct(logits, labels, T)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    sa = np.bincount(idx, weights=ok.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sc / count, sa / count, count


def ece(logits, labels, T=1.0, n_bins=N_BINS) -> float:
    mc, acc, count = reliability(logits, labels, T, n_bins)
    m = count > 0
    return float((count[m] / count.sum() * np.abs(acc[m] - mc[m])).sum())


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plot(val, test, T, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for ax, (name, (lg, lb)) in zip(axes, (("validation", val), ("test", test))):
        for t, label, color in ((1.0, "before (T=1)", "tab:red"), (T, f"after (T={T:.3f})", "tab:blue")):
            mc, acc, count = reliability(lg, lb, t)
            m = count > 0
            ax.plot(mc[m], acc[m], "o-", color=color, label=f"{label}, ECE={ece(lg, lb, t):.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("confidence")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    vpath, tpath = LOGITS_DIR / "val_logits.npz", LOGITS_DIR / "test_logits.npz"
    with np.load(vpath) as z:
        val = flatten(z["logits"], z["labels"])
    with np.load(tpath) as z:
        test = flatten(z["logits"], z["labels"])
    T = fit_temperature(*val)
    out = {"T": T}
    for name, (lg, lb) in (("val", val), ("test", test)):
        out[f"ece_{name}_before"] = ece(lg, lb, 1.0)
        out[f"ece_{name}_after"] = ece(lg, lb, T)
    for name, (lg, lb) in (("val", val), ("test", test)):
        out[f"nll_{name}_before"] = nll(lg, lb, 1.0)
        out[f"nll_{name}_after"] = nll(lg, lb, T)
    out["n_pixels_val"] = int(len(val[1]))
    out["n_pixels_test"] = int(len(test[1]))
    out["val_logits_sha256"] = sha256(vpath)
    out["test_logits_sha256"] = sha256(tpath)
    with np.load(vpath) as z:
        for k in ("weights_sha256", "weights_hash", "fingerprint"):
            if k in z.files:
                out[f"archive_{k}"] = str(z[k])
    CALIBRATION.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    plot(val, test, T, RUNS_DIR / "calib" / "reliability.png")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
