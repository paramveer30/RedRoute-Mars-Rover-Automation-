import numpy as np

from ml.calibrate import ece, fit_temperature, flatten


def _sample(n=60000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2.0, (n, k))
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    labels = np.array([rng.choice(k, p=r) for r in p])
    return logits, labels


def test_ece_near_zero_when_calibrated():
    logits, labels = _sample()
    assert ece(logits, labels) < 0.02


def test_fit_recovers_temperature():
    logits, labels = _sample()
    t = fit_temperature(logits / 2.5, labels)
    assert abs(t - 1 / 2.5) < 0.05 * (1 / 2.5) + 0.02
    t = fit_temperature(logits * 0.5, labels)
    assert abs(t - 0.5) < 0.05


def test_ignores_255():
    logits, labels = _sample(2000)
    lg4 = logits.T.reshape(1, 4, 1, -1)
    lb = labels.reshape(1, 1, -1).copy()
    lb2 = np.concatenate([lb, np.full((1, 1, 500), 255)], axis=2)
    lg2 = np.concatenate([lg4, np.random.default_rng(1).normal(size=(1, 4, 1, 500))], axis=3)
    a, b = flatten(lg4, lb)
    c, d = flatten(lg2, lb2)
    assert len(d) == 2000 and np.allclose(a, c) and (b == d).all()
