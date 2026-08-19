"""Unit tests for the theorem-to-code path. Run: python -m cus.tests"""
import numpy as np
from cus import crc


def test_prop2_reduces_to_unweighted():
    """At w == 1 everywhere, Proposition 2 must equal Eq. (4) exactly."""
    rng = np.random.default_rng(0)
    lambdas = np.linspace(0, 1, 200)
    for _ in range(50):
        n = rng.integers(20, 300)
        s = rng.random(n)
        wrong = rng.random(n) < 0.3
        losses = crc.commit_error_losses(s, wrong, lambdas)
        alpha = float(rng.uniform(0.05, 0.3))
        lam_u = crc.lhat_unweighted(losses, lambdas, alpha)
        lam_p = crc.lhat_prop2(losses, lambdas, alpha,
                               np.ones(n), np.ones(7))
        assert np.allclose(lam_p, lam_u), (lam_u, lam_p)


def test_prop2_monotone_in_test_weight():
    """Bigger w(x) charges a bigger pseudo-loss, so lam_hat(x) rises."""
    rng = np.random.default_rng(1)
    lambdas = np.linspace(0, 1, 200)
    s = rng.random(500)
    wrong = rng.random(500) < 0.3
    losses = crc.commit_error_losses(s, wrong, lambdas)
    lams = crc.lhat_prop2(losses, lambdas, 0.1, np.ones(500),
                          np.array([0.1, 1.0, 5.0, 50.0]))
    assert np.all(np.diff(lams) >= 0), lams


if __name__ == "__main__":
    test_prop2_reduces_to_unweighted()
    test_prop2_monotone_in_test_weight()
    print("[tests] prop2 == unweighted at w=1: PASS")
    print("[tests] prop2 monotone in w(x):     PASS")
