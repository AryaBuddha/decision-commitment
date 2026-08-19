"""APE/CFR audit for the cross-environment hypothesis. Runs AFTER an
environment's gates pass and BEFORE any sweep of that environment; the
frozen recipe lives in registrations/cross_environment_hypothesis.json.

APE(level) = E_target[ m(s, dec, X) * 1{s >= lambda0} ] - alpha, with m an
audit model fit on labelled SOURCE draws only and the expectation over
UNLABELLED target covariates: exactly the data a deployment has.

Run: python experiments/gates/audit_ape.py --env tickets
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cus import crc                                   # noqa: E402
from cus.envs.family import GenEnv, SPECS             # noqa: E402
from run_gates_family import BETA_GRIDS               # noqa: E402

ALPHA = 0.10
N_AUDIT = 60000
N_LAMBDA0 = 50000
N_TARGET = 200000


def main() -> None:
    name = sys.argv[sys.argv.index("--env") + 1]
    spec = SPECS[name]
    env = GenEnv.induce(name)
    root = pathlib.Path(__file__).resolve().parents[2]

    from sklearn.linear_model import LogisticRegression
    r1 = np.random.default_rng([spec.env_seed, 3001])
    ct = env.case_table(r1, N_AUDIT)
    dec, _ = env.route(ct.X)
    Z = np.column_stack([ct.s, dec, ct.X, dec[:, None] * ct.X])
    m = LogisticRegression(max_iter=2000).fit(Z, ct.wrong)

    r2 = np.random.default_rng([spec.env_seed, 3002])
    cal = env.case_table(r2, N_LAMBDA0)
    lambdas = np.linspace(0.0, 1.0, 400)
    losses = crc.commit_error_losses(cal.s, cal.wrong, lambdas)
    lam0 = crc.lhat_unweighted(losses, lambdas, ALPHA)

    rows = []
    for li, beta in enumerate(BETA_GRIDS[name]):
        r3 = np.random.default_rng([spec.env_seed, 3100 + li])
        Xt, _ = env.draw_instances(r3, N_TARGET, beta, spec.primary_tilt)
        dect, st = None, None
        dect, st = env.route(Xt)
        Zt = np.column_stack([st, dect, Xt, dect[:, None] * Xt])
        p = m.predict_proba(Zt)[:, 1]
        commit = st >= lam0
        vals = p * commit
        ape = float(vals.mean() - ALPHA)
        ape_se = float(vals.std() / np.sqrt(N_TARGET))
        cfr = float(commit.mean())
        cfr_se = float(commit.std() / np.sqrt(N_TARGET))
        rows.append({"beta": beta, "APE": ape, "APE_se": ape_se,
                     "CFR": cfr, "CFR_se": cfr_se})
        print(f"[ape] {name} beta={beta:<7} APE={ape:+.4f}±{ape_se:.4f} "
              f"CFR={cfr:.4f}")

    reg = root / "registrations" / "cross_environment_hypothesis.json"
    doc = json.loads(reg.read_text())
    doc["audits"].append({
        "date": "2026-08-19",
        "environment": name,
        "lambda0": float(lam0),
        "gate_report": f"see registrations/env_{name}.json gates key",
        "levels": rows,
        "status": "filed BEFORE any sweep of this environment",
    })
    reg.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[ape] audit appended to {reg.name}")


if __name__ == "__main__":
    main()
