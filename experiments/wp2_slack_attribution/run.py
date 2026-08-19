"""WP2: the registered slack attribution (wp2_slack_attribution).

Consumes the v3 validation's per-draw parts and runs the registered
lever probe. Analysis-only registration; recipe stated there.

Run: python experiments/wp2_slack_attribution/run.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cus import tests                                    # noqa: E402
from cus.certificate import Audit, certificate           # noqa: E402

V3 = "wp2cert_7f8658cdfc4cac42"


def main():
    tests.test_prop2_reduces_to_unweighted()
    d = json.loads((ROOT / "artifacts" / V3 / "results.json").read_text())
    rows = d["rows"]
    dang = [r for r in rows if r["excess_archived"] >= 0.005]
    out = {}

    # P-SA1 / P-SA2 / P-SA3: term attribution on dangerous cells
    slack, calerr, calgap, calconf, aover, bown, sea = [], [], [], [], [], [], []
    for r in dang:
        dr = r["draws"]
        b = float(np.mean(dr["excess_bound"]))
        slack.append(b - r["excess_archived"])
        calerr.append(float(np.mean(dr["cal_err_loc"])))
        calgap.append(float(np.mean(dr["cal_err_gap"])))
        calconf.append(float(np.mean(dr["cal_err_conf"])))
        aover.append(float(np.mean(dr["a_plugin"])) - r["excess_archived"])
        bown.append(float(np.mean(dr["b_own_ucb"])))
        sea.append(1.645 * float(np.mean(dr["se_a"])))
    ms, mc = float(np.mean(slack)), float(np.mean(calerr))
    out["P-SA1"] = {"mean_slack": ms, "mean_cal_err": mc,
                    "cal_err_share": mc / ms}
    out["P-SA2"] = {"mean_gap_part": float(np.mean(calgap)),
                    "mean_conf_part": float(np.mean(calconf)),
                    "conf_share_of_calerr": float(np.mean(calconf)) / mc}
    accounted = (float(np.mean(aover)) + mc + float(np.mean(bown))
                 + float(np.mean(sea)))
    out["P-SA3"] = {"a_plugin_overshoot": float(np.mean(aover)),
                    "b_own_ucb": float(np.mean(bown)),
                    "z_se_a": float(np.mean(sea)),
                    "accounted_share": accounted / ms}
    print(f"[sa] dangerous cells n={len(dang)}: mean slack {ms:+.4f}")
    print(f"[sa] P-SA1 CalErr share of slack: {out['P-SA1']['cal_err_share']:.1%}")
    print(f"[sa] P-SA2 confidence share of CalErr: {out['P-SA2']['conf_share_of_calerr']:.1%} "
          f"(gap {np.mean(calgap):+.4f}, conf {np.mean(calconf):+.4f})")
    print(f"[sa] P-SA3 accounted share: {out['P-SA3']['accounted_share']:.1%} "
          f"(a_over {np.mean(aover):+.4f}, b_own {np.mean(bown):+.4f}, "
          f"z se {np.mean(sea):+.4f})")

    # P-SA4: the lever probe on the 30 largest-excess dangerous cells
    sys.path.insert(0, str(ROOT / "experiments" / "wp2_certificate_v3"))
    from run import make_env, build_what_fn, iter_cells, CONFIG  # noqa: E402
    dang_sorted = sorted(dang, key=lambda r: -r["excess_archived"])[:30]
    keyset = {(r["battery"], r["env"], r["kind"], r["setting"], r["beta"],
               r["alpha"]) for r in dang_sorted}
    envs, audits = {}, {}
    lever = {10000: [], 40000: []}
    for i, (bat, ename, feat, beta, alpha, kind, sd, exc) in \
            enumerate(iter_cells(CONFIG)):
        key = (bat, ename, kind, json.dumps(sd), beta, alpha)
        if key not in keyset:
            continue
        if ename not in envs:
            envs[ename] = make_env(ename)
            arng = np.random.default_rng([CONFIG["seed"], 900,
                                          abs(hash(ename)) % 10000])
            audits[ename] = Audit(envs[ename], arng, CONFIG["n_audit"])
        env, aud = envs[ename], audits[ename]
        for nsz in (10000, 40000):
            rng = np.random.default_rng([CONFIG["seed"], 903, i, nsz])
            bs = []
            for _ in range(10):
                what_fn = build_what_fn(env, ename, feat, beta, kind, sd,
                                        rng, CONFIG)
                bs.append(certificate(env, aud, alpha, rng, feat, beta,
                                      what_fn, n_cal=CONFIG["n_cal"],
                                      n_src=nsz, n_tgt=nsz,
                                      n_lambda=CONFIG["n_lambda"],
                                      z=CONFIG["z"],
                                      n_bins=CONFIG["n_bins"])["excess_bound"])
            lever[nsz].append(float(np.mean(bs)))
    m10, m40 = float(np.mean(lever[10000])), float(np.mean(lever[40000]))
    out["P-SA4"] = {"n_cells": len(lever[10000]),
                    "mean_bound_10k": m10, "mean_bound_40k": m40,
                    "reduction": m10 - m40}
    print(f"[sa] P-SA4 lever: mean bound {m10:+.4f} at 10k vs {m40:+.4f} at 40k "
          f"(reduction {m10 - m40:+.4f}, n={len(lever[10000])} cells)")

    dd = ROOT / "artifacts" / "wp2sa_analysis"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "report.json").write_text(json.dumps(out, indent=2))
    print(f"[out] {dd}")


if __name__ == "__main__":
    main()
