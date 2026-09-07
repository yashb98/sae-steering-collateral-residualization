#!/usr/bin/env python
"""Feature-sample robustness: crowding vs collateral across seed runs of the same setting.

Reads results/<setting>/ and results/<setting>_seed<k>/ per_feature.csv, reports raw Spearman
and the two partial correlations (robust pair; primary set) per seed, plus mean and sd.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from residualize import CONTROLS, partial_spearman  # noqa: E402


def main(results="results"):
    dirs = sorted(glob.glob(os.path.join(results, "*", "per_feature.csv")))
    groups = {}
    for p in dirs:
        s = os.path.basename(os.path.dirname(p))
        if "smoke" in s or "_v1" in s:
            continue
        m = re.match(r"(.*?)(?:_seed(\d+))?$", s)
        base, seed = m.group(1), int(m.group(2) or 0)
        if base not in {"gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"}:
            continue
        groups.setdefault(base, []).append((seed, pd.read_csv(p)))
    rng = np.random.default_rng(0)
    rows = []
    for base, runs in groups.items():
        for seed, df in sorted(runs):
            for tgt in ["collateral_raw", "collateral_ctilde"]:
                for cname in ["none", "robust", "primary"]:
                    r = partial_spearman(df, "crowding", tgt, CONTROLS[cname], 200, rng)
                    rows.append(dict(setting=base, seed=seed, target=tgt, control=cname, rho=r["rho"]))
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["setting", "target", "control"], columns="seed", values="rho")
    piv["mean"] = piv.mean(axis=1)
    piv["sd"] = piv.drop(columns=["mean"]).std(axis=1, ddof=1)
    piv["n_seeds"] = piv.drop(columns=["mean", "sd"]).notna().sum(axis=1)
    os.makedirs(os.path.join(results, "analysis"), exist_ok=True)
    piv.round(3).to_csv(os.path.join(results, "analysis", "seed_summary.csv"))
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
