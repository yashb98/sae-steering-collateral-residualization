#!/usr/bin/env python
"""Robustness variants: label reliability, steering-coefficient sweep, random directions, dense exclusion.

Reads results/<setting>_ctxA and _ctxB (split-half contexts), results/<setting>_alpha<x> and _q95
(steering coefficient), results/<setting>_random (random directions) and the canonical
results/<setting> (dense-excluded labels). Writes results/analysis/variants.json and variants.md.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from residualize import CONTROLS, partial_spearman  # noqa: E402

ORDER = ["gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"]
PRETTY = {"gpt2_small": "GPT-2-small", "pythia_70m_deduped": "Pythia-70M-deduped", "gemma_2_2b": "Gemma-2-2B", "llama_3_1_8b": "Llama-3.1-8B"}
LABELS = ["collateral_raw", "collateral_ctilde", "effect_l2", "stab_signed", "stab_abs", "kl_mean"]


def load(results, name):
    p = os.path.join(results, name, "per_feature.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def crowding_stats(df, n_boot, rng, controls=("none", "robust", "primary")):
    out = {}
    for tgt in ["collateral_raw", "collateral_ctilde"]:
        for c in controls:
            r = partial_spearman(df, "crowding", tgt, CONTROLS[c], n_boot, rng)
            out[f"{tgt}|{c}"] = {k: r[k] for k in ["rho", "ci_lo", "ci_hi", "p", "n"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {"reliability": {}, "alpha": {}, "random": {}, "dense": {}}
    lines = ["# Robustness variants", ""]

    # ---- 1. split-half label reliability ----
    lines += ["## Label reliability (split-half over contexts)", "",
              "Same 300 features, labels measured once on the even-indexed contexts (A) and once on the odd-indexed contexts (B). "
              "Spearman between A and B is the test-retest reliability; the attenuation-corrected crowding correlation is rho / sqrt(reliability).", "",
              "| Setting | Label | reliability rho(A, B) | crowding rho (A) | crowding rho (B) | corrected, raw count |", "|---|---|---|---|---|---|"]
    for s in ORDER:
        A, B = load(a.results, f"{s}_ctxA"), load(a.results, f"{s}_ctxB")
        if A is None or B is None:
            continue
        m = A.merge(B, on="feature", suffixes=("_A", "_B"))
        rel = {lab: float(stats.spearmanr(m[f"{lab}_A"], m[f"{lab}_B"])[0]) for lab in LABELS if f"{lab}_A" in m}
        cA = {t: float(stats.spearmanr(m["crowding_A"], m[f"{t}_A"])[0]) for t in ["collateral_raw", "collateral_ctilde"]}
        cB = {t: float(stats.spearmanr(m["crowding_B"], m[f"{t}_B"])[0]) for t in ["collateral_raw", "collateral_ctilde"]}
        canon = load(a.results, s)
        rho_full = float(stats.spearmanr(canon.crowding, canon.collateral_raw)[0]) if canon is not None else float("nan")
        corrected = rho_full / np.sqrt(rel["collateral_raw"]) if rel["collateral_raw"] > 0 else float("nan")
        out["reliability"][s] = dict(reliability=rel, crowding_A=cA, crowding_B=cB, crowding_full=rho_full, corrected_raw=float(corrected), n=int(len(m)))
        for lab in LABELS:
            if lab in rel:
                extra = f"{cA['collateral_raw']:+.3f} | {cB['collateral_raw']:+.3f} | {corrected:+.3f}" if lab == "collateral_raw" else \
                        (f"{cA['collateral_ctilde']:+.3f} | {cB['collateral_ctilde']:+.3f} | " if lab == "collateral_ctilde" else " | | ")
                lines.append(f"| {PRETTY[s]} | {lab} | {rel[lab]:+.3f} | {extra} |")
    lines.append("")

    # ---- 2. steering coefficient ----
    lines += ["## Steering coefficient", "",
              "Crowding vs collateral under different additive coefficients. alpha = 1.0 is the paper's value; q95 adds alpha times the feature's "
              "95th-percentile natural activation, so the perturbation is scaled to each feature.", "",
              "| Setting | Variant | raw count: rho | primary partial | C-tilde: rho | primary partial | median intervention value |", "|---|---|---|---|---|---|---|"]
    for s in ORDER:
        dirs = sorted(glob.glob(os.path.join(a.results, f"{s}_alpha*"))) + sorted(glob.glob(os.path.join(a.results, f"{s}_q95*")))
        canon = load(a.results, s)
        variants = ([("alpha 1.0 (reported)", canon)] if canon is not None else []) + \
                   [(os.path.basename(d).replace(f"{s}_", ""), load(a.results, os.path.basename(d))) for d in dirs]
        for label, df in variants:
            if df is None or "crowding" not in df:
                continue
            st = crowding_stats(df, a.n_boot, rng, controls=("none", "primary"))
            out["alpha"].setdefault(s, {})[label] = st
            r = st
            lines.append(f"| {PRETTY[s]} | {label} | {r['collateral_raw|none']['rho']:+.3f} | {r['collateral_raw|primary']['rho']:+.3f} | "
                         f"{r['collateral_ctilde|none']['rho']:+.3f} | {r['collateral_ctilde|primary']['rho']:+.3f} | {df.intervention_value.median():.3g} |")
    lines.append("")

    # ---- 3. random directions ----
    lines += ["## Random-direction control", "",
              "300 random directions with norms drawn from the decoder norms, steered on 48 random contexts each. Crowding is the same top-20 "
              "mean absolute cosine to the dictionary. Frequency and activation controls do not exist for random directions, so the control is "
              "effect magnitude only; the SAE-feature row uses the same control for comparison.", "",
              "| Setting | Steered vectors | raw count: rho [95% CI] | partial given E_f | C-tilde: rho [95% CI] | partial given E_f | median collateral |", "|---|---|---|---|---|---|---|"]
    for s in ORDER:
        R, canon = load(a.results, f"{s}_random"), load(a.results, s)
        if R is None or canon is None:
            continue
        for label, df in [("SAE features", canon), ("random directions", R)]:
            st = crowding_stats(df, a.n_boot, rng, controls=("none", "effect_only"))
            out["random"].setdefault(s, {})[label] = st
            r = st
            lines.append(f"| {PRETTY[s]} | {label} | {r['collateral_raw|none']['rho']:+.3f} [{r['collateral_raw|none']['ci_lo']:+.3f}, {r['collateral_raw|none']['ci_hi']:+.3f}] | "
                         f"{r['collateral_raw|effect_only']['rho']:+.3f} | {r['collateral_ctilde|none']['rho']:+.3f} [{r['collateral_ctilde|none']['ci_lo']:+.3f}, {r['collateral_ctilde|none']['ci_hi']:+.3f}] | "
                         f"{r['collateral_ctilde|effect_only']['rho']:+.3f} | {df.collateral_raw.median():.1f} |")
    lines.append("")

    # ---- 4. dense exclusion and residual-space collateral ----
    lines += ["## Dense-panel exclusion and residual-space collateral", "",
              "Collateral recomputed without downstream panel features that fire in more than 10% of contexts, and the downstream residual-stream "
              "change itself (its norm, and the share of it the downstream SAE reconstructs).", "",
              "| Setting | panel kept | crowding vs count (all) | crowding vs count (no dense) | primary partial (no dense) | crowding vs residual change norm | mean SAE-explained share of the change |", "|---|---|---|---|---|---|---|"]
    for s in ORDER:
        df = load(a.results, s)
        if df is None or "collateral_raw_nodense" not in df:
            continue
        meta = json.load(open(os.path.join(a.results, s, "meta.json")))
        kept = meta.get("variant", {}).get("n_panel_nodense")
        r_all = partial_spearman(df, "crowding", "collateral_raw", [], a.n_boot, rng)
        r_nd = partial_spearman(df, "crowding", "collateral_raw_nodense", [], a.n_boot, rng)
        r_ndp = partial_spearman(df, "crowding", "collateral_raw_nodense", CONTROLS["primary"], a.n_boot, rng)
        r_res = partial_spearman(df, "crowding", "resid_delta_norm", [], a.n_boot, rng)
        out["dense"][s] = dict(panel_kept=kept, rho_all=r_all["rho"], rho_nodense=r_nd["rho"], partial_nodense=r_ndp["rho"],
                               rho_resid_norm=r_res["rho"], mean_sae_frac=float(df.resid_sae_frac.mean()))
        lines.append(f"| {PRETTY[s]} | {kept} of {meta['sizes']['panel']} | {r_all['rho']:+.3f} | {r_nd['rho']:+.3f} | {r_ndp['rho']:+.3f} | {r_res['rho']:+.3f} | {df.resid_sae_frac.mean():.3f} |")
    lines.append("")

    os.makedirs(os.path.join(a.results, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(a.results, "analysis", "variants.json"), "w"), indent=1)
    open(os.path.join(a.results, "analysis", "variants.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
