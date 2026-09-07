#!/usr/bin/env python
"""Generate the robustness block from saved feature labels and matched controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from residualize import CONTROLS, partial_spearman

ORDER = ["gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"]
PRETTY = dict(zip(ORDER, ["GPT-2-small", "Pythia-70M-deduped", "Gemma-2-2B", "Llama-3.1-8B"]))
LABELS = ["collateral_raw", "collateral_ctilde", "effect_l2", "stab_signed", "stab_abs", "kl_mean",
          "stab_anti_frac", "effect_pc1_ratio"]


def load(results, name):
    return pd.read_csv(Path(results) / name / "per_feature.csv")


def label_reliability(A, B, n_boot, rng):
    if not A.feature.is_unique or not B.feature.is_unique or set(A.feature) != set(B.feature):
        raise ValueError("reliability requires the same unique features in both draws")
    m = A.merge(B, on="feature", suffixes=("_A", "_B"), validate="one_to_one")
    np.testing.assert_allclose(m.crowding_A, m.crowding_B, rtol=0, atol=0)
    return {lab: partial_spearman(m, f"{lab}_A", f"{lab}_B", [], n_boot, rng)
            for lab in LABELS if f"{lab}_A" in m and f"{lab}_B" in m}


def crowding_stats(df, n_boot, rng, controls):
    return {f"{target}|{control}": partial_spearman(df, "crowding", target, CONTROLS[control], n_boot, rng)
            for target in ["collateral_raw", "collateral_ctilde"] for control in controls}


def fmt(r):
    if not np.isfinite(r["rho"]):
        return "undefined (constant label)"
    if not np.isfinite(r["ci_lo"]) or not np.isfinite(r["ci_hi"]):
        return f"{r['rho']:+.3f} [CI undefined]"
    return f"{r['rho']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.results)
    rng = np.random.default_rng(a.seed)
    out = {"meta": {"n_boot": a.n_boot, "seed": a.seed, "version": 2},
           "reliability": {}, "alpha": {}, "random": {}, "dense": {}, "stability": {}}
    lines = ["# Robustness variants", "", "## Context-pool reproducibility", "",
             "The same 300 features and clean-activation predictors are retained. Labels use 48 mixed contexts selected from "
             "the 1,024 even-indexed contexts (A) or the 1,024 odd-indexed contexts (B) of the same ordered corpus pool. "
             "The halves are disjoint, but are not independent corpus draws; neighboring texts can be related. "
             "These are split-pool Spearman correlations, conditional on one dictionary and feature sample. "
             "No predictor ceiling or attenuation-corrected correlation is claimed. Correlation is undefined when one draw gives a constant label. Percentile intervals resample matched feature rows and are undefined if a draw is degenerate; JSON records the valid-draw count.", "",
             "| Setting | Label | rho(A, B) [95% CI] | crowding rho (A) | crowding rho (B) |",
             "|---|---|---|---|---|"]
    for s in ORDER:
        A, B = load(root, s + "_ctxA"), load(root, s + "_ctxB")
        canon = load(root, s)
        assert set(A.feature) == set(canon.feature), s
        rel = label_reliability(A, B, a.n_boot, rng)
        cA = {t: float(stats.spearmanr(A.crowding, A[t])[0]) for t in ["collateral_raw", "collateral_ctilde"]}
        cB = {t: float(stats.spearmanr(B.crowding, B[t])[0]) for t in cA}
        out["reliability"][s] = dict(labels=rel, crowding_A=cA, crowding_B=cB, n=len(A))
        for label, r in rel.items():
            extra = f"{cA[label]:+.3f} | {cB[label]:+.3f}" if label in cA else " | "
            lines.append(f"| {PRETTY[s]} | {label} | {fmt(r)} | {extra} |")

    lines += ["", "## Steering coefficient", "",
              "Only GPT-2 and Pythia have coefficient sweeps. q95 is the 95th percentile conditional on firing "
              "(activation > 1e-6), multiplied by alpha; it is not the unconditional activation percentile. "
              "The primary control includes the varying intervention value for q95. These are coefficient sensitivity checks, "
              "without evidence that the selected scale improves generated behavior.", "",
              "| Setting | Variant | raw count: rho | primary partial [95% CI] | C-tilde: rho | primary partial [95% CI] | median coefficient |",
              "|---|---|---|---|---|---|---|"]
    for s in ORDER[:2]:
        for suffix, name in [("", "alpha1"), ("_alpha0.5", "alpha0.5"), ("_alpha2", "alpha2"), ("_alpha4", "alpha4"), ("_q95", "q95 firing")]:
            df = load(root, s + suffix)
            r = crowding_stats(df, a.n_boot, rng, ("none", "primary"))
            r["median_intervention_value"] = float(df.intervention_value.median())
            out["alpha"].setdefault(s, {})[name] = r
            lines.append(f"| {PRETTY[s]} | {name} | {r['collateral_raw|none']['rho']:+.3f} | {fmt(r['collateral_raw|primary'])} | "
                         f"{r['collateral_ctilde|none']['rho']:+.3f} | {fmt(r['collateral_ctilde|primary'])} | {df.intervention_value.median():.3g} |")

    lines += ["", "## Paired random-direction control", "",
              "Each of 300 isotropic Gaussian directions is normalized to unit length, then scaled to the corresponding "
              "sampled SAE decoder norm. Each pair uses identical 48-context indices, the same coefficient, downstream panel, "
              "and clean forward pass. This retains the nonunit Llama decoder scales. Direction seed is 2000; vectors and "
              "context indices are saved in each *_random_paired directory. Frequency and natural activation are undefined "
              "for random directions, so both arms use effect-only adjustment. The older *_random files used unmatched "
              "random contexts and are retained as exploratory artifacts. Within-arm correlations and pointwise CIs do not "
              "establish a difference between correlations, a causal role for crowding, or improved behavioral utility.", "",
              "| Setting | Vectors | raw count: rho [95% CI] | partial given E_f [95% CI] | C-tilde: rho [95% CI] | partial given E_f [95% CI] | median count | median effect L2 | median C-tilde |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in ORDER:
        R, canon = load(root, s + "_random_paired"), load(root, s)
        paired = R.merge(canon, left_on="matched_feature", right_on="feature", suffixes=("_random", "_sae"), validate="one_to_one")
        assert len(paired) == len(canon) == 300, s
        np.testing.assert_allclose(paired.dec_norm_random, paired.dec_norm_sae, rtol=0, atol=0)
        sa = json.loads((root/s/"selection.json").read_text())
        sb = json.loads((root/(s+"_random_paired")/"selection.json").read_text())
        assert sa["contexts"] == sb["contexts"] and sa["panel"] == sb["panel"], s
        for label, df in [("SAE features", canon), ("paired random directions", R)]:
            r = crowding_stats(df, a.n_boot, rng, ("none", "effect_only"))
            r["median_collateral_raw"] = float(df.collateral_raw.median())
            r["median_effect_l2"] = float(df.effect_l2.median())
            r["median_collateral_ctilde"] = float(df.collateral_ctilde.median())
            out["random"].setdefault(s, {})[label] = r
            lines.append(f"| {PRETTY[s]} | {label} | {fmt(r['collateral_raw|none'])} | {fmt(r['collateral_raw|effect_only'])} | "
                         f"{fmt(r['collateral_ctilde|none'])} | {fmt(r['collateral_ctilde|effect_only'])} | {df.collateral_raw.median():.1f} | {df.effect_l2.median():.3g} | {df.collateral_ctilde.median():.3g} |")

    lines += ["", "## Dense-panel exclusion", "",
              "Panel features firing in more than 10% of clean contexts are excluded. The remaining panel is still selected "
              "by frequency; this check does not provide coverage of all sparse downstream features.", "",
              "| Setting | panel kept | crowding vs count (all) | crowding vs count (no dense) | primary partial (no dense) [95% CI] |",
              "|---|---|---|---|---|"]
    residual_rows, stability_rows = [], []
    for s in ORDER:
        df = load(root, s)
        meta = json.loads((root/s/"meta.json").read_text())
        kept = meta["variant"]["n_panel_nodense"]
        all_r = partial_spearman(df, "crowding", "collateral_raw", [], a.n_boot, rng)
        nd_r = partial_spearman(df, "crowding", "collateral_raw_nodense", [], a.n_boot, rng)
        nd_p = partial_spearman(df, "crowding", "collateral_raw_nodense", CONTROLS["primary"], a.n_boot, rng)
        res_r = partial_spearman(df, "crowding", "resid_delta_norm", [], a.n_boot, rng)
        data = dict(panel_kept=kept, rho_all=all_r, rho_nodense=nd_r, partial_nodense=nd_p,
                    rho_resid_norm=res_r, mean_reconstruction_norm_ratio=float(df.resid_reconstruction_norm_ratio.mean()),
                    mean_change_explained_energy=float(df.resid_change_explained_energy.mean()),
                    median_change_explained_energy=float(df.resid_change_explained_energy.median()),
                    negative_explained_fraction=float((df.resid_change_explained_energy < 0).mean()),
                    mean_residual_error_change_norm=float(df.resid_error_delta_norm.mean()))
        out["dense"][s] = data
        lines.append(f"| {PRETTY[s]} | {kept} of {meta['sizes']['panel']} | {all_r['rho']:+.3f} | {nd_r['rho']:+.3f} | {fmt(nd_p)} |")
        residual_rows.append(f"| {PRETTY[s]} | {fmt(res_r)} | {data['mean_reconstruction_norm_ratio']:.3f} | "
                             f"{data['mean_change_explained_energy']:+.3f} | {data['median_change_explained_energy']:+.3f} | {data['negative_explained_fraction']:.1%} |")
        st = dict(median_anti_frac=float(df.stab_anti_frac.median()), median_pc1_ratio=float(df.effect_pc1_ratio.median()))
        out["stability"][s] = st
        stability_rows.append(f"| {PRETTY[s]} | {st['median_anti_frac']:.3f} | {st['median_pc1_ratio']:.3f} |")
    lines += ["", "## Residual-space change", "",
              "Let dh be the downstream hidden-state change and dr the change in its full SAE reconstruction. "
              "The unrepresented change is de = dh - dr. For each feature we report 1 - sum_context ||de||^2 / "
              "sum_context ||dh||^2, a reconstruction score relative to predicting zero change. It can be negative "
              "when reconstruction change is a worse prediction than zero. It is not a bounded variance partition: "
              "dr and de need not be orthogonal. The reconstruction norm ratio mean_context ||dr||/||dh|| is reported "
              "separately and is not an explained fraction. Neither measure proves behavioral harm or residual-mediated "
              "recovery. Per-context dh and dr are saved locally as residual_deltas.npz and verified against the CSV; "
              "these large arrays are excluded from git and can be regenerated with --save-residual-deltas.", "",
              "| Setting | crowding vs ||dh|| [95% CI] | mean reconstruction norm ratio | mean explained-change score | median score | features with score < 0 |",
              "|---|---|---|---|---|---|"] + residual_rows
    lines += ["", "## Additional stability summaries", "",
              "PC1 is the top eigenvalue divided by the trace of the centered logit-change Gram matrix over 48 contexts. "
              "Anti-aligned fraction counts contexts whose change has negative cosine with that feature's mean change, "
              "including the context in the mean. These are descriptive summaries inspired by FEGA and steering-reliability "
              "work; they do not reproduce feature removal or behavior-score multiplier slopes.", "",
              "| Setting | median anti-aligned context fraction | median centered PC1 ratio |", "|---|---|---|"] + stability_rows
    dest = root/"analysis"
    dest.mkdir(exist_ok=True)
    (dest/"variants.json").write_text(json.dumps(json_safe(out), indent=2, allow_nan=False) + "\n")
    (dest/"variants.md").write_text("\n".join(lines) + "\n")
    print("wrote", dest/"variants.md")


if __name__ == "__main__":
    main()
