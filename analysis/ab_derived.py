#!/usr/bin/env python
"""Derived A/B analyses from existing artifacts (no GPU).

Motivated by the 2026-09-13 anchor deep-reads (Ong et al. 2608.11227;
Upadhyaya and Sikdar 2608.29936). Computes, per setting:

1. Spearman-Brown measurement ceilings from the ctxA/ctxB split-half
   correlations in variants.json, and attenuation-corrected crowding partial
   rhos (partial rho / sqrt(R_label)). Corrected values can exceed 1; they are
   point estimates under a classical measurement-error assumption, reported
   alongside the uncorrected values.
2. Precision at the top decile: fraction of a predictor's top-decile features
   that also land in the collateral top decile (baseline 10%).
3. Max-vs-mean aggregation: crowd_max (max |cos|) against crowding (mean
   top-20 |cos|) under the primary control.
4. Orthogonal-selection demo: within effect_l2 terciles, median collateral in
   the low-crowding vs high-crowding half.
5. Dose-response across steering coefficients: median collateral per alpha
   and the per-feature Spearman(collateral, alpha) distribution, for every
   setting with >=3 alpha runs.

Outputs: results/analysis/ab_derived.json and ab_derived.md.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "results"
OUT_JSON = os.path.join(RESULTS, "analysis", "ab_derived.json")
OUT_MD = os.path.join(RESULTS, "analysis", "ab_derived.md")
CANONICAL = ["gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"]
TARGETS = ["collateral_raw", "collateral_ctilde"]


def spearman_brown(r):
    return 2.0 * r / (1.0 + r)


def load(setting):
    path = os.path.join(RESULTS, setting, "per_feature.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def top_decile_precision(df, predictor, target):
    df = df[[predictor, target]].dropna()
    if len(df) < 30:
        return None
    px = df[predictor] >= df[predictor].quantile(0.9)
    py = df[target] >= df[target].quantile(0.9)
    if px.sum() == 0:
        return None
    return dict(precision=float(py[px].mean()), n_flag=int(px.sum()),
                overlap=int((px & py).sum()), baseline=0.1)


def selection_demo(df, target):
    df = df[["effect_l2", "crowding", target]].dropna()
    if len(df) < 60:
        return None
    df = df.copy()
    df["tercile"] = pd.qcut(df["effect_l2"], 3, labels=["low", "mid", "high"])
    rows = {}
    for t, g in df.groupby("tercile", observed=True):
        med = g["crowding"].median()
        lo, hi = g[g["crowding"] <= med], g[g["crowding"] > med]
        rows[t] = dict(median_collateral_low_crowding=float(lo[target].median()),
                       median_collateral_high_crowding=float(hi[target].median()),
                       n_low=int(len(lo)), n_high=int(len(hi)),
                       median_effect=float(g["effect_l2"].median()))
    return rows


def dose_response(setting):
    runs = {}
    for d in sorted(glob.glob(os.path.join(RESULTS, setting + "_alpha*"))):
        alpha = os.path.basename(d).split("_alpha")[1]
        path = os.path.join(d, "per_feature.csv")
        if os.path.exists(path):
            runs[float(alpha)] = pd.read_csv(path).set_index("feature")
    base = load(setting)
    if base is None or len(runs) < 2:
        return None
    runs[1.0] = base.set_index("feature")
    if len(runs) < 3:
        return None
    alphas = sorted(runs)
    common = set.intersection(*[set(r.index) for r in runs.values()])
    out = {"alphas": alphas, "n_common_features": len(common)}
    for tgt in TARGETS:
        mat = np.array([[runs[a].loc[f, tgt] for a in alphas] for f in common], float)
        med = np.nanmedian(mat, axis=0)
        rhos = np.array([stats.spearmanr(alphas, row)[0] for row in mat])
        out[tgt] = dict(median_collateral_per_alpha=[float(x) for x in med],
                        frac_features_monotone_up=float(np.nanmean(rhos > 0.99)),
                        per_feature_rho_median=float(np.nanmedian(rhos)),
                        per_feature_rho_iqr=[float(np.nanpercentile(rhos, 25)),
                                             float(np.nanpercentile(rhos, 75))])
    return out


def main():
    variants = json.load(open(os.path.join(RESULTS, "analysis", "variants.json")))
    part = pd.read_csv(os.path.join(RESULTS, "analysis", "partial_correlations.csv"))
    report = {}
    for s in CANONICAL:
        df = load(s)
        if df is None:
            continue
        entry = {}
        rel = variants["reliability"].get(s, {}).get("labels", {})
        ceiling = {}
        for tgt in TARGETS:
            r = rel.get(tgt, {}).get("rho")
            if r and 0 < r < 1:
                R = spearman_brown(r)
                row = part[(part.setting == s) & (part.target == tgt)
                           & (part.control == "primary") & (part.predictor == "crowding")]
                if len(row):
                    rho = float(row.iloc[0].rho)
                    ceiling[tgt] = dict(split_half_r=float(r), spearman_brown_R=float(R),
                                        partial_rho=rho,
                                        attenuation_corrected=float(rho / np.sqrt(R)))
        entry["measurement_ceiling"] = ceiling
        entry["top_decile_precision"] = {
            tgt: {p: top_decile_precision(df, p, tgt)
                  for p in ["crowding", "crowd_max", "logit_l2", "enc_dec_cos"]
                  if top_decile_precision(df, p, tgt)}
            for tgt in TARGETS}
        maxmean = {}
        for tgt in TARGETS:
            sub = part[(part.setting == s) & (part.target == tgt) & (part.control == "primary")
                       & (part.predictor.isin(["crowding", "crowd_max"]))]
            maxmean[tgt] = {r.predictor: dict(rho=float(r.rho), ci=[float(r.ci_lo), float(r.ci_hi)])
                            for r in sub.itertuples()}
        entry["max_vs_mean"] = maxmean
        entry["orthogonal_selection_demo"] = {tgt: selection_demo(df, tgt) for tgt in TARGETS}
        entry["dose_response"] = dose_response(s)
        report[s] = entry

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(report, open(OUT_JSON, "w"), indent=1)

    lines = ["# Derived A/B analyses (2026-09-13, branch ab-battery-2026-09-13)", "",
             "Inputs: canonical per-feature CSVs, alpha-variant CSVs, variants.json,",
             "partial_correlations.csv. No new GPU measurements.", ""]
    for s, e in report.items():
        lines.append(f"## {s}")
        lines.append("")
        lines.append("Measurement ceiling (ctxA/ctxB split-half, Spearman-Brown) and attenuation-corrected crowding partial rho:")
        lines.append("")
        lines.append("| target | split-half r | SB reliability R | partial rho | corrected |")
        lines.append("|---|---|---|---|---|")
        for tgt, c in e["measurement_ceiling"].items():
            lines.append(f"| {tgt} | {c['split_half_r']:+.3f} | {c['spearman_brown_R']:.3f} | "
                         f"{c['partial_rho']:+.3f} | {c['attenuation_corrected']:+.3f} |")
        lines.append("")
        lines.append("Top-decile precision (fraction of predictor-top-decile features in collateral top decile; baseline 0.10):")
        lines.append("")
        lines.append("| target | predictor | precision | n flagged |")
        lines.append("|---|---|---|---|")
        for tgt, preds in e["top_decile_precision"].items():
            for p, r in preds.items():
                lines.append(f"| {tgt} | {p} | {r['precision']:.3f} | {r['n_flag']} |")
        lines.append("")
        lines.append("Max-vs-mean crowding aggregation (primary control):")
        lines.append("")
        lines.append("| target | crowding (mean top-20) | crowd_max (max) |")
        lines.append("|---|---|---|")
        for tgt, mm in e["max_vs_mean"].items():
            a, b = mm.get("crowding"), mm.get("crowd_max")
            if a and b:
                lines.append(f"| {tgt} | {a['rho']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | "
                             f"{b['rho']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] |")
        lines.append("")
        demo = e["orthogonal_selection_demo"]["collateral_raw"]
        if demo:
            lines.append("Orthogonal-selection demo (median raw collateral within effect_l2 terciles):")
            lines.append("")
            lines.append("| effect tercile | low crowding | high crowding |")
            lines.append("|---|---|---|")
            for t, r in demo.items():
                lines.append(f"| {t} | {r['median_collateral_low_crowding']:.2f} | {r['median_collateral_high_crowding']:.2f} |")
            lines.append("")
        dr = e["dose_response"]
        if dr:
            lines.append(f"Dose-response over alphas {dr['alphas']} (n={dr['n_common_features']} common features):")
            lines.append("")
            lines.append("| target | median collateral per alpha | per-feature rho(collateral, alpha) median [IQR] |")
            lines.append("|---|---|---|")
            for tgt in TARGETS:
                d = dr[tgt]
                lines.append(f"| {tgt} | {['%.2f' % x for x in d['median_collateral_per_alpha']]} | "
                             f"{d['per_feature_rho_median']:+.3f} [{d['per_feature_rho_iqr'][0]:+.3f}, {d['per_feature_rho_iqr'][1]:+.3f}] |")
            lines.append("")
    open(OUT_MD, "w").write("\n".join(lines))
    print("wrote", OUT_JSON, "and", OUT_MD)


if __name__ == "__main__":
    main()
