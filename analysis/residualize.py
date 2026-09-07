#!/usr/bin/env python
"""Collateral-side residualization across settings.

For every results/<setting>/per_feature.csv and both collateral metrics (raw downstream count
and C-tilde = C / (E_f + eps)):

1. Partial Spearman correlation of each predictor with the label, rank-based, with both sides
   OLS-residualised on the ranked controls. Control sets: none (raw Spearman); primary
   (effect magnitude E_f, intervention value, natural activation, firing frequency; constant
   columns are dropped); robust (frequency and activation magnitude). Bootstrap 95% CIs over
   features.
2. Table B3 analog: the label is OLS-residualised on the same controls, then predicted from
   each predictor family by 5-fold cross-validated ridge; score = Spearman between held-out
   predictions and the residualised label.

Outputs go to results/analysis/.
"""
import argparse
import glob
import hashlib
from importlib.metadata import version
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

PRED_FAMILIES = {
    "geometry": ["crowding", "crowd_max", "dec_norm", "enc_norm", "enc_dec_cos"],
    "activation_nonmag": ["frequency", "bin_entropy", "act_entropy", "act_kurtosis"],
    "activation_mag": ["act_mag", "act_mean_firing", "act_std", "act_max"],
    "coactivation": ["coact_entropy", "coact_count"],
    "direct_logit": ["logit_l2", "logit_linf", "logit_entropy", "logit_top10_mass"],
}
ALL_PREDICTORS = sum(PRED_FAMILIES.values(), [])
PRED_SETS = {
    "frequency_only": ["frequency"],
    "actmag_only": ["act_mag"],
    "geometry_only": PRED_FAMILIES["geometry"],
    "direct_logit_only": PRED_FAMILIES["direct_logit"],
    "coactivation_only": PRED_FAMILIES["coactivation"],
    "full_no_magnitude": PRED_FAMILIES["geometry"] + PRED_FAMILIES["activation_nonmag"]
                         + PRED_FAMILIES["coactivation"] + PRED_FAMILIES["direct_logit"],
    "full_all": ALL_PREDICTORS,
}
TARGETS = ["collateral_raw", "collateral_ctilde"]
EXTRA_TARGETS = ["kl_mean", "kl_per_effect", "collateral_raw_nodense", "collateral_ctilde_nodense", "resid_delta_norm", "resid_error_delta_norm", "resid_change_explained_energy", "stab_anti_frac", "effect_pc1_ratio"]
CONTROLS = {
    "none": [],
    "primary": ["effect_l2", "intervention_value", "act_mag", "frequency"],
    "robust": ["frequency", "act_mag"],
    "effect_only": ["effect_l2"],
}


def holm(p):
    """Holm adjustment within the supplied family; undefined tests stay undefined."""
    p = np.asarray(p, float)
    adj = np.full(p.shape, np.nan)
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order])]
    if len(order):
        adj[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(order))) * p[order]))
    return adj


def benjamini_hochberg(p):
    """BH adjustment; undefined entries still count toward the supplied family size."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order])]
    if len(order):
        raw = len(p) * p[order] / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    return q


def rank_rows(x):
    """Average-rank along the feature axis (axis 1): works for [B, n] and [B, n, k] arrays."""
    return stats.rankdata(x, axis=1)


def partial_corr_batched(x, y, Z):
    """x, y: [B, n]; Z: [B, n, k] (k may be 0). Returns Pearson corr of OLS residuals per row."""
    B, n = x.shape
    ones = np.ones((B, n, 1))
    D = np.concatenate([ones, Z], axis=2) if Z.shape[2] else ones
    # batched least squares via normal equations (tiny ridge for numerical stability)
    DtD = np.einsum("bnk,bnl->bkl", D, D) + 1e-9 * np.eye(D.shape[2])[None]
    inv = np.linalg.inv(DtD)
    def resid(v):
        beta = np.einsum("bkl,bnl,bn->bk", inv, D, v)
        return v - np.einsum("bnk,bk->bn", D, beta)
    rx, ry = resid(x), resid(y)
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    num = (rx * ry).sum(1)
    vx, vy = (rx ** 2).sum(1), (ry ** 2).sum(1)
    raw_vx = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
    raw_vy = ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
    valid = (raw_vx > 0) & (raw_vy > 0) & (vx > 1e-20 * raw_vx) & (vy > 1e-20 * raw_vy)
    return np.divide(num, np.sqrt(vx * vy), out=np.full_like(num, np.nan), where=valid)


def partial_spearman(df, pred, tgt, controls, n_boot, rng):
    controls = [c for c in controls if c != pred and c in df.columns and df[c].nunique(dropna=True) > 1]
    df = df[[pred, tgt] + controls].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n <= len(controls) + 2 or df[pred].nunique() < 2 or df[tgt].nunique() < 2:
        return dict(rho=float("nan"), p=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    n=n, k_controls=len(controls), controls=",".join(controls), n_boot=n_boot)
    x = df[pred].to_numpy(float)[None]
    y = df[tgt].to_numpy(float)[None]
    Z = np.stack([df[c].to_numpy(float) for c in controls], axis=1)[None] if controls else np.zeros((1, n, 0))
    r0 = partial_corr_batched(rank_rows(x), rank_rows(y), rank_rows(Z) if controls else Z)[0]
    k = len(controls)
    dfree = n - 2 - k
    t = r0 * np.sqrt(dfree / max(1e-12, 1 - r0 ** 2))
    p = 2 * stats.t.sf(abs(t), dfree)
    idx = rng.integers(0, n, size=(n_boot, n))
    xb, yb = x[0][idx], y[0][idx]
    Zb = Z[0][idx] if controls else np.zeros((n_boot, n, 0))
    Zb_r = rank_rows(Zb) if controls else Zb
    rb = partial_corr_batched(rank_rows(xb), rank_rows(yb), Zb_r)
    lo, hi = np.percentile(rb, [2.5, 97.5])
    return dict(rho=float(r0), p=float(p), ci_lo=float(lo), ci_hi=float(hi), n=n, k_controls=k,
                controls=",".join(controls), n_boot=n_boot, n_boot_valid=int(np.isfinite(rb).sum()))


def partial_spearman_many(df, predictors, tgt, controls, n_boot, rng, batch_size=128):
    """Share bootstrap draws and nuisance fits across a complete predictor family."""
    controls = [c for c in controls if c in df and df[c].nunique(dropna=True) > 1]
    columns = list(dict.fromkeys(predictors + [tgt] + controls))
    if (not predictors or any(p in controls for p in predictors)
            or not np.isfinite(df[columns].to_numpy(float)).all()
            or any(df[p].nunique() < 2 for p in predictors + [tgt])):
        return {p: partial_spearman(df, p, tgt, controls, n_boot, rng) for p in predictors}
    n, k = len(df), len(controls)
    if n <= k + 2:
        return {p: partial_spearman(df, p, tgt, controls, n_boot, rng) for p in predictors}
    values = df[predictors + [tgt]].to_numpy(float)
    nuisance = df[controls].to_numpy(float)

    def correlate(v, z):
        v = stats.rankdata(v, axis=1)
        D = np.concatenate([np.ones((*v.shape[:2], 1)), stats.rankdata(z, axis=1)], axis=2)
        DtD = np.einsum("bnk,bnl->bkl", D, D) + 1e-9 * np.eye(D.shape[2])[None]
        beta = np.linalg.inv(DtD) @ (D.transpose(0, 2, 1) @ v)
        residual = v - D @ beta
        residual -= residual.mean(1, keepdims=True)
        x, y = residual[:, :, :-1], residual[:, :, -1:]
        vx, vy = np.sum(x*x, axis=1), np.sum(y*y, axis=1)
        raw_var = np.sum((v - v.mean(1, keepdims=True)) ** 2, axis=1)
        raw_x, raw_y = raw_var[:, :-1], raw_var[:, -1:]
        valid = (raw_x > 0) & (raw_y > 0) & (vx > 1e-20 * raw_x) & (vy > 1e-20 * raw_y)
        return np.divide(np.sum(x*y, axis=1), np.sqrt(vx*vy), out=np.full_like(vx, np.nan), where=valid)

    r0 = correlate(values[None], nuisance[None])[0]
    boot = np.empty((n_boot, len(predictors)))
    for start in range(0, n_boot, batch_size):
        size = min(batch_size, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        boot[start:start+size] = correlate(values[idx], nuisance[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    dfree = n - 2 - k
    t = r0 * np.sqrt(dfree / np.maximum(1e-12, 1-r0*r0))
    pvals = 2 * stats.t.sf(np.abs(t), dfree)
    return {pred: dict(rho=float(r0[i]), p=float(pvals[i]), ci_lo=float(lo[i]), ci_hi=float(hi[i]),
                       n=n, k_controls=k, controls=",".join(controls), n_boot=n_boot, n_boot_valid=int(np.isfinite(boot[:, i]).sum()))
            for i, pred in enumerate(predictors)}


def residualized_cv(df, tgt, controls, pred_set, n_splits=5, seed=0, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    controls = [c for c in controls if c in df.columns and df[c].nunique(dropna=True) > 1]
    pred_set = [c for c in pred_set if c in df.columns and df[c].notna().all()]
    missing = dict(cv_spearman_mean=float("nan"), cv_spearman_sd=float("nan"),
                   oof_spearman=float("nan"), n_pred=len(pred_set), n_defined_folds=0, controls=",".join(controls))
    if not pred_set:
        return missing
    columns = list(dict.fromkeys([tgt] + controls + pred_set))
    df = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2 * n_splits or df[tgt].nunique() < 2:
        return missing
    y = df[tgt].to_numpy(float)
    Z = np.c_[np.ones(len(df)), df[controls].to_numpy(float)]
    X = df[pred_set].to_numpy(float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rhos, oof, heldout_residual = [], np.zeros_like(y), np.zeros_like(y)
    for tr, te in kf.split(X):
        if controls:
            beta, *_ = np.linalg.lstsq(Z[tr], y[tr], rcond=None)
            train_y = y[tr] - Z[tr] @ beta
            test_y = y[te] - Z[te] @ beta
        else:
            train_y, test_y = y[tr], y[te]
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), train_y)
        pred = m.predict(sc.transform(X[te]))
        oof[te] = pred
        heldout_residual[te] = test_y
        threshold = 1e-10 * max(np.std(train_y), np.finfo(float).tiny)
        defined = np.std(pred) > threshold and np.std(test_y) > threshold
        fold_rhos.append(stats.spearmanr(pred, test_y)[0] if defined else float("nan"))
    threshold = 1e-10 * max(np.std(heldout_residual), np.finfo(float).tiny)
    pooled = stats.spearmanr(oof, heldout_residual)[0] if np.std(oof) > threshold else float("nan")
    return dict(cv_spearman_mean=float(np.mean(fold_rhos)), cv_spearman_sd=float(np.std(fold_rhos)),
                oof_spearman=float(pooled), n_pred=len(pred_set), n_defined_folds=int(np.isfinite(fold_rhos).sum()),
                controls=",".join(controls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--settings", nargs="*", default=["gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"], help="setting dirs to include (default: four canonical settings)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="optional content-validated cache of per-setting statistics")
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "analysis")
    os.makedirs(out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.results, "*", "per_feature.csv")))
    settings = {}
    for p in paths:
        s = os.path.basename(os.path.dirname(p))
        if args.settings is not None and s not in args.settings:
            continue
        if args.settings is None and ("smoke" in s or s.endswith("_v1")):
            continue
        settings[s] = pd.read_csv(p)
    assert settings, "no per_feature.csv found"
    print("settings:", {k: len(v) for k, v in settings.items()})

    dependencies = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}
    rng = np.random.default_rng(args.seed)
    rows, cv_rows = [], []
    for s, df in settings.items():
        setting_seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, setting_seed]))
        cache_file = None
        if args.cache:
            identity = hashlib.sha256(open(__file__, "rb").read() + df.to_csv(index=False).encode()
                                      + f"{s}|{args.n_boot}|{args.seed}".encode()
                                      + json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
            cache_file = os.path.join(args.cache, f"{s}-{identity}.json")
            if os.path.exists(cache_file):
                cached = json.load(open(cache_file))
                rows.extend(cached["partial"])
                cv_rows.extend(cached["cv"])
                print(f"  {s}: reused content-validated statistics", flush=True)
                continue
        first_row, first_cv = len(rows), len(cv_rows)
        usable_preds = [p for p in ALL_PREDICTORS if p in df.columns and df[p].notna().sum() >= 30]
        for tgt in TARGETS + [t for t in EXTRA_TARGETS if t in df.columns]:
            for cname, controls in CONTROLS.items():
                if any(c not in df.columns or df[c].isna().all() for c in controls):
                    continue
                tested = [p for p in usable_preds if p not in controls or cname == "none"]
                family = partial_spearman_many(df, tested, tgt, controls, args.n_boot, rng)
                for pred, r in family.items():
                    rows.append(dict(setting=s, target=tgt, control=cname, predictor=pred, **r))
                for pset, preds in PRED_SETS.items():
                    cv = residualized_cv(df, tgt, controls, preds, seed=args.seed)
                    cv_rows.append(dict(setting=s, target=tgt, control=cname, predictor_set=pset, **cv))
            print(f"  {s} {tgt}: done", flush=True)
        if cache_file:
            os.makedirs(args.cache, exist_ok=True)
            with open(cache_file + ".tmp", "w") as stream:
                json.dump({"partial": rows[first_row:], "cv": cv_rows[first_cv:]}, stream)
            os.replace(cache_file + ".tmp", cache_file)
    part = pd.DataFrame(rows)
    # multiple-comparison correction across predictors within each (setting, target, control)
    part["p_holm"] = np.nan
    part["q_bh"] = np.nan
    for _, idx in part.groupby(["setting", "target", "control"]).groups.items():
        part.loc[idx, "p_holm"] = holm(part.loc[idx, "p"].to_numpy())
        part.loc[idx, "q_bh"] = benjamini_hochberg(part.loc[idx, "p"].to_numpy())
    cv = pd.DataFrame(cv_rows)
    part.to_csv(os.path.join(out, "partial_correlations.csv"), index=False)
    cv.to_csv(os.path.join(out, "residualized_cv_ridge.csv"), index=False)

    # ---- markdown summary: crowding first, then the leading predictor per setting ----
    lines = ["# Collateral-side residualization: summary", "",
             f"n_boot = {args.n_boot}, bootstrap over features with replacement, 95% percentile CIs.",
             "Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; "
             "robust = frequency + act_mag; none = raw Spearman.", ""]
    for tgt in TARGETS:
        lines += [f"## Target: {tgt}", "", "### Decoder crowding", "",
                  "| setting | control | rho | 95% CI | p | controls |", "|---|---|---|---|---|---|"]
        sub = part[(part.target == tgt) & (part.predictor == "crowding")]
        for _, r in sub.iterrows():
            lines.append(f"| {r.setting} | {r.control} | {r.rho:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p:.1e} | {r.controls or '-'} |")
        lines += ["", "### Strongest predictor per setting under the primary control (by |rho|)", "",
                  "| setting | predictor | partial rho | 95% CI | p | Holm p |", "|---|---|---|---|---|---|"]
        sub = part[(part.target == tgt) & (part.control == "primary")]
        for s in settings:
            top = sub[sub.setting == s].dropna(subset=["rho"]).sort_values("rho", key=lambda v: v.abs(), ascending=False).head(3)
            for _, r in top.iterrows():
                lines.append(f"| {s} | {r.predictor} | {r.rho:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p:.1e} | {r.p_holm:.1e} |")
        lines += ["", "### Table B3 analog: CV ridge Spearman on the residualized target (primary control)", "",
                  "| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |", "|---|---|---|---|---|"]
        sub = cv[(cv.target == tgt) & (cv.control == "primary")]
        for _, r in sub.iterrows():
            lines.append(f"| {r.setting} | {r.predictor_set} | {r.cv_spearman_mean:+.3f} | {r.cv_spearman_sd:.3f} | {r.oof_spearman:+.3f} |")
        lines.append("")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))
    json.dump({"n_boot": args.n_boot, "seed": args.seed, "settings": list(settings), "versions": dependencies,
               "controls": CONTROLS, "predictor_sets": PRED_SETS,
               "assumptions": ["natural activation a-bar_f := act_mag (mean final-token activation over all contexts)",
                               "intervention value c_f is constant under fixed_global_add (alpha = 1.0) and is dropped",
                               "ridge alpha = 1.0 on standardised predictors, 5-fold KFold shuffle seed 0",
                               "nuisance regression and predictor scaling are fitted inside each training fold",
                               "CV Spearman is undefined for constant or numerically zero predictions; this commonly occurs when a baseline uses only nuisance variables",
                               "partial rho is undefined when controls exhaust a ranked variable; percentile CIs are undefined if any bootstrap draw is degenerate (n_boot_valid records usable draws)",
                               "bootstrap draws and nuisance fits are shared within each complete predictor family",
                               "bootstrap RNG is seeded per setting with SeedSequence([seed, first four SHA256 bytes of setting name as little-endian integer]); results do not depend on setting execution order",
                               "Holm and BH adjustments apply within each setting/target/control family (up to 19 predictors); they do not test whether the top-ranked predictor exceeds the runner-up"]},
              open(os.path.join(out, "analysis_meta.json"), "w"), indent=1)

    # ---- figure: crowding partial rho with CIs, settings x metrics ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, tgt in zip(axes, TARGETS):
            sub = part[(part.target == tgt) & (part.predictor == "crowding")]
            names = list(settings)
            xs = np.arange(len(names))
            w = 0.26
            for j, (cname, col) in enumerate([("none", "#2a78d6"), ("robust", "#6a4c93"), ("primary", "#1baf7a")]):
                v = [sub[(sub.setting == s) & (sub.control == cname)].iloc[0] for s in names]
                ax.bar(xs + (j - 1) * w, [r.rho for r in v], w, color=col, label=f"control: {cname}")
                ax.errorbar(xs + (j - 1) * w, [r.rho for r in v],
                            yerr=[[r.rho - r.ci_lo for r in v], [r.ci_hi - r.rho for r in v]],
                            fmt="none", ecolor="#333", elinewidth=1, capsize=3)
            ax.axhline(0, color="#555", lw=1)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=15)
            ax.set_title(f"crowding vs {tgt}")
            ax.set_ylabel("Spearman rho (partial where controlled)")
        axes[0].legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "crowding_partials.png"), dpi=160)
    except Exception as e:
        print("figure skipped:", e)
    print("wrote", out)


if __name__ == "__main__":
    main()
