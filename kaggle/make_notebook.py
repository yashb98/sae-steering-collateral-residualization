#!/usr/bin/env python
"""Build kaggle/eda.ipynb: EDA and figures over the per-feature results, plus a live GPT-2 and
Pythia reproduction on the Kaggle GPU. `--run-local` executes the figure cells against results/
and writes the PNGs to a directory for inspection.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATASET_SLUG = "sae-collateral-residualization-results"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


INTRO = """# SAE steering side effects: collateral-side residualization, EDA and figures

Companion notebook for the four-setting extension of *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (Duan, [arXiv:2606.08365](https://arxiv.org/abs/2606.08365)). The paper controls its stability labels for effect magnitude, intervention value and natural activation but never applies that control to its collateral labels. The attached dataset holds the per-feature predictors and steering labels measured from scratch in all four of the paper's settings (GPT-2-small, Pythia-70M-deduped, Gemma-2-2B, Llama-3.1-8B), and this notebook walks through the data and the controlled results. Code and write-up: [github.com/yashb98/sae-steering-collateral-residualization](https://github.com/yashb98/sae-steering-collateral-residualization).

The last section re-runs the GPT-2-small and Pythia settings on this Kaggle GPU and compares the headline statistics with the dataset. Gemma-2-2B and Llama-3.1-8B are not recomputed here: both are gated on Hugging Face and the 8B model does not fit the Kaggle session; their rows come from the dataset."""

SETUP = r'''import glob, json, os, warnings
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
warnings.filterwarnings("ignore")

# ---- data location: the Kaggle dataset when attached, otherwise a local results/ directory ----
hits = glob.glob("/kaggle/input/**/gpt2_small/per_feature.csv", recursive=True) + glob.glob("results/gpt2_small/per_feature.csv") + glob.glob("../results/gpt2_small/per_feature.csv")
if not hits:
    raise FileNotFoundError("results dataset not found; /kaggle/input contains: " + str(glob.glob("/kaggle/input/**", recursive=True)[:40]))
DATA = os.path.dirname(os.path.dirname(hits[0]))
print("data:", DATA)

ORDER = ["gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"]
PRETTY = {"gpt2_small": "GPT-2-small", "pythia_70m_deduped": "Pythia-70M", "gemma_2_2b": "Gemma-2-2B", "llama_3_1_8b": "Llama-3.1-8B"}
COLOR = {"gpt2_small": "#2a78d6", "pythia_70m_deduped": "#eb6834", "gemma_2_2b": "#1baf7a", "llama_3_1_8b": "#4a3aa7"}
CONTROL_COLOR = {"none": "#86b6ef", "robust": "#2a78d6", "primary": "#104281"}
CONTROL_LABEL = {"none": "no control", "robust": "frequency + activation magnitude", "primary": "Section 3.8 set + frequency"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bluegrayred", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False, "font.family": "sans-serif"})

part = pd.read_csv(os.path.join(DATA, "analysis", "partial_correlations.csv"))
settings = [s for s in ORDER if os.path.exists(os.path.join(DATA, s, "per_feature.csv")) and s in set(part.setting)]
feat = {s: pd.read_csv(os.path.join(DATA, s, "per_feature.csv")) for s in settings}
meta = {s: json.load(open(os.path.join(DATA, s, "meta.json"))) for s in settings}
cvr = pd.read_csv(os.path.join(DATA, "analysis", "residualized_cv_ridge.csv"))
seeds = pd.read_csv(os.path.join(DATA, "analysis", "seed_summary.csv"))
xcheck_path = os.path.join(DATA, "crosscheck_kaggle_t4", "headline.json")
xcheck = json.load(open(xcheck_path)) if os.path.exists(xcheck_path) else None
gate_path = os.path.join(DATA, "gpt2_small_v1", "meta.json")
gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None

inv = pd.DataFrame([{"setting": PRETTY[s], "features": len(feat[s]), "eligible": meta[s]["sizes"]["n_eligible"],
                     "d_sae": meta[s]["sizes"]["d_sae_primary"], "mean L0 (primary)": meta[s]["sizes"].get("mean_l0_primary"),
                     "panel": meta[s]["sizes"]["panel"], "dtype": meta[s]["dtype"], "wall clock (s)": meta[s]["wall_clock_s"],
                     "device": meta[s]["device"]} for s in settings]).set_index("setting")
inv'''

EDA_MD = """## 1. What the sampled features look like

Each setting has 300 features sampled with seed 0 from the final-token firing-frequency band [0.002, 0.50]. The predictors are computed before any steering; the labels come from steering each feature additively (alpha = 1.0) at the final token of 48 mixed contexts and measuring what moves downstream."""

DESCRIBE = r'''KEY = ["crowding", "frequency", "act_mag", "effect_l2", "collateral_raw", "collateral_ctilde"]
LABEL = {"crowding": "decoder crowding (top-20 mean |cos|)", "frequency": "firing frequency", "act_mag": "mean activation",
         "effect_l2": "effect magnitude E_f", "collateral_raw": "collateral count C", "collateral_ctilde": "C-tilde = C / E_f"}
rows = []
for s in settings:
    q = feat[s][KEY].quantile([0.1, 0.5, 0.9]).T
    for k in KEY:
        rows.append({"setting": PRETTY[s], "variable": LABEL[k], "p10": q.loc[k, 0.1], "median": q.loc[k, 0.5], "p90": q.loc[k, 0.9]})
summary = pd.DataFrame(rows).set_index(["variable", "setting"]).unstack("setting").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
summary.round(3)'''

HIST = r'''LOGX = {"frequency", "effect_l2", "collateral_ctilde"}
fig, axes = plt.subplots(len(KEY), len(settings), figsize=(3.1 * len(settings), 2.0 * len(KEY)), sharey=False)
axes = np.atleast_2d(axes)
for j, s in enumerate(settings):
    for i, k in enumerate(KEY):
        ax = axes[i, j]
        v = feat[s][k].to_numpy()
        if k in LOGX:
            v = np.log10(v[v > 0] + 1e-12)
        ax.hist(v, bins=30, color=COLOR[s], edgecolor=SURFACE, linewidth=0.6)
        ax.set_yticks([])
        ax.grid(False)
        if i == 0:
            ax.set_title(PRETTY[s])
        if j == 0:
            ax.set_ylabel(LABEL[k] + (" (log10)" if k in LOGX else ""), fontsize=8.5, rotation=0, ha="right", va="center", labelpad=6)
fig.suptitle("Distributions of the main predictors and labels, 300 features per setting", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()'''

HEATMAP = r'''PRED = ["crowding", "crowd_max", "dec_norm", "enc_norm", "enc_dec_cos", "frequency", "act_mag", "act_std", "act_kurtosis",
        "coact_entropy", "coact_count", "logit_l2", "logit_linf", "logit_entropy"]
LAB = ["collateral_raw", "collateral_ctilde", "effect_l2", "stab_signed", "stab_abs", "kl_mean"]
cols = PRED + LAB
fig, axes = plt.subplots(1, len(settings), figsize=(5.4 * len(settings), 5.6))
axes = np.atleast_1d(axes)
for ax, s in zip(axes, settings):
    C = feat[s][cols].corr(method="spearman").to_numpy()
    im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    ax.grid(False)
    ax.axhline(len(PRED) - 0.5, color=INK, lw=0.8); ax.axvline(len(PRED) - 0.5, color=INK, lw=0.8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(C[i, j]) >= 0.5:
                ax.text(j, i, f"{C[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color=INK if abs(C[i, j]) < 0.75 else SURFACE)
    ax.set_title(f"{PRETTY[s]}: Spearman correlations", loc="left")
cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.012, pad=0.01)
cb.set_label("Spearman rho", color=INK2)
fig.suptitle("Predictors (top-left block) and steering labels (bottom-right block); the off-diagonal block is what the analysis is about", x=0.01, ha="left", color=INK, fontsize=12)
plt.show()'''

SCATTER_MD = """## 2. Crowding versus collateral

The paper's headline (Table 2) is that decoder crowding predicts the raw downstream collateral count in GPT-2-small (rho = 0.466). Section 3.4 names C-tilde, the count per unit of logit effect, as the primary collateral metric. Both are shown."""

SCATTER = r'''fig, axes = plt.subplots(2, len(settings), figsize=(3.6 * len(settings), 6.6))
axes = np.atleast_2d(axes)
for j, s in enumerate(settings):
    df = feat[s]
    for i, (tgt, name, logy) in enumerate([("collateral_raw", "collateral count C", False), ("collateral_ctilde", "C-tilde = C / E_f", True)]):
        ax = axes[i, j]
        y = df[tgt].to_numpy()
        ax.scatter(df.crowding, y, s=22, color=COLOR[s], alpha=0.75, edgecolors=SURFACE, linewidths=0.8)
        if logy:
            ax.set_yscale("symlog", linthresh=max(np.percentile(y[y > 0], 5), 1e-3))
        r, p = stats.spearmanr(df.crowding, y)
        ax.text(0.02, 0.97, f"rho = {r:+.3f}", transform=ax.transAxes, va="top", color=INK, fontsize=10)
        if i == 0:
            ax.set_title(PRETTY[s])
        if j == 0:
            ax.set_ylabel(name)
        if i == 1:
            ax.set_xlabel("decoder crowding")
fig.suptitle("Decoder crowding against both collateral metrics (Spearman rho, n = 300 per panel)", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()'''

MEDIATION = r'''fig, axes = plt.subplots(2, len(settings), figsize=(3.6 * len(settings), 6.4))
axes = np.atleast_2d(axes)
for j, s in enumerate(settings):
    df = feat[s]
    for i, (x, y, xl, yl) in enumerate([("crowding", "effect_l2", "decoder crowding", "effect magnitude E_f"),
                                        ("effect_l2", "collateral_raw", "effect magnitude E_f", "collateral count C")]):
        ax = axes[i, j]
        ax.scatter(df[x], df[y], s=22, color=COLOR[s], alpha=0.75, edgecolors=SURFACE, linewidths=0.8)
        if x == "effect_l2":
            ax.set_xscale("log")
        if y == "effect_l2":
            ax.set_yscale("log")
        r, _ = stats.spearmanr(df[x], df[y])
        ax.text(0.02, 0.97, f"rho = {r:+.3f}", transform=ax.transAxes, va="top", color=INK, fontsize=10)
        if i == 0:
            ax.set_title(PRETTY[s])
        if j == 0:
            ax.set_ylabel(yl)
        ax.set_xlabel(xl)
fig.suptitle("Effect magnitude correlates with collateral; its relationship to crowding varies by setting", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()'''

CONTROL_MD = """## 3. The control

Partial Spearman correlations: predictor and label are rank-transformed, both are residualized on the ranked control variables, and the residuals are correlated. Controls: none; the robust pair (frequency, activation magnitude); the primary set (the paper's Section 3.8 nuisance variables, effect magnitude E_f, intervention value and natural activation, plus firing frequency). Whiskers are 95% bootstrap intervals over the 300 features (10,000 resamples)."""

PARTIALS = r'''fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
controls = ["none", "robust", "primary"]
w = 0.24
for ax, (tgt, name) in zip(axes, [("collateral_raw", "collateral count C"), ("collateral_ctilde", "C-tilde = C / E_f")]):
    sub = part[(part.target == tgt) & (part.predictor == "crowding")]
    xs = np.arange(len(settings))
    for k, c in enumerate(controls):
        rows = [sub[(sub.setting == s) & (sub.control == c)].iloc[0] for s in settings]
        v = [r.rho for r in rows]
        ax.bar(xs + (k - 1) * w, v, w - 0.03, color=CONTROL_COLOR[c], label=CONTROL_LABEL[c], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xs + (k - 1) * w, v, yerr=[[r.rho - r.ci_lo for r in rows], [r.ci_hi - r.rho for r in rows]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        if c == "primary":
            for x, r in zip(xs + w, rows):
                ax.annotate(f"{r.rho:+.2f}", (x, r.ci_hi), ha="center", va="bottom", fontsize=9, color=INK2, xytext=(0, 3), textcoords="offset points")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([PRETTY[s] for s in settings])
    ax.set_title(f"crowding vs {name}", loc="left")
axes[0].set_ylabel("Spearman rho (partial where controlled)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, fontsize=9)
fig.suptitle("Crowding partial correlations vary across the four model/SAE settings", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.88))
plt.show()'''

TOP_PREDICTORS = r'''fig, axes = plt.subplots(2, len(settings), figsize=(3.9 * len(settings), 6.8))
axes = np.atleast_2d(axes)
for j, s in enumerate(settings):
    for i, (tgt, name) in enumerate([("collateral_raw", "collateral count C"), ("collateral_ctilde", "C-tilde")]):
        ax = axes[i, j]
        sub = part[(part.setting == s) & (part.target == tgt) & (part.control == "primary")]
        sub = sub.dropna(subset=["rho"]).sort_values("rho", key=lambda v: v.abs(), ascending=False).head(6).iloc[::-1]
        ys = np.arange(len(sub))
        ax.barh(ys, sub.rho, 0.6, color=COLOR[s], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(sub.rho, ys, xerr=[sub.rho - sub.ci_lo, sub.ci_hi - sub.rho], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        ax.set_yticks(ys); ax.set_yticklabels(sub.predictor, fontsize=8.5)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlim(-0.75, 0.75)
        if i == 0:
            ax.set_title(PRETTY[s])
        if j == 0:
            ax.set_ylabel(name)
        if i == 1:
            ax.set_xlabel("partial Spearman rho, primary control")
fig.suptitle("Strongest six predictors after the primary control; the leading predictor changes with the setting", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()'''

CV_MD = """## 4. Predictor sets on the residualized target

The paper's Table B3 residualizes the stability labels and re-runs its cross-validated ridge regressions on the residuals. Here the nuisance model is fitted inside each training fold and applied to that fold's held-out features; ridge predicts these collateral residuals (alpha = 1, predictor scaling fitted on the training fold) under five-fold cross-validation. Score = Spearman between held-out predictions and the residualized label, mean over folds, whiskers = fold standard deviation. Undefined scores from constant residualized predictions are omitted; nuisance-only linear baselines have no training-residual signal."""

CV_RIDGE = r'''SETS = ["frequency_only", "actmag_only", "coactivation_only", "direct_logit_only", "geometry_only", "full_no_magnitude", "full_all"]
fig, axes = plt.subplots(2, len(settings), figsize=(3.9 * len(settings), 6.4), sharex=True)
axes = np.atleast_2d(axes)
for j, s in enumerate(settings):
    for i, (tgt, name) in enumerate([("collateral_raw", "collateral count C"), ("collateral_ctilde", "C-tilde")]):
        ax = axes[i, j]
        sub = cvr[(cvr.setting == s) & (cvr.target == tgt) & (cvr.control == "primary")].set_index("predictor_set").loc[SETS]
        ys = np.arange(len(SETS))
        ax.barh(ys, sub.cv_spearman_mean, 0.6, color=COLOR[s], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(sub.cv_spearman_mean, ys, xerr=sub.cv_spearman_sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        ax.set_yticks(ys); ax.set_yticklabels([k.replace("_", " ") for k in SETS], fontsize=8.5)
        ax.axvline(0, color=AXIS, lw=1)
        if i == 0:
            ax.set_title(PRETTY[s])
        if j == 0:
            ax.set_ylabel(name)
        if i == 1:
            ax.set_xlabel("CV Spearman on the residualized label")
fig.suptitle("Predictor-family comparison with nuisance fitting and scaling inside each training fold", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()'''

ROBUST_MD = """## 5. Robustness: feature samples, machines, protocol

Three checks. Different samples of 300 features and random-context selections (seeds 0, 1, 2, same underlying context pool). The same code on a Kaggle T4 with an older library stack. And the change from the original notebook's context construction (protocol v1, which left 33 of 2,048 GPT-2-small contexts ending in a pad token) to the reported protocol v2."""

ROBUST = r'''fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
# seeds
ax = axes[0]
sd = seeds[(seeds.target == "collateral_raw") & (seeds.n_seeds >= 2)]
xs = 0
ticks, labels = [], []
for s in [s for s in settings if s in set(sd.setting)]:
    for c in ["none", "robust", "primary"]:
        r = sd[(sd.setting == s) & (sd.control == c)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        vals = [r[k] for k in ["0", "1", "2"] if k in r and pd.notna(r[k])]
        ax.scatter([xs] * len(vals), vals, s=40, color=CONTROL_COLOR[c], edgecolors=SURFACE, linewidths=1, zorder=3)
        ticks.append(xs); labels.append(f"{PRETTY[s]}\n{c}")
        xs += 1
    xs += 0.6
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5, rotation=90)
ax.set_ylabel("crowding vs collateral count, rho")
ax.set_title("Three feature samples per setting", loc="left")
# cross-machine
ax = axes[1]
if xcheck:
    keys = ["rho_crowding__collateral_raw", "partial_crowding__collateral_raw__given_freq_actmag", "rho_crowding__collateral_ctilde", "partial_crowding__collateral_ctilde__given_freq_actmag"]
    short = ["raw", "raw, partial", "C-tilde", "C-tilde, partial"]
    xs = np.arange(len(keys))
    for k, s in enumerate([s for s in settings if s in xcheck["headline"]]):
        a = [meta[s]["headline"][key] for key in keys]
        b = [xcheck["headline"][s][key] for key in keys]
        ax.scatter(xs + k * 0.25 - 0.12, a, s=44, color=COLOR[s], edgecolors=SURFACE, linewidths=1, label=f"{PRETTY[s]}, GB10", zorder=3)
        ax.scatter(xs + k * 0.25 - 0.12, b, s=44, facecolors=SURFACE, edgecolors=COLOR[s], linewidths=1.6, label=f"{PRETTY[s]}, Kaggle T4", zorder=3)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=8.5)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_title("Same code, two machines", loc="left")
# protocol
ax = axes[2]
if gate:
    keys = ["rho_crowding__collateral_raw", "partial_crowding__collateral_raw__given_freq_actmag", "rho_frequency__collateral_raw", "rho_act_mag__collateral_raw"]
    short = ["crowding", "crowding, partial", "frequency", "activation magnitude"]
    xs = np.arange(len(keys))
    v1 = [gate["headline"][k] for k in keys]
    v2 = [meta["gpt2_small"]["headline"][k] for k in keys]
    ax.bar(xs - 0.18, v1, 0.32, color="#86b6ef", edgecolor=SURFACE, linewidth=2, label="protocol v1 (original notebook)")
    ax.bar(xs + 0.18, v2, 0.32, color=COLOR["gpt2_small"], edgecolor=SURFACE, linewidth=2, label="protocol v2 (reported)")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=8.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.72)
    ax.set_title("GPT-2-small, context protocol", loc="left")
fig.suptitle("Robustness checks", x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()'''

VARIANTS_MD = '''## 6. Context reproducibility, coefficients and paired random directions

The even/odd context-pool split uses the same features and predictor estimates. It measures reproducibility within one corpus pool, not a universal predictor ceiling. Coefficient sweeps cover GPT-2 and Pythia; q95 uses the activation percentile conditional on firing. Paired random directions are normalized and scaled to each SAE decoder norm, then steered on exactly the same contexts. Both arms use effect-only adjustment because feature frequency is undefined for random directions. Intervals are pointwise feature-bootstrap intervals, not tests of a between-arm difference.'''

VARIANTS = r'''variants = json.load(open(os.path.join(DATA, "analysis", "variants.json")))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
labels = [("collateral_raw", "count"), ("collateral_ctilde", "C-tilde"), ("kl_mean", "KL")]
for i, s in enumerate(settings):
    for j, (label, short) in enumerate(labels):
        r = variants["reliability"][s]["labels"][label]
        x = i + (j - 1) * 0.2
        axes[0].errorbar(x, r["rho"], yerr=[[r["rho"]-r["ci_lo"]], [r["ci_hi"]-r["rho"]]],
                         fmt=["o", "s", "^"][j], color=COLOR[s], capsize=2, label=short if i == 0 else None)
axes[0].set_xticks(range(len(settings))); axes[0].set_xticklabels([PRETTY[s] for s in settings], rotation=15)
axes[0].set_ylabel("Spearman between disjoint context pools")
axes[0].set_ylim(0, 1.05); axes[0].set_title("Label reproducibility"); axes[0].legend()
for s in settings[:2]:
    for x, label in enumerate(["alpha0.5", "alpha1", "alpha2", "alpha4", "q95 firing"]):
        r = variants["alpha"][s][label]["collateral_raw|primary"]
        axes[1].errorbar(x, r["rho"], yerr=[[r["rho"]-r["ci_lo"]], [r["ci_hi"]-r["rho"]]],
                         fmt="o", color=COLOR[s], capsize=3, label=PRETTY[s] if x == 0 else None)
axes[1].set_xticks(range(5)); axes[1].set_xticklabels(["0.5", "1", "2", "4", "q95"])
axes[1].set_xlabel("additive coefficient"); axes[1].set_ylabel("crowding vs count, primary partial rho")
axes[1].set_title("Coefficient sensitivity"); axes[1].legend(); axes[1].axhline(0, color=AXIS)
for i, s in enumerate(settings):
    for j, (arm, marker) in enumerate([("SAE features", "o"), ("paired random directions", "s")]):
        r = variants["random"][s][arm]["collateral_raw|effect_only"]
        axes[2].errorbar(i+(j-0.5)*0.2, r["rho"], yerr=[[r["rho"]-r["ci_lo"]], [r["ci_hi"]-r["rho"]]],
                         fmt=marker, color=COLOR[s], capsize=3, label=arm if i == 0 else None)
axes[2].set_xticks(range(len(settings))); axes[2].set_xticklabels([PRETTY[s] for s in settings], rotation=15)
axes[2].set_ylabel("crowding vs count, effect-only partial rho")
axes[2].set_title("Same contexts and decoder norms"); axes[2].legend(); axes[2].axhline(0, color=AXIS)
plt.tight_layout(); plt.show()'''

RESIDUAL_MD = '''## 7. Change outside the SAE reconstruction

For downstream change dh and reconstruction change dr, the unrepresented change is de = dh - dr. The explained-change score is 1 - sum(||de||²) / sum(||dh||²) across contexts for each feature. Negative scores mean predicting zero change beats using dr. Because dr and de need not be orthogonal, this is not a nonnegative variance partition. The separate norm ratio does not measure explanation. These hidden-state quantities do not establish behavioral harm.'''

RESIDUAL = r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for s in settings:
    d = feat[s]
    axes[0].scatter(d.resid_reconstruction_norm_ratio, d.resid_change_explained_energy,
                    s=14, alpha=0.55, color=COLOR[s], label=PRETTY[s])
    axes[1].scatter(d.crowding, d.resid_error_delta_norm, s=14, alpha=0.55, color=COLOR[s])
axes[0].axhline(0, color=AXIS); axes[0].set_yscale("symlog", linthresh=1)
axes[0].set_xlabel("reconstruction norm ratio"); axes[0].set_ylabel("explained-change score (symlog)")
axes[0].set_title("A norm ratio is not an explained fraction"); axes[0].legend()
axes[1].set_xlabel("decoder crowding"); axes[1].set_ylabel("mean norm of unrepresented change")
axes[1].set_title("Residual reconstruction-error change")
plt.tight_layout(); plt.show()'''

LIVE_MD = """## 8. Live reproduction on this GPU

The two ungated settings are recomputed here from scratch with the same script as the dataset (protocol v2, seed 0) and compared with the dataset's headline statistics. GPT-2-small takes a few minutes on a T4, Pythia-70M about one."""

CLOSING = """## Notes

Natural activation is taken as the mean final-token activation over the 2,048 contexts, and the intervention value is constant under fixed_global_add with alpha = 1.0, so it drops out of the regression. Feature sampling uses seed 0 over the firing-frequency band [0.002, 0.50]. The downstream panel is the most frequently active downstream features on clean contexts (2,048; 1,024 for Llama). Gemma-2-2B and Llama-3.1-8B are loaded from Hugging Face under their gated licences on the original machine; Llama runs in bfloat16 without TransformerLens weight processing to fit in memory. Model, dictionary and loading choices vary together; mean L0 is a loading sanity check, not proof of preprocessing equivalence. Full write-up and LaTeX table: `RESULTS.md` in the repository."""


def live_cells():
    src = open(os.path.join(ROOT, "src", "run_setting.py")).read()
    cfgs = {n: open(os.path.join(ROOT, "configs", f"{n}.yaml")).read() for n in ["pythia_70m_deduped", "gpt2_small"]}
    write = "\n".join("open('configs/%s.yaml', 'w').write(json.loads(%s))" % (n, json.dumps(json.dumps(c))) for n, c in cfgs.items())
    c1 = ("import subprocess, sys, os, json, time\n"
          "t0 = time.time()\n"
          "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformer-lens<3', 'sae-lens<6', 'transformers<5', 'pyyaml'], check=True)\n"
          "import importlib.metadata as im\n"
          "print('installed transformer-lens', im.version('transformer-lens'), '| sae-lens', im.version('sae-lens'), '| transformers', im.version('transformers'), '| torch', im.version('torch'), f'({time.time()-t0:.0f} s)')\n"
          "os.makedirs('src', exist_ok=True); os.makedirs('configs', exist_ok=True); os.makedirs('results/logs', exist_ok=True)\n"
          "open('src/run_setting.py', 'w').write(json.loads(%s))\n%s\n"
          "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'" % (json.dumps(json.dumps(src)), write))
    c2 = "subprocess.run([sys.executable, 'src/run_setting.py', '--config', 'configs/pythia_70m_deduped.yaml', '--protocol', 'v2', '--out', 'results/live_pythia'], check=True)"
    c3 = "subprocess.run([sys.executable, 'src/run_setting.py', '--config', 'configs/gpt2_small.yaml', '--protocol', 'v2', '--out', 'results/live_gpt2'], check=True)"
    c4 = r'''rows = []
for s, live in [("pythia_70m_deduped", "results/live_pythia/meta.json"), ("gpt2_small", "results/live_gpt2/meta.json")]:
    if not os.path.exists(live):
        raise FileNotFoundError(live)
    L = json.load(open(live))
    for k, v in meta[s]["headline"].items():
        rows.append({"setting": PRETTY[s], "statistic": k, "dataset (GB10)": v, "this Kaggle run": L["headline"][k], "difference": L["headline"][k] - v})
    print(PRETTY[s], "| eligible features here:", L["sizes"]["n_eligible"], "vs dataset:", meta[s]["sizes"]["n_eligible"],
          "| mean L0 here:", L["sizes"].get("mean_l0_primary"), "vs dataset:", meta[s]["sizes"].get("mean_l0_primary"), "| device:", L["device"])
pd.DataFrame(rows).round(3)'''
    return [code(c1), code(c2), code(c3), code(c4)]


def build():
    cells = [md(INTRO), code(SETUP), md(EDA_MD), code(DESCRIBE), code(HIST), code(HEATMAP), md(SCATTER_MD), code(SCATTER),
             code(MEDIATION), md(CONTROL_MD), code(PARTIALS), code(TOP_PREDICTORS), md(CV_MD), code(CV_RIDGE), md(ROBUST_MD),
             code(ROBUST), md(VARIANTS_MD), code(VARIANTS), md(RESIDUAL_MD), code(RESIDUAL), md(LIVE_MD)] + live_cells() + [md(CLOSING)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(HERE, "eda.ipynb"), "w"), indent=1)
    print("wrote", os.path.join(HERE, "eda.ipynb"), len(cells), "cells")


def run_local(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    os.chdir(ROOT)
    counter = {"n": 0}

    def show():
        counter["n"] += 1
        plt.savefig(os.path.join(outdir, f"fig{counter['n']:02d}.png"), dpi=110, bbox_inches="tight")
        plt.close("all")
    plt.show = show
    g = {}
    for name, src in [("setup", SETUP), ("describe", DESCRIBE), ("hist", HIST), ("heatmap", HEATMAP), ("scatter", SCATTER),
                      ("mediation", MEDIATION), ("partials", PARTIALS), ("top", TOP_PREDICTORS), ("cv", CV_RIDGE), ("robust", ROBUST), ("variants", VARIANTS), ("residual", RESIDUAL)]:
        exec(compile(src, name, "exec"), g)
        print("ok:", name)
    print("figures:", counter["n"], "->", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-local", metavar="OUTDIR")
    a = ap.parse_args()
    if a.run_local:
        run_local(a.run_local)
    else:
        build()
