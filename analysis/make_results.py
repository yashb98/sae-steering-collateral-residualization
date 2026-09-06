#!/usr/bin/env python
"""Build RESULTS.md and a LaTeX table from the analysis CSVs and the per-setting meta.json files.

Every number in the write-up is read from results/, so regenerating after a new setting lands is
one command: python analysis/make_results.py
"""
import glob
import json
import os

import pandas as pd

ORDER = ["gpt2_small", "pythia_70m_deduped", "gemma_2_2b", "llama_3_1_8b"]
PRETTY = {"gpt2_small": "GPT-2-small", "pythia_70m_deduped": "Pythia-70M-deduped",
          "gemma_2_2b": "Gemma-2-2B", "llama_3_1_8b": "Llama-3.1-8B"}
TGT = {"collateral_raw": "raw downstream count C_{f,0.05}", "collateral_ctilde": "C-tilde = C / (E_f + eps)"}
KAGGLE = {"rho_crowding__collateral_raw": 0.548, "partial_crowding__collateral_raw__given_freq_actmag": 0.569,
          "rho_crowding__collateral_ctilde": 0.421, "partial_crowding__collateral_ctilde__given_freq_actmag": 0.416,
          "rho_frequency__collateral_raw": 0.248, "rho_frequency__collateral_ctilde": 0.146,
          "rho_act_mag__collateral_raw": 0.230, "crowd_rho_lowfreq_half": 0.651, "crowd_rho_highfreq_half": 0.449}
PAPER_TABLE2 = {"gpt2_small": "decoder crowding, downstream count, rho = 0.466",
                "pythia_70m_deduped": "direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360)",
                "gemma_2_2b": "encoder norm, signed stability, rho = 0.350",
                "llama_3_1_8b": "direct-logit L2, downstream count, rho = 0.464"}


def fmt(r):
    return f"{r.rho:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"


def main(results="results"):
    part = pd.read_csv(os.path.join(results, "analysis", "partial_correlations.csv"))
    cv = pd.read_csv(os.path.join(results, "analysis", "residualized_cv_ridge.csv"))
    seeds = None
    sp = os.path.join(results, "analysis", "seed_summary.csv")
    if os.path.exists(sp):
        seeds = pd.read_csv(sp)
    metas = {}
    for s in ORDER:
        p = os.path.join(results, s, "meta.json")
        if os.path.exists(p):
            metas[s] = json.load(open(p))
    settings = [s for s in ORDER if s in metas and s in set(part.setting)]
    amet = json.load(open(os.path.join(results, "analysis", "analysis_meta.json")))
    n_boot = amet["n_boot"]

    L = []
    L += ["# Collateral-side residualization across four model/SAE settings", "",
          "Companion results for the revision of *Pre-Intervention Prediction of Sparse Autoencoder Steering "
          "Side Effects* (Duan, arXiv:2606.08365). All numbers below are read from `results/` by "
          "`analysis/make_results.py`; nothing is copied from the paper except where labelled as the paper's value.", "",
          f"Settings included: {', '.join(PRETTY[s] for s in settings)}."
          + ("" if len(settings) == 4 else f" Still to run: {', '.join(PRETTY[s] for s in ORDER if s not in settings)}."), ""]

    # ---- 1. setup ----
    L += ["## 1. Setup per setting", "",
          "| Setting | Model | Primary SAE / hook | Downstream SAE / hook | Panel | Eligible features | Mean L0 (primary / downstream) | Contexts, texts dropped | dtype | Wall clock |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in settings:
        m = metas[s]; c = m["config"]; z = m["sizes"]; cb = m["context_build"]
        def sae_hook(sid, hook):
            return f"`{sid}`" if sid == hook else f"`{sid}` at `{hook}`"
        L.append(f"| {PRETTY[s]} | {c['model_name']} | {c['sae_release']} {sae_hook(c['primary_sae_id'], c['primary_hook'])} | "
                 f"{sae_hook(c['downstream_sae_id'], c['downstream_hook'])} | {z['panel']} | {z['n_eligible']} of {z['d_sae_primary']} | "
                 f"{z.get('mean_l0_primary', 'n/a')} / {z.get('mean_l0_downstream', 'n/a')} | {z['n_contexts']} x {z['seq_len']} tokens, "
                 f"{cb['short_texts_dropped']} short texts dropped | {m['dtype']} | {m['wall_clock_s']:.0f} s on {m['device']} |")
    L += ["", "Common: Wikitext-103 train split, 8,000 texts, 300 features sampled with seed 0 from the final-token firing-frequency "
          "band [0.002, 0.50], 16 contexts per type (top / random / low), additive steering alpha = 1.0 at the final token, "
          "tau = 0.05, epsilon_fire = 1e-6, crowding = top-20 mean absolute cosine, protocol v2 (no pad final tokens, "
          "whitespace-normalised dedup).", ""]

    # ---- 2. crowding ----
    L += ["## 2. Decoder crowding vs collateral, with and without controls", "",
          f"Spearman rho; partial correlations are rank-based (both sides OLS-residualised on the ranked controls). "
          f"95% CIs from {n_boot} bootstrap resamples of the 300 features. Controls: **primary** = effect magnitude E_f, "
          "intervention value (constant, dropped), natural activation (mean final-token activation), firing frequency; "
          "**robust** = frequency and activation magnitude only.", ""]
    for tgt, name in TGT.items():
        L += [f"### Target: {name}", "", "| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |",
              "|---|---|---|---|---|"]
        for s in settings:
            sub = part[(part.setting == s) & (part.target == tgt) & (part.predictor == "crowding")].set_index("control")
            L.append(f"| {PRETTY[s]} | {fmt(sub.loc['none'])} | {fmt(sub.loc['robust'])} | {fmt(sub.loc['primary'])} | {PAPER_TABLE2[s]} |")
        L.append("")

    # ---- 3. leading predictors ----
    L += ["## 3. Strongest predictors under the primary control", "",
          "Top three predictors by absolute partial rho after the primary control, per setting and metric.", ""]
    for tgt, name in TGT.items():
        L += [f"### Target: {name}", "", "| Setting | Predictor | partial rho [95% CI] | p |", "|---|---|---|---|"]
        for s in settings:
            sub = part[(part.setting == s) & (part.target == tgt) & (part.control == "primary")]
            sub = sub.iloc[(-sub.rho.abs()).argsort()[:3]]
            for _, r in sub.iterrows():
                L.append(f"| {PRETTY[s]} | {r.predictor} | {fmt(r)} | {r.p:.1e} |")
        L.append("")

    # ---- 4. CV ridge ----
    L += ["## 4. Table B3 analog: predictor sets on the residualized collateral target", "",
          "The collateral label is OLS-residualised against the primary control set, then predicted with ridge "
          "(alpha = 1, standardised predictors) under 5-fold cross-validation; score = Spearman between held-out "
          "predictions and residualised label, mean over folds (sd in parentheses).", ""]
    sets = ["frequency_only", "actmag_only", "geometry_only", "direct_logit_only", "coactivation_only", "full_no_magnitude", "full_all"]
    for tgt, name in TGT.items():
        L += [f"### Target: {name}", "", "| Setting | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
        for s in settings:
            sub = cv[(cv.setting == s) & (cv.target == tgt) & (cv.control == "primary")].set_index("predictor_set")
            L.append(f"| {PRETTY[s]} | " + " | ".join(f"{sub.loc[k].cv_spearman_mean:+.3f} ({sub.loc[k].cv_spearman_sd:.2f})" for k in sets) + " |")
        L.append("")

    # ---- 5. seeds ----
    if seeds is not None:
        L += ["## 5. Feature-sample robustness (crowding)", "",
              "Same contexts, different random sample of 300 features (seeds 0, 1, 2). Values are crowding vs collateral rho.", "",
              "| Setting | Target | Control | seed 0 | seed 1 | seed 2 | mean | sd |", "|---|---|---|---|---|---|---|---|"]
        for _, r in seeds.iterrows():
            if r.setting not in settings:
                continue
            L.append(f"| {PRETTY[r.setting]} | {r.target} | {r.control} | {r.get('0', float('nan')):+.3f} | {r.get('1', float('nan')):+.3f} | "
                     f"{r.get('2', float('nan')):+.3f} | {r['mean']:+.3f} | {r['sd']:.3f} |")
        L.append("")

    # ---- 6. gate + protocol ----
    g = os.path.join(results, "gpt2_small_v1", "meta.json")
    if os.path.exists(g):
        gm = json.load(open(g)); h = gm["headline"]
        L += ["## 6. Regression gate against the published GPT-2-small notebook", "",
              "`src/run_setting.py --protocol v1` reproduces the Kaggle notebook's context construction exactly. Same seed, same "
              "eligible-feature count (" + str(gm["sizes"]["n_eligible"]) + "). Differences are GPU floating-point noise (T4 vs GB10).", "",
              "| Statistic | Kaggle notebook (Aug 7 version) | this code, protocol v1 |", "|---|---|---|"]
        for k, v in KAGGLE.items():
            L.append(f"| {k} | {v:+.3f} | {h[k]:+.3f} |")
        L += ["", f"Protocol v1 leaves {gm['context_build']['contexts_final_token_is_pad']} of {gm['sizes']['n_contexts']} contexts with a pad token in the "
              "final position (short texts are right-padded). Protocol v2, used for every reported setting, drops texts shorter than "
              "48 tokens instead. On GPT-2-small this moves crowding vs raw count from "
              f"{h['rho_crowding__collateral_raw']:+.3f} (v1) to {metas['gpt2_small']['headline']['rho_crowding__collateral_raw']:+.3f} (v2) and the "
              f"frequency baseline from {h['rho_frequency__collateral_raw']:+.3f} to {metas['gpt2_small']['headline']['rho_frequency__collateral_raw']:+.3f}; "
              "the feature sample also changes because the eligible set changes.", ""]

    # ---- 7. assumptions ----
    L += ["## 7. Assumptions and conventions", ""]
    for a in amet["assumptions"]:
        L.append(f"- {a}.")
    L += ["- Feature sampling band [0.002, 0.50] on final-token firing frequency; the paper says only \"the non-degenerate range\".",
          "- Downstream panel = the most frequently active downstream features on the clean contexts (2048; 1024 for Llama).",
          "- Models loaded with TransformerLens default weight processing plus each SAE's `model_from_pretrained_kwargs`, in float32. "
          "TransformerLens centres writing weights only for LayerNorm models, so Gemma-2 and Llama residual streams match the "
          "Hugging Face values the Gemma Scope and Llama Scope SAEs were trained on; the mean L0 column above is the empirical check.",
          "- Direct-logit predictors use the model's unembedding as loaded (centred for GPT-2, Pythia and Llama; Gemma-2 keeps its logit softcap, so its unembedding is not centred).",
          "- Effect magnitude E_f and the stability cosines are computed on final-token logit differences in float32.", ""]
    L += ["## Files", "", "- `results/<setting>/per_feature.csv`: one row per sampled feature, all predictors and labels.",
          "- `results/<setting>/selection.json`: sampled feature indices and downstream panel indices.",
          "- `results/<setting>/meta.json`: resolved config, sizes, versions, timings, headline statistics.",
          "- `results/analysis/partial_correlations.csv`, `residualized_cv_ridge.csv`, `seed_summary.csv`, `summary.md`, `crowding_partials.png`.",
          "- `results/analysis/table_collateral_residualized.tex`: LaTeX rows for the appendix.", ""]
    open("RESULTS.md", "w").write("\n".join(L))

    # ---- LaTeX rows ----
    T = ["% Collateral-side residualization, generated by analysis/make_results.py",
         "\\begin{tabular}{llccc}", "\\toprule",
         "Model & Collateral label & Raw $\\rho$ & Partial $\\rho$ (freq, act.\\ mag.) & Partial $\\rho$ (Sec.~3.8 set + freq) \\\\", "\\midrule"]
    for s in settings:
        for tgt, lab in [("collateral_raw", "downstream count"), ("collateral_ctilde", "$\\widetilde{C}_{f,0.05}$")]:
            sub = part[(part.setting == s) & (part.target == tgt) & (part.predictor == "crowding")].set_index("control")
            def tex(r):
                return f"{r.rho:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
            T.append(f"{PRETTY[s]} & {lab} & {tex(sub.loc['none'])} & {tex(sub.loc['robust'])} & {tex(sub.loc['primary'])} \\\\")
    T += ["\\bottomrule", "\\end{tabular}",
          f"% Decoder crowding vs collateral, Spearman, n = 300 features per setting, 95\\% bootstrap CIs ({n_boot} resamples)."]
    open(os.path.join(results, "analysis", "table_collateral_residualized.tex"), "w").write("\n".join(T))
    print("wrote RESULTS.md and", os.path.join(results, "analysis", "table_collateral_residualized.tex"), "| settings:", settings)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
