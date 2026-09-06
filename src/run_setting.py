#!/usr/bin/env python
"""Phase 1 + Phase 2 for one model/SAE setting of arXiv 2606.08365 (Duan, 2026).

Computes intervention-free predictors for 300 sampled SAE features and then measures their
steering labels (collateral spread, effect magnitude, stability, KL shift) with additive
steering at the final token, alpha = 1.0. Everything is computed here from scratch; no number
is copied from the paper. Output: results/<name>/per_feature.csv plus meta.json.

This generalises the GPT-2-small Kaggle notebook
(kaggle.com/code/yashbishnoi98/sae-steering-side-effects-reproduced) to a per-setting config.

Protocols for building the 2,048 contexts:
  v1  exact replication of the notebook: texts are tokenised in batches of 32, a batch is
      skipped only if its LONGEST text is shorter than seq_len, and shorter texts inside a kept
      batch are right-padded. Used only as the regression gate on GPT-2-small.
  v2  per-context filter: a text is kept only if it has at least seq_len real tokens (BOS
      included), so the final token is never a pad token; texts are also deduplicated after
      whitespace normalisation (paper Section 3.2). Used for all reported settings.
"""
import argparse
import json
import os
import platform
import re
import sys
import time

import numpy as np
import torch
import yaml

T0 = time.time()
TIMINGS = {}


def tick(msg, key=None):
    t = time.time() - T0
    print(f"[{t:7.1f}s] {msg}", flush=True)
    if key:
        TIMINGS[key] = round(t, 1)


def _pkg_version(dist_name, module):
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:
        return getattr(module, "__version__", "unknown")


def sae_field(sae, name, default=None):
    """Read a config field across SAELens versions (cfg.metadata.<name>, cfg.<name>, dict)."""
    cfg = sae.cfg
    for obj in (getattr(cfg, "metadata", None), cfg):
        if obj is not None and hasattr(obj, name):
            v = getattr(obj, name)
            if v is not None:
                return v
    try:
        d = cfg.to_dict()
        if d.get(name) is not None:
            return d[name]
        md = d.get("metadata")
        if isinstance(md, dict) and md.get(name) is not None:
            return md[name]
    except Exception:
        pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dir (default results/<name>)")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes to validate the pipeline")
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v2")
    ap.add_argument("--dtype", default=None, help="override model dtype: float32|bfloat16|float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    out_dir = args.out or os.path.join("results", name + ("_smoke" if args.smoke else "") +
                                       ("_v1" if args.protocol == "v1" else ""))
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        N_TEXTS, N_CONTEXTS, N_FEATURES, CTX_PER_TYPE, PANEL = 600, 128, 12, 4, 256
    else:
        N_TEXTS, N_CONTEXTS, N_FEATURES = cfg["n_texts"], cfg["n_contexts"], cfg["n_features"]
        CTX_PER_TYPE, PANEL = cfg["ctx_per_type"], cfg["panel_size"]
    SEQ_LEN = cfg["seq_len"]
    ALPHA = float(cfg["alpha"])
    TAU = float(cfg["tau"])
    EPS_FIRE = float(cfg["eps_fire"])
    TOPK_CROWD = int(cfg["topk_crowding"])
    FREQ_LO, FREQ_HI = cfg["freq_band"]
    MIN_CHARS = int(cfg.get("min_text_chars", 200))
    MODEL_BATCH = int(cfg.get("model_batch", 32))
    SEED = args.seed
    dtype_name = args.dtype or cfg.get("dtype", "float32")
    dtype = getattr(torch, dtype_name)
    device = args.device

    print("config:", json.dumps(dict(name=name, protocol=args.protocol, smoke=args.smoke,
                                     N_TEXTS=N_TEXTS, N_CONTEXTS=N_CONTEXTS, N_FEATURES=N_FEATURES,
                                     CTX_PER_TYPE=CTX_PER_TYPE, PANEL=PANEL, SEQ_LEN=SEQ_LEN,
                                     ALPHA=ALPHA, TAU=TAU, TOPK_CROWD=TOPK_CROWD,
                                     FREQ_BAND=[FREQ_LO, FREQ_HI], dtype=dtype_name, seed=SEED)))

    import transformer_lens
    import sae_lens
    import transformers
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from datasets import load_dataset
    import scipy.stats as st

    torch.set_grad_enabled(False)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    assert torch.cuda.is_available() or device == "cpu", "no CUDA device found"
    tick(f"device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}; loading SAEs")

    # ---- SAEs first, so the model can be loaded with the kwargs the SAE release expects ----
    def load_sae(sae_id):
        out = SAE.from_pretrained(release=cfg["sae_release"], sae_id=sae_id, device=device)
        sae = out[0] if isinstance(out, (tuple, list)) else out
        return sae.eval()

    sae_p = load_sae(cfg["primary_sae_id"])
    sae_d = load_sae(cfg["downstream_sae_id"])
    hook_p = sae_field(sae_p, "hook_name")
    hook_d = sae_field(sae_d, "hook_name")
    assert hook_p == cfg["primary_hook"], f"primary SAE hook {hook_p} != config {cfg['primary_hook']}"
    assert hook_d == cfg["downstream_hook"], f"downstream SAE hook {hook_d} != config {cfg['downstream_hook']}"
    prepend_bos = bool(sae_field(sae_p, "prepend_bos", True))
    model_kwargs = dict(sae_field(sae_p, "model_from_pretrained_kwargs", {}) or {})
    sae_meta = {
        "primary": {"hook_name": hook_p, "d_sae": int(sae_p.cfg.d_sae), "d_in": int(sae_p.cfg.d_in),
                    "architecture": str(sae_field(sae_p, "architecture", type(sae_p).__name__)),
                    "prepend_bos": prepend_bos, "model_from_pretrained_kwargs": model_kwargs},
        "downstream": {"hook_name": hook_d, "d_sae": int(sae_d.cfg.d_sae), "d_in": int(sae_d.cfg.d_in),
                       "architecture": str(sae_field(sae_d, "architecture", type(sae_d).__name__))},
    }
    tick(f"SAEs loaded: primary d_sae={sae_p.cfg.d_sae} ({hook_p}), downstream d_sae={sae_d.cfg.d_sae} ({hook_d}); "
         f"model_from_pretrained_kwargs={model_kwargs}; prepend_bos={prepend_bos}", "saes_loaded")

    model = HookedTransformer.from_pretrained(cfg["model_name"], device=device, dtype=dtype, **model_kwargs).eval()
    tick(f"model loaded: {cfg['model_name']} d_model={model.cfg.d_model} n_layers={model.cfg.n_layers} "
         f"norm={model.cfg.normalization_type} dtype={dtype_name}", "model_loaded")
    W_dec = sae_p.W_dec.detach().float()                    # [d_sae, d_model]
    W_enc = sae_p.W_enc.detach().float()                    # [d_model, d_sae]
    d_sae, d_model = W_dec.shape
    assert d_model == model.cfg.d_model

    # ---- contexts ----
    tick("streaming wikitext-103 train split")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    seen = set()
    n_dupes = 0
    for row in ds:
        t = row["text"].strip()
        if len(t) <= MIN_CHARS:
            continue
        if args.protocol == "v2":
            key = re.sub(r"\s+", " ", t)
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)
        texts.append(t)
        if len(texts) >= N_TEXTS:
            break
    tick(f"collected {len(texts)} texts (dupes removed: {n_dupes}); tokenising and caching clean activations")

    def encode_final(sae, resid):                            # [B, d_model] -> [B, d_sae]
        return sae.encode(resid.to(device).float())

    tok_rows = []
    n_short_dropped = 0
    pad_id = model.tokenizer.pad_token_id
    if args.protocol == "v1":
        batches = []
        made = 0
        for i in range(0, len(texts), 32):
            toks = model.to_tokens(texts[i:i + 32], prepend_bos=prepend_bos)
            if toks.shape[1] < SEQ_LEN:
                continue
            batches.append(toks[:, :SEQ_LEN].cpu())
            made += toks.shape[0]
            if made >= N_CONTEXTS:
                break
        all_toks = torch.cat(batches)[:N_CONTEXTS]
    else:
        for t in texts:
            toks = model.to_tokens(t, prepend_bos=prepend_bos)[0]
            if toks.shape[0] < SEQ_LEN:
                n_short_dropped += 1
                continue
            tok_rows.append(toks[:SEQ_LEN].cpu())
            if len(tok_rows) >= N_CONTEXTS:
                break
        all_toks = torch.stack(tok_rows)
    N = all_toks.shape[0]
    assert N == N_CONTEXTS, f"only {N} contexts built, wanted {N_CONTEXTS}: raise n_texts"
    n_final_pad = int((all_toks[:, -1] == pad_id).sum().item()) if pad_id is not None else -1
    tick(f"{N} contexts x {SEQ_LEN} tokens; short texts dropped (v2): {n_short_dropped}; "
         f"contexts whose final token is the pad token: {n_final_pad}")

    prim_clean, down_clean = [], []
    for i in range(0, N, MODEL_BATCH):
        toks = all_toks[i:i + MODEL_BATCH].to(device)
        _, cache = model.run_with_cache(toks, names_filter=[hook_p, hook_d])
        prim_clean.append(encode_final(sae_p, cache[hook_p][:, -1, :]).cpu())
        down_clean.append(encode_final(sae_d, cache[hook_d][:, -1, :]).cpu())
        del cache
    prim_clean = torch.cat(prim_clean)                       # [N, d_sae]  primary-site feature acts
    down_clean = torch.cat(down_clean)                       # [N, d_sae_down]
    tick(f"cached clean final-token activations: prim {tuple(prim_clean.shape)} down {tuple(down_clean.shape)}",
         "cache_done")

    # ---- Phase 1: intervention-free predictors ----
    freq = (prim_clean > EPS_FIRE).float().mean(0).numpy()          # final-token firing frequency
    act_mag = prim_clean.mean(0).numpy()                            # mean final-token activation

    Wn = torch.nn.functional.normalize(W_dec, dim=-1)
    crowd = np.empty(d_sae, dtype=np.float32)
    crowd_max = np.empty(d_sae, dtype=np.float32)
    for chunk in torch.split(torch.arange(d_sae), 2048):
        sims = (Wn[chunk] @ Wn.T).abs()
        sims[torch.arange(len(chunk)), chunk] = 0.0                  # exclude self
        top = torch.topk(sims, TOPK_CROWD, dim=-1).values
        crowd[chunk.numpy()] = top.mean(-1).cpu().numpy()
        crowd_max[chunk.numpy()] = top[:, 0].cpu().numpy()
        del sims
    tick("predictors: crowding done")

    rng = np.random.default_rng(SEED)                                 # same call order as the notebook
    elig = np.where((freq >= FREQ_LO) & (freq <= FREQ_HI))[0]
    feats = rng.choice(elig, size=min(N_FEATURES, len(elig)), replace=False)
    feats.sort()
    print(f"eligible features: {len(elig)} of {d_sae} | sampled: {len(feats)}")

    down_freq = (down_clean > EPS_FIRE).float().mean(0)
    panel = torch.topk(down_freq, min(PANEL, down_clean.shape[1])).indices.to(device)

    F = torch.as_tensor(feats)
    A = prim_clean[:, F].float()                                      # [N, 300] sampled feature acts
    fires = (A > EPS_FIRE).float()
    n_fire = fires.sum(0).clamp(min=1)
    act_mean_firing = ((A * fires).sum(0) / n_fire).numpy()
    act_std = A.std(0).numpy()
    act_max = A.max(0).values.numpy()
    act_kurt = st.kurtosis(A.numpy(), axis=0, fisher=True, bias=True)
    p_f = freq[feats]
    eps = 1e-12
    bin_entropy = -(p_f * np.log(p_f + eps) + (1 - p_f) * np.log(1 - p_f + eps))
    r = A / (A.sum(0, keepdim=True) + eps)
    act_entropy = (-(r * torch.log(r + eps)).sum(0) / np.log(N)).numpy()

    B = (prim_clean > EPS_FIRE).float().to(device)                    # [N, d_sae]
    Bf = B[:, F.to(device)]                                            # [N, 300]
    co = (Bf.T @ B) / (Bf.sum(0, keepdim=True).T + eps)                # q_{f,j}  [300, d_sae]
    co[torch.arange(len(feats)), F.to(device)] = 0.0
    pi = co / (co.sum(1, keepdim=True) + eps)
    coact_entropy = (-(pi * torch.log(pi + eps)).sum(1)).cpu().numpy()
    coact_count = ((Bf.T @ B.sum(1, keepdim=True)).squeeze(1) / (Bf.sum(0) + eps) - 1).cpu().numpy()
    del B, Bf, co, pi

    W_U = model.W_U.detach().float()                                  # [d_model, vocab]
    r_f = W_dec[F.to(device)] @ W_U                                    # direct-logit vectors [300, vocab]
    logit_l2 = r_f.norm(dim=-1).cpu().numpy()
    logit_linf = r_f.abs().max(-1).values.cpu().numpy()
    s = r_f.abs() / (r_f.abs().sum(-1, keepdim=True) + eps)
    logit_entropy = (-(s * torch.log(s + eps)).sum(-1)).cpu().numpy()
    logit_top10_mass = torch.topk(s, 10, dim=-1).values.sum(-1).cpu().numpy()
    del r_f, s

    dec_norm = W_dec[F.to(device)].norm(dim=-1).cpu().numpy()
    enc_vec = W_enc[:, F.to(device)].T                                 # [300, d_model]
    enc_norm = enc_vec.norm(dim=-1).cpu().numpy()
    enc_dec_cos = torch.nn.functional.cosine_similarity(enc_vec, W_dec[F.to(device)], dim=-1).cpu().numpy()
    tick("predictors: all done", "predictors_done")

    # ---- Phase 2: steering labels on the mixed context set ----
    def pick_contexts(fi):
        a = prim_clean[:, fi].numpy()
        order = np.argsort(-a)
        top = order[:CTX_PER_TYPE]
        low = order[-CTX_PER_TYPE:]
        mid = rng.choice(np.setdiff1d(order, np.concatenate([top, low])), size=CTX_PER_TYPE, replace=False)
        return np.concatenate([top, mid, low])

    def steered_forward(toks, d_f):
        store = {}

        def steer(resid, hook):
            resid[:, -1, :] = resid[:, -1, :] + ALPHA * d_f.to(resid.dtype)
            return resid

        def grab(resid, hook):
            store["d"] = resid[:, -1, :].detach()
            return resid

        logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=[(hook_p, steer), (hook_d, grab)])
        return logits[:, -1, :].detach().float(), store["d"]

    rows = []
    n_ctx = 3 * CTX_PER_TYPE
    tick(f"steering {len(feats)} features x {n_ctx} contexts")
    for n, fi in enumerate(feats):
        ctx = pick_contexts(fi)
        toks = all_toks[ctx].to(device)
        d_f = W_dec[fi]
        logit_c, cache = model.run_with_cache(toks, names_filter=[hook_d], return_type="logits")
        logit_c = logit_c[:, -1, :].float()
        u_clean = encode_final(sae_d, cache[hook_d][:, -1, :])
        del cache
        logit_s, down_resid = steered_forward(toks, d_f)
        u_steer = encode_final(sae_d, down_resid)
        du = (u_steer - u_clean)[:, panel].abs()                      # [n_ctx, PANEL]
        coll = (du > TAU).float().sum(-1).mean().item()               # C_{f,tau}
        dl = logit_s - logit_c                                         # [n_ctx, vocab]
        dl_norm = dl.norm(dim=-1)
        Ef = dl_norm.mean().item()                                     # E_f
        dl_mean = dl.mean(0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(dl, dl_mean.expand_as(dl), dim=-1)
        logp = torch.log_softmax(logit_c, -1)
        logq = torch.log_softmax(logit_s, -1)
        kl = (logp.exp() * (logp - logq)).sum(-1).mean().item()
        rows.append({
            "feature": int(fi),
            "crowding": float(crowd[fi]), "crowd_max": float(crowd_max[fi]),
            "dec_norm": float(dec_norm[n]), "enc_norm": float(enc_norm[n]), "enc_dec_cos": float(enc_dec_cos[n]),
            "frequency": float(freq[fi]), "act_mag": float(act_mag[fi]),
            "act_mean_firing": float(act_mean_firing[n]), "act_std": float(act_std[n]), "act_max": float(act_max[n]),
            "act_kurtosis": float(act_kurt[n]), "bin_entropy": float(bin_entropy[n]), "act_entropy": float(act_entropy[n]),
            "coact_entropy": float(coact_entropy[n]), "coact_count": float(coact_count[n]),
            "logit_l2": float(logit_l2[n]), "logit_linf": float(logit_linf[n]),
            "logit_entropy": float(logit_entropy[n]), "logit_top10_mass": float(logit_top10_mass[n]),
            "collateral_raw": coll, "effect_l2": Ef, "collateral_ctilde": coll / (Ef + 1e-8),
            "stab_signed": cos.mean().item(), "stab_abs": cos.abs().mean().item(),
            "kl_mean": kl, "kl_per_effect": kl / (Ef + 1e-8),
            "effect_cv": (dl_norm.std() / (dl_norm.mean() + 1e-8)).item(),
            "ctx_mean_act": float(prim_clean[ctx, fi].mean().item()),
            "intervention_value": ALPHA, "n_ctx": int(n_ctx),
        })
        if (n + 1) % max(1, len(feats) // 6) == 0:
            tick(f"  steered {n + 1}/{len(feats)}")
    tick("steering done", "steering_done")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_feature.csv"), index=False)
    json.dump({"features": [int(x) for x in feats], "panel": [int(x) for x in panel.cpu().numpy()]},
              open(os.path.join(out_dir, "selection.json"), "w"))

    # ---- quick headline, same statistics as the notebook, for the regression gate ----
    from scipy.stats import rankdata, pearsonr

    def sp(x, y):
        r_, p_ = st.spearmanr(df[x], df[y])
        return round(float(r_), 4), float(p_)

    def partial(x, y, zs):
        def resid(a, Z):
            Z1 = np.c_[np.ones(len(a)), Z]
            coef, *_ = np.linalg.lstsq(Z1, a, rcond=None)
            return a - Z1 @ coef
        Z = np.c_[[rankdata(df[z]) for z in zs]].T
        r_, p_ = pearsonr(resid(rankdata(df[x]), Z), resid(rankdata(df[y]), Z))
        return round(float(r_), 4), float(p_)

    head = {}
    for tgt in ["collateral_raw", "collateral_ctilde"]:
        for pred in ["crowding", "frequency", "act_mag"]:
            head[f"rho_{pred}__{tgt}"] = sp(pred, tgt)[0]
        head[f"partial_crowding__{tgt}__given_freq_actmag"] = partial("crowding", tgt, ["frequency", "act_mag"])[0]
    med = df["frequency"].median()
    lo, hi = df[df.frequency <= med], df[df.frequency > med]
    head["crowd_rho_lowfreq_half"] = round(float(st.spearmanr(lo.crowding, lo.collateral_raw)[0]), 4)
    head["crowd_rho_highfreq_half"] = round(float(st.spearmanr(hi.crowding, hi.collateral_raw)[0]), 4)
    print(json.dumps(head, indent=1))

    meta = {
        "setting": name, "protocol": args.protocol, "smoke": args.smoke, "seed": SEED, "config": cfg,
        "sizes": dict(n_texts=len(texts), n_contexts=N, seq_len=SEQ_LEN, n_features=len(feats), n_eligible=int(len(elig)),
                      ctx_per_type=CTX_PER_TYPE, n_ctx_per_feature=n_ctx, panel=int(panel.numel()),
                      d_sae_primary=int(d_sae), d_sae_downstream=int(down_clean.shape[1])),
        "context_build": dict(dupes_removed=n_dupes, short_texts_dropped=n_short_dropped,
                              contexts_final_token_is_pad=n_final_pad, pad_token_id=pad_id),
        "sae": sae_meta, "dtype": dtype_name, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "versions": dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                         transformer_lens=_pkg_version("transformer-lens", transformer_lens),
                         sae_lens=_pkg_version("sae-lens", sae_lens), numpy=np.__version__),
        "timings_s": TIMINGS, "wall_clock_s": round(time.time() - T0, 1), "headline": head,
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    tick(f"wrote {out_dir}/per_feature.csv, selection.json, meta.json")


if __name__ == "__main__":
    main()
