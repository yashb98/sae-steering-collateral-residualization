#!/usr/bin/env python
"""Predictors and steering labels for one model/SAE setting of arXiv 2606.08365.

Computes the intervention-free predictors for 300 sampled SAE features, steers each one
additively at the final token (alpha = 1.0) and measures collateral, effect magnitude,
stability and KL shift. Writes results/<name>/per_feature.csv, selection.json, meta.json.

Protocols: v1 rebuilds the contexts exactly as the original GPT-2-small notebook did (used
only as a regression gate); v2 keeps texts with at least seq_len real tokens and deduplicates
them, and is used for every reported setting.
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


def residual_change_metrics(dh, drec):
    error = dh - drec
    energy = dh.square().sum()
    explained = 1 - error.square().sum() / energy if energy > 0 else dh.new_tensor(float("nan"))
    return {
        "resid_delta_norm": dh.norm(dim=-1).mean().item(),
        "resid_reconstruction_norm_ratio": (drec.norm(dim=-1) / (dh.norm(dim=-1) + 1e-8)).mean().item(),
        "resid_error_delta_norm": error.norm(dim=-1).mean().item(),
        "resid_change_explained_energy": explained.item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dir (default results/<name>)")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes to validate the pipeline")
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v2")
    ap.add_argument("--dtype", default=None, help="override model dtype: float32|bfloat16|float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None, help="override the config's steering coefficient")
    ap.add_argument("--alpha-mode", choices=["fixed", "q95"], default="fixed",
                    help="fixed: add alpha * d_f; q95: add alpha * q95(natural activation of f) * d_f")
    ap.add_argument("--context-split", choices=["none", "A", "B"], default="none",
                    help="build each feature's context set from even (A) or odd (B) context indices only")
    ap.add_argument("--random-directions", action="store_true",
                    help="steer with random directions (norms matched to the decoder) instead of SAE features")
    ap.add_argument("--trace-steering", action="store_true", help="log intervention stages and save temporary checkpoints every 50 features")
    ap.add_argument("--paired-random-control", action="store_true",
                    help="also measure random unit directions scaled to each sampled decoder norm on identical contexts")
    ap.add_argument("--save-residual-deltas", action="store_true",
                    help="save per-context downstream and reconstructed changes in residual_deltas.npz")
    ap.add_argument("--dense-freq", type=float, default=0.10,
                    help="downstream panel features firing more often than this are excluded from the *_nodense labels")
    args = ap.parse_args()

    if args.paired_random_control and args.random_directions:
        ap.error("--paired-random-control cannot be combined with --random-directions")
    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    ALPHA = float(args.alpha) if args.alpha is not None else float(cfg["alpha"])
    suffix = ("_smoke" if args.smoke else "") + ("_v1" if args.protocol == "v1" else "")
    if args.alpha is not None and args.alpha != float(cfg["alpha"]):
        suffix += f"_alpha{args.alpha:g}"
    if args.alpha_mode == "q95":
        suffix += "_q95"
    if args.context_split != "none":
        suffix += f"_ctx{args.context_split}"
    if args.random_directions:
        suffix += "_random"
    out_dir = args.out or os.path.join("results", name + suffix)
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        N_TEXTS, N_CONTEXTS, N_FEATURES, CTX_PER_TYPE, PANEL = 600, 128, 12, 4, 256
    else:
        N_TEXTS, N_CONTEXTS, N_FEATURES = cfg["n_texts"], cfg["n_contexts"], cfg["n_features"]
        CTX_PER_TYPE, PANEL = cfg["ctx_per_type"], cfg["panel_size"]
    SEQ_LEN = cfg["seq_len"]
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
                                     ALPHA=ALPHA, alpha_mode=args.alpha_mode, context_split=args.context_split,
                                     random_directions=args.random_directions, TAU=TAU, TOPK_CROWD=TOPK_CROWD,
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

    # ---- SAEs first: the model is loaded with the kwargs the SAE release expects ----
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

    # transformers 5 renamed NeoX's embed_out to lm_head; TransformerLens still reads embed_out.
    hf_model = None
    if "pythia" in cfg["model_name"].lower():
        from transformers import AutoModelForCausalLM
        hf_name = cfg.get("hf_model_name", "EleutherAI/" + cfg["model_name"])
        hf_model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=dtype)
        if not hasattr(hf_model, "embed_out") and hasattr(hf_model, "lm_head"):
            hf_model.embed_out = hf_model.lm_head
    elif "llama" in cfg["model_name"].lower():
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype, low_cpu_mem_usage=True)
    # tl_processing: "default" applies TransformerLens weight processing; "none" skips it, which
    # roughly halves peak memory for large models and leaves the residual stream unchanged.
    tl_processing = cfg.get("tl_processing", "default")
    loader = HookedTransformer.from_pretrained_no_processing if tl_processing == "none" else HookedTransformer.from_pretrained
    model = loader(cfg["model_name"], hf_model=hf_model, device=device, dtype=dtype, **model_kwargs).eval()
    del hf_model
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
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
    l0_p = (prim_clean > EPS_FIRE).float().sum(1).mean().item()
    l0_d = (down_clean > EPS_FIRE).float().sum(1).mean().item()
    dead_p = ((prim_clean > EPS_FIRE).float().sum(0) == 0).float().mean().item()
    tick(f"cached clean final-token activations: prim {tuple(prim_clean.shape)} down {tuple(down_clean.shape)}; "
         f"mean L0 primary={l0_p:.1f} downstream={l0_d:.1f}; primary features never firing={dead_p:.1%}", "cache_done")

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
    panel_nodense = (down_freq[panel.cpu()] <= args.dense_freq).to(device)      # mask over the panel
    print(f"panel {panel.numel()} features; {int(panel_nodense.sum())} with firing frequency <= {args.dense_freq}")

    # steering vectors: the sampled features' decoder rows, or random directions with matched norms
    if args.random_directions:
        rng_dir = np.random.default_rng(SEED + 1000)
        g = torch.as_tensor(rng_dir.standard_normal((len(feats), d_model)), dtype=torch.float32)
        norms = torch.as_tensor(rng_dir.choice(W_dec.norm(dim=-1).cpu().numpy(), size=len(feats)), dtype=torch.float32)
        steer_vecs = (torch.nn.functional.normalize(g, dim=-1) * norms[:, None]).to(device)
        sims_r = (torch.nn.functional.normalize(steer_vecs, dim=-1) @ Wn.T).abs()
        top_r = torch.topk(sims_r, TOPK_CROWD, dim=-1).values
        crowd_vec, crowdmax_vec = top_r.mean(-1).cpu().numpy(), top_r[:, 0].cpu().numpy()
        del sims_r
    else:
        steer_vecs = W_dec[torch.as_tensor(feats).to(device)]
        crowd_vec, crowdmax_vec = crowd[feats], crowd_max[feats]

    paired_vecs = None
    if args.paired_random_control:
        rng_pair = np.random.default_rng(SEED + 2000)
        g_pair = torch.as_tensor(rng_pair.standard_normal((len(feats), d_model)), dtype=torch.float32, device=device)
        paired_vecs = torch.nn.functional.normalize(g_pair, dim=-1) * steer_vecs.norm(dim=-1, keepdim=True)
        pair_sim = (torch.nn.functional.normalize(paired_vecs, dim=-1) @ Wn.T).abs()
        pair_top = torch.topk(pair_sim, TOPK_CROWD, dim=-1).values
        pair_crowd = pair_top.mean(-1).cpu().numpy()
        pair_max = pair_top[:, 0].cpu().numpy()
        pair_logits = paired_vecs @ model.W_U.detach().float()
        pair_logit_l2 = pair_logits.norm(dim=-1).cpu().numpy()
        pair_logit_linf = pair_logits.abs().max(-1).values.cpu().numpy()
        pair_mass = pair_logits.abs() / (pair_logits.abs().sum(-1, keepdim=True) + 1e-12)
        pair_logit_entropy = (-(pair_mass * torch.log(pair_mass + 1e-12)).sum(-1)).cpu().numpy()
        pair_top10 = torch.topk(pair_mass, 10, dim=-1).values.sum(-1).cpu().numpy()
        del pair_sim, pair_top, pair_logits, pair_mass

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
    r_f = steer_vecs @ W_U                                             # direct-logit vectors [300, vocab]
    logit_l2 = r_f.norm(dim=-1).cpu().numpy()
    logit_linf = r_f.abs().max(-1).values.cpu().numpy()
    s = r_f.abs() / (r_f.abs().sum(-1, keepdim=True) + eps)
    logit_entropy = (-(s * torch.log(s + eps)).sum(-1)).cpu().numpy()
    logit_top10_mass = torch.topk(s, 10, dim=-1).values.sum(-1).cpu().numpy()
    del r_f, s

    dec_norm = steer_vecs.norm(dim=-1).cpu().numpy()
    enc_vec = W_enc[:, F.to(device)].T                                 # [300, d_model]
    enc_norm = enc_vec.norm(dim=-1).cpu().numpy()
    enc_dec_cos = torch.nn.functional.cosine_similarity(enc_vec, W_dec[F.to(device)], dim=-1).cpu().numpy()
    tick("predictors: all done", "predictors_done")

    # ---- Phase 2: steering labels on the mixed context set ----
    pool = np.arange(N)
    if args.context_split == "A":
        pool = pool[pool % 2 == 0]
    elif args.context_split == "B":
        pool = pool[pool % 2 == 1]

    def pick_contexts(fi):
        if args.random_directions:
            return rng.choice(pool, size=3 * CTX_PER_TYPE, replace=False)
        a = prim_clean[pool, fi].numpy()
        order = np.argsort(-a)
        top = order[:CTX_PER_TYPE]
        low = order[-CTX_PER_TYPE:]
        mid = rng.choice(np.setdiff1d(order, np.concatenate([top, low])), size=CTX_PER_TYPE, replace=False)
        return pool[np.concatenate([top, mid, low])]

    def feature_alpha(fi):
        if args.alpha_mode == "fixed" or args.random_directions:
            return ALPHA
        a = prim_clean[:, fi].numpy()
        firing = a[a > EPS_FIRE]
        return float(ALPHA * (np.quantile(firing, 0.95) if len(firing) >= 2 else a.max()))

    def steered_forward(toks, d_f, alpha_f):
        store = {}

        def steer(resid, hook):
            resid[:, -1, :] = resid[:, -1, :] + alpha_f * d_f.to(resid.dtype)
            return resid

        def grab(resid, hook):
            store["d"] = resid[:, -1, :].detach()
            return resid

        logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=[(hook_p, steer), (hook_d, grab)])
        return logits[:, -1, :].detach().float(), store["d"]

    rows, paired_rows, context_indices = [], [], []
    residual_deltas, reconstructed_deltas = [], []
    n_ctx = 3 * CTX_PER_TYPE
    tick(f"steering {len(feats)} features x {n_ctx} contexts")
    NA = float("nan")
    for n, fi in enumerate(feats):
        ctx = pick_contexts(fi)
        context_indices.append(ctx.tolist())
        toks = all_toks[ctx].to(device)
        d_f = steer_vecs[n]
        alpha_f = feature_alpha(fi)
        if args.trace_steering:
            tick(f"feature {n+1}: clean forward")
        logit_c, cache = model.run_with_cache(toks, names_filter=[hook_d], return_type="logits")
        logit_c = logit_c[:, -1, :].float()
        h_clean = cache[hook_d][:, -1, :].detach().float()
        u_clean = encode_final(sae_d, h_clean)
        del cache
        if args.trace_steering:
            tick(f"feature {n+1}: SAE forward")
        logit_s, down_resid = steered_forward(toks, d_f, alpha_f)
        down_resid = down_resid.float()
        u_steer = encode_final(sae_d, down_resid)
        du = (u_steer - u_clean)[:, panel].abs()                      # [n_ctx, PANEL]
        coll = (du > TAU).float().sum(-1).mean().item()               # C_{f,tau}
        coll_nd = (du[:, panel_nodense] > TAU).float().sum(-1).mean().item()
        dh = down_resid - h_clean                                      # downstream residual change
        dh_norm = dh.norm(dim=-1)
        drec = sae_d.decode(u_steer) - sae_d.decode(u_clean)          # the part of dh the downstream SAE sees
        residual_labels = residual_change_metrics(dh, drec)
        if args.save_residual_deltas:
            residual_deltas.append(dh.cpu().numpy())
            reconstructed_deltas.append(drec.cpu().numpy())
        dl = logit_s - logit_c                                         # [n_ctx, vocab]
        dl_norm = dl.norm(dim=-1)
        Ef = dl_norm.mean().item()                                     # E_f
        dl_mean = dl.mean(0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(dl, dl_mean.expand_as(dl), dim=-1)
        dlc = dl - dl_mean
        if args.trace_steering:
            tick(f"feature {n+1}: SAE eigenvalues")
        ev = torch.linalg.eigvalsh(dlc @ dlc.T).clamp(min=0)
        pc1 = (ev.max() / (ev.sum() + 1e-12)).item()
        logp = torch.log_softmax(logit_c, -1)
        logq = torch.log_softmax(logit_s, -1)
        kl = (logp.exp() * (logp - logq)).sum(-1).mean().item()
        rnd = args.random_directions
        rows.append({
            "feature": int(fi) if not rnd else int(n),
            "crowding": float(crowd_vec[n]), "crowd_max": float(crowdmax_vec[n]),
            "dec_norm": float(dec_norm[n]), "enc_norm": NA if rnd else float(enc_norm[n]),
            "enc_dec_cos": NA if rnd else float(enc_dec_cos[n]),
            "frequency": NA if rnd else float(freq[fi]), "act_mag": NA if rnd else float(act_mag[fi]),
            "act_mean_firing": NA if rnd else float(act_mean_firing[n]), "act_std": NA if rnd else float(act_std[n]),
            "act_max": NA if rnd else float(act_max[n]), "act_kurtosis": NA if rnd else float(act_kurt[n]),
            "bin_entropy": NA if rnd else float(bin_entropy[n]), "act_entropy": NA if rnd else float(act_entropy[n]),
            "coact_entropy": NA if rnd else float(coact_entropy[n]), "coact_count": NA if rnd else float(coact_count[n]),
            "logit_l2": float(logit_l2[n]), "logit_linf": float(logit_linf[n]),
            "logit_entropy": float(logit_entropy[n]), "logit_top10_mass": float(logit_top10_mass[n]),
            "collateral_raw": coll, "effect_l2": Ef, "collateral_ctilde": coll / (Ef + 1e-8),
            "collateral_raw_nodense": coll_nd, "collateral_ctilde_nodense": coll_nd / (Ef + 1e-8),
            **residual_labels,
            "stab_signed": cos.mean().item(), "stab_abs": cos.abs().mean().item(),
            "stab_anti_frac": (cos < 0).float().mean().item(), "effect_pc1_ratio": pc1,
            "kl_mean": kl, "kl_per_effect": kl / (Ef + 1e-8),
            "effect_cv": (dl_norm.std() / (dl_norm.mean() + 1e-8)).item(),
            "ctx_mean_act": NA if rnd else float(prim_clean[ctx, fi].mean().item()),
            "intervention_value": alpha_f, "n_ctx": int(n_ctx),
        })
        if paired_vecs is not None:
            if args.trace_steering:
                tick(f"feature {n+1}: random forward")
            pair_ls, pair_h = steered_forward(toks, paired_vecs[n], alpha_f)
            pair_h = pair_h.float()
            pair_u = encode_final(sae_d, pair_h)
            pair_du = (pair_u - u_clean)[:, panel].abs()
            pair_coll = (pair_du > TAU).float().sum(-1).mean().item()
            pair_nd = (pair_du[:, panel_nodense] > TAU).float().sum(-1).mean().item()
            pair_dl = pair_ls - logit_c
            pair_norm = pair_dl.norm(dim=-1)
            pair_E = pair_norm.mean().item()
            pair_mean = pair_dl.mean(0, keepdim=True)
            pair_cos = torch.nn.functional.cosine_similarity(pair_dl, pair_mean.expand_as(pair_dl), dim=-1)
            pair_centered = pair_dl - pair_mean
            if args.trace_steering:
                tick(f"feature {n+1}: random eigenvalues (CPU)")
            pair_ev = torch.linalg.eigvalsh((pair_centered @ pair_centered.T).cpu()).clamp(min=0)
            pair_kl = (logp.exp() * (logp - torch.log_softmax(pair_ls, -1))).sum(-1).mean().item()
            pair_row = {k: NA for k in rows[-1]}
            pair_row.update({
                "feature": int(n), "matched_feature": int(fi), "crowding": float(pair_crowd[n]),
                "crowd_max": float(pair_max[n]), "dec_norm": float(dec_norm[n]),
                "logit_l2": float(pair_logit_l2[n]), "logit_linf": float(pair_logit_linf[n]),
                "logit_entropy": float(pair_logit_entropy[n]), "logit_top10_mass": float(pair_top10[n]),
                "collateral_raw": pair_coll, "effect_l2": pair_E, "collateral_ctilde": pair_coll / (pair_E + 1e-8),
                "collateral_raw_nodense": pair_nd, "collateral_ctilde_nodense": pair_nd / (pair_E + 1e-8),
                **residual_change_metrics(pair_h - h_clean, sae_d.decode(pair_u) - sae_d.decode(u_clean)),
                "stab_signed": pair_cos.mean().item(), "stab_abs": pair_cos.abs().mean().item(),
                "stab_anti_frac": (pair_cos < 0).float().mean().item(),
                "effect_pc1_ratio": (pair_ev.max() / (pair_ev.sum() + 1e-12)).item(),
                "kl_mean": pair_kl, "kl_per_effect": pair_kl / (pair_E + 1e-8),
                "effect_cv": (pair_norm.std() / (pair_norm.mean() + 1e-8)).item(),
                "intervention_value": alpha_f, "n_ctx": n_ctx,
            })
            paired_rows.append(pair_row)
        if args.trace_steering:
            tick(f"feature {n+1}: complete")
            if (n + 1) % 50 == 0:
                import pandas as pd
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_feature.partial.csv"), index=False)
                pd.DataFrame(paired_rows).to_csv(os.path.join(out_dir, "paired.partial.csv"), index=False)
                if args.save_residual_deltas:
                    np.savez_compressed(os.path.join(out_dir, "residual_deltas.partial.npz"),
                                        delta=np.stack(residual_deltas), reconstruction_delta=np.stack(reconstructed_deltas),
                                        features=feats[:n+1], contexts=np.asarray(context_indices))
        if (n + 1) % max(1, len(feats) // 6) == 0:
            tick(f"  steered {n + 1}/{len(feats)}")
    tick("steering done", "steering_done")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_feature.csv"), index=False)
    if args.save_residual_deltas:
        np.savez_compressed(os.path.join(out_dir, "residual_deltas.npz"),
                            delta=np.stack(residual_deltas), reconstruction_delta=np.stack(reconstructed_deltas),
                            features=feats, contexts=np.asarray(context_indices))
    json.dump({"features": [int(x) for x in feats], "contexts": context_indices,
               "panel": [int(x) for x in panel.cpu().numpy()]},
              open(os.path.join(out_dir, "selection.json"), "w"))

    # ---- headline statistics (regression gate) ----
    from scipy.stats import rankdata, pearsonr

    def sp(x, y):
        d_ = df[[x, y]].dropna()
        if len(d_) < 3:
            return float("nan"), float("nan")
        r_, p_ = st.spearmanr(d_[x], d_[y])
        return round(float(r_), 4), float(p_)

    def partial(x, y, zs):
        def resid(a, Z):
            Z1 = np.c_[np.ones(len(a)), Z]
            coef, *_ = np.linalg.lstsq(Z1, a, rcond=None)
            return a - Z1 @ coef
        d_ = df[[x, y] + zs].dropna()
        if len(d_) < 3:
            return float("nan"), float("nan")
        Z = np.c_[[rankdata(d_[z]) for z in zs]].T
        r_, p_ = pearsonr(resid(rankdata(d_[x]), Z), resid(rankdata(d_[y]), Z))
        return round(float(r_), 4), float(p_)

    head = {}
    for tgt in ["collateral_raw", "collateral_ctilde"]:
        for pred in ["crowding", "frequency", "act_mag"]:
            head[f"rho_{pred}__{tgt}"] = sp(pred, tgt)[0]
        head[f"partial_crowding__{tgt}__given_freq_actmag"] = partial("crowding", tgt, ["frequency", "act_mag"])[0]
    if df["frequency"].notna().all():
        med = df["frequency"].median()
        lo, hi = df[df.frequency <= med], df[df.frequency > med]
        head["crowd_rho_lowfreq_half"] = round(float(st.spearmanr(lo.crowding, lo.collateral_raw)[0]), 4)
        head["crowd_rho_highfreq_half"] = round(float(st.spearmanr(hi.crowding, hi.collateral_raw)[0]), 4)
    head["rho_crowding__collateral_raw_nodense"] = sp("crowding", "collateral_raw_nodense")[0]
    head["rho_crowding__resid_delta_norm"] = sp("crowding", "resid_delta_norm")[0]
    head["mean_resid_reconstruction_norm_ratio"] = round(float(df["resid_reconstruction_norm_ratio"].mean()), 4)
    head["mean_resid_change_explained_energy"] = round(float(df["resid_change_explained_energy"].mean()), 4)
    print(json.dumps(head, indent=1))

    meta = {
        "setting": name, "protocol": args.protocol, "smoke": args.smoke, "seed": SEED, "config": cfg,
        "variant": dict(alpha=ALPHA, alpha_mode=args.alpha_mode, context_split=args.context_split,
                        random_directions=args.random_directions, dense_freq=args.dense_freq,
                        n_panel_nodense=int(panel_nodense.sum()), paired_random_control=args.paired_random_control,
                        residual_metric_version=2, saved_residual_deltas=args.save_residual_deltas),
        "sizes": dict(n_texts=len(texts), n_contexts=N, seq_len=SEQ_LEN, n_features=len(feats), n_eligible=int(len(elig)),
                      ctx_per_type=CTX_PER_TYPE, n_ctx_per_feature=n_ctx, panel=int(panel.numel()),
                      d_sae_primary=int(d_sae), d_sae_downstream=int(down_clean.shape[1]),
                      mean_l0_primary=round(l0_p, 2), mean_l0_downstream=round(l0_d, 2), frac_primary_never_firing=round(dead_p, 4)),
        "context_build": dict(dupes_removed=n_dupes, short_texts_dropped=n_short_dropped,
                              contexts_final_token_is_pad=n_final_pad, pad_token_id=pad_id),
        "sae": sae_meta, "dtype": dtype_name, "tl_processing": tl_processing, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "versions": dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                         transformer_lens=_pkg_version("transformer-lens", transformer_lens),
                         sae_lens=_pkg_version("sae-lens", sae_lens), numpy=np.__version__),
        "timings_s": TIMINGS, "wall_clock_s": round(time.time() - T0, 1), "headline": head,
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    if paired_rows:
        pair_out = out_dir + "_random_paired"
        os.makedirs(pair_out, exist_ok=True)
        pd.DataFrame(paired_rows).to_csv(os.path.join(pair_out, "per_feature.csv"), index=False)
        pair_meta = dict(meta, headline={})
        pair_meta["variant"] = dict(meta["variant"], random_directions=True,
                                    random_contexts="matched_to_sampled_feature", random_seed=SEED + 2000,
                                    norms="matched_per_sampled_feature", saved_residual_deltas=False, pc1_eigensolver_device="cpu")
        json.dump(pair_meta, open(os.path.join(pair_out, "meta.json"), "w"), indent=1)
        json.dump({"features": list(range(len(feats))), "matched_features": feats.tolist(),
                   "contexts": context_indices, "panel": panel.cpu().tolist()},
                  open(os.path.join(pair_out, "selection.json"), "w"))
        np.savez_compressed(os.path.join(pair_out, "directions.npz"), directions=paired_vecs.cpu().numpy())
    for temporary in ["per_feature.partial.csv", "paired.partial.csv", "residual_deltas.partial.npz"]:
        path = os.path.join(out_dir, temporary)
        if os.path.exists(path):
            os.remove(path)
    tick(f"wrote {out_dir}/per_feature.csv, selection.json, meta.json")


if __name__ == "__main__":
    main()
