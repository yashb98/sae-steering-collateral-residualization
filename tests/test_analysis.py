import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from residualize import ALL_PREDICTORS, CONTROLS, partial_spearman, residualized_cv  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def make_df(seed=0, n=300):
    rng = np.random.default_rng(seed)
    d = {c: rng.normal(size=n) for c in ALL_PREDICTORS}
    d["frequency"] = rng.uniform(0.002, 0.5, n)
    d["act_mag"] = np.abs(rng.normal(size=n))
    d["effect_l2"] = np.abs(rng.normal(size=n)) + 0.1
    d["collateral_raw"] = np.maximum(0, 5 * d["crowding"] + 30 * d["frequency"] + 2 * d["effect_l2"] + rng.normal(size=n))
    d["collateral_ctilde"] = d["collateral_raw"] / (d["effect_l2"] + 1e-8)
    d["intervention_value"] = 1.0
    return pd.DataFrame(d)


def reference_partial(df, x, y, zs):
    def resid(a, Z):
        Z1 = np.c_[np.ones(len(a)), Z]
        coef, *_ = np.linalg.lstsq(Z1, a, rcond=None)
        return a - Z1 @ coef
    Z = np.c_[[rankdata(df[z]) for z in zs]].T
    return pearsonr(resid(rankdata(df[x]), Z), resid(rankdata(df[y]), Z))[0]


def test_partial_matches_reference_implementation():
    df = make_df()
    rng = np.random.default_rng(0)
    for cname in ["robust", "primary"]:
        controls = [c for c in CONTROLS[cname] if df[c].nunique() > 1]
        got = partial_spearman(df, "crowding", "collateral_raw", CONTROLS[cname], 200, rng)
        assert abs(got["rho"] - reference_partial(df, "crowding", "collateral_raw", controls)) < 1e-9
        assert got["ci_lo"] <= got["rho"] <= got["ci_hi"]
        assert got["k_controls"] == len(controls)


def test_no_control_equals_spearman():
    df = make_df(1)
    got = partial_spearman(df, "crowding", "collateral_raw", [], 200, np.random.default_rng(0))
    assert abs(got["rho"] - spearmanr(df.crowding, df.collateral_raw)[0]) < 1e-9


def test_constant_control_is_dropped():
    df = make_df(2)
    got = partial_spearman(df, "crowding", "collateral_raw", ["intervention_value", "frequency"], 200, np.random.default_rng(0))
    assert got["controls"] == "frequency"


def test_bootstrap_is_reproducible():
    df = make_df(3)
    a = partial_spearman(df, "crowding", "collateral_raw", CONTROLS["robust"], 500, np.random.default_rng(7))
    b = partial_spearman(df, "crowding", "collateral_raw", CONTROLS["robust"], 500, np.random.default_rng(7))
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_residualized_cv_runs():
    r = residualized_cv(make_df(4), "collateral_raw", CONTROLS["primary"], ["crowding", "frequency"])
    assert -1 <= r["cv_spearman_mean"] <= 1
    assert r["n_pred"] == 2


def test_result_files_are_complete():
    required = set(ALL_PREDICTORS) | {"feature", "collateral_raw", "effect_l2", "collateral_ctilde",
                                       "stab_signed", "stab_abs", "kl_mean", "intervention_value"}
    paths = [p for p in glob.glob(os.path.join(ROOT, "results", "*", "per_feature.csv")) if "smoke" not in p]
    assert paths, "no result files"
    for p in paths:
        df = pd.read_csv(p)
        assert len(df) == 300, p
        assert required <= set(df.columns), p
        meta = __import__("json").load(open(os.path.join(os.path.dirname(p), "meta.json")))
        finite_required = required.copy()
        if meta.get("variant", {}).get("random_directions"):
            undefined = {"enc_norm", "enc_dec_cos", "frequency", "act_mag", "act_mean_firing", "act_std",
                         "act_max", "act_kurtosis", "bin_entropy", "act_entropy", "coact_entropy", "coact_count"}
            assert df[sorted(undefined)].isna().all().all(), p
            finite_required -= undefined
        assert np.isfinite(df[sorted(finite_required)].to_numpy()).all(), p
        assert df.feature.is_unique, p


def test_holm_and_bh_against_reference():
    from residualize import benjamini_hochberg, holm
    p = np.array([0.01, 0.04, 0.03, 0.20, 0.50])
    # Holm: sort p, multiply by (m - rank), enforce monotonicity, cap at 1
    assert np.allclose(holm(p), [0.05, 0.12, 0.12, 0.40, 0.50])
    # BH: m * p / rank with step-up monotonicity
    assert np.allclose(benjamini_hochberg(p), [0.05, 0.0667, 0.0667, 0.25, 0.50], atol=1e-4)
    assert (holm(p) >= p).all() and (benjamini_hochberg(p) >= p).all()


def test_partial_handles_missing_predictor_values():
    df = make_df(5)
    df.loc[:9, "frequency"] = np.nan
    got = partial_spearman(df, "crowding", "collateral_raw", CONTROLS["robust"], 100, np.random.default_rng(0))
    assert got["n"] == 290


def test_cv_matches_independent_fold_local_reference():
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    d = make_df(17)
    d.loc[:40, "collateral_raw"] += 40 * d.loc[:40, "frequency"]
    columns = ["crowding", "frequency"]
    Z = np.c_[np.ones(len(d)), d[["frequency", "effect_l2"]]]
    y = d.collateral_raw.to_numpy()
    pred, residual, scores = np.zeros(len(d)), np.zeros(len(d)), []
    for train, test in KFold(5, shuffle=True, random_state=0).split(d):
        beta = np.linalg.lstsq(Z[train], y[train], rcond=None)[0]
        train_y, test_y = y[train] - Z[train] @ beta, y[test] - Z[test] @ beta
        sc = StandardScaler().fit(d[columns].iloc[train])
        model = Ridge(alpha=1).fit(sc.transform(d[columns].iloc[train]), train_y)
        pred[test] = model.predict(sc.transform(d[columns].iloc[test]))
        residual[test] = test_y
        scores.append(spearmanr(pred[test], test_y)[0])
    got = residualized_cv(d, "collateral_raw", ["frequency", "effect_l2"], columns)
    assert got["n_pred"] == 2
    np.testing.assert_allclose(got["cv_spearman_mean"], np.mean(scores), atol=1e-12)
    np.testing.assert_allclose(got["oof_spearman"], spearmanr(pred, residual)[0], atol=1e-12)


def test_undefined_correlations_and_adjustments_stay_undefined():
    from residualize import holm, benjamini_hochberg
    d = make_df()
    d["collateral_raw"] = 0
    got = partial_spearman(d, "crowding", "collateral_raw", [], 100, np.random.default_rng(0))
    assert np.isnan(got["rho"]) and np.isnan(got["p"])
    d["crowding"] = np.nan
    assert partial_spearman(d, "crowding", "collateral_raw", [], 100, np.random.default_rng(0))["n"] == 0
    for adjust in [holm, benjamini_hochberg]:
        got = adjust([0.01, np.nan, 0.2])
        assert np.isnan(got[1])
        np.testing.assert_allclose(got[[0, 2]], [0.03, 0.4] if adjust is holm else [0.03, 0.3])


def test_reconstruction_norm_is_not_explained_change():
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from run_setting import residual_change_metrics
    dh = torch.tensor([[1., 0.], [2., 0.]])
    parallel = residual_change_metrics(dh, dh)
    opposite = residual_change_metrics(dh, -dh)
    orthogonal = residual_change_metrics(dh, torch.tensor([[0., 1.], [0., 2.]]))
    assert parallel["resid_reconstruction_norm_ratio"] == opposite["resid_reconstruction_norm_ratio"] == 1
    assert parallel["resid_change_explained_energy"] == 1
    assert opposite["resid_change_explained_energy"] == -3
    assert orthogonal["resid_change_explained_energy"] == -1
    assert np.isnan(residual_change_metrics(torch.zeros_like(dh), torch.zeros_like(dh))["resid_change_explained_energy"])


def test_shared_bootstrap_matches_single_predictor_draws():
    from residualize import partial_spearman_many
    d = make_df(8)
    d["crowding"] = np.round(d.crowding, 1)
    preds = ["crowding", "logit_l2", "enc_norm"]
    for control in [[], CONTROLS["primary"]]:
        group = partial_spearman_many(d, preds, "collateral_raw", control, 257, np.random.default_rng(9), batch_size=64)
        for pred in preds:
            single = partial_spearman(d, pred, "collateral_raw", control, 257, np.random.default_rng(9))
            for field in ["rho", "ci_lo", "ci_hi", "p"]:
                np.testing.assert_allclose(group[pred][field], single[field], rtol=1e-7, atol=1e-9)


def test_reliability_requires_matched_unique_feature_ids():
    import pytest
    from variants import label_reliability
    A = make_df(9).assign(feature=np.arange(300))
    B = A.iloc[::-1].copy()
    r = label_reliability(A, B, 100, np.random.default_rng(1))
    assert abs(r["collateral_raw"]["rho"] - 1) < 1e-12
    with pytest.raises(ValueError):
        label_reliability(A, B.iloc[:-1], 100, np.random.default_rng(1))
    B.iloc[0, B.columns.get_loc("feature")] = B.iloc[1].feature
    with pytest.raises(ValueError):
        label_reliability(A, B, 100, np.random.default_rng(1))


def test_nuisance_only_cv_does_not_score_roundoff_noise():
    for scale in [1, 1e-6]:
        d = make_df(20)
        d["collateral_raw"] *= scale
        r = residualized_cv(d, "collateral_raw", ["frequency", "act_mag"], ["frequency"])
        assert np.isnan(r["cv_spearman_mean"])
        assert np.isnan(r["oof_spearman"])
        assert r["n_defined_folds"] == 0


def test_constant_reliability_serializes_as_null():
    import json
    from variants import label_reliability, json_safe
    A = make_df(21).assign(feature=np.arange(300), stab_anti_frac=0.)
    B = A.copy()
    B.loc[0, "stab_anti_frac"] = 0.1
    r = label_reliability(A, B, 100, np.random.default_rng(0))
    assert np.isnan(r["stab_anti_frac"]["rho"])
    value = json.loads(json.dumps(json_safe(r), allow_nan=False))
    assert value["stab_anti_frac"]["rho"] is None


def test_frequency_rank_alias_is_undefined_after_control():
    from residualize import partial_spearman_many
    d = make_df(22)
    f = d.frequency.to_numpy()
    d["bin_entropy"] = -f*np.log(f) - (1-f)*np.log(1-f)
    one = partial_spearman(d, "bin_entropy", "collateral_raw", ["frequency"], 100, np.random.default_rng(0))
    many = partial_spearman_many(d, ["crowding", "bin_entropy"], "collateral_raw", ["frequency"], 100, np.random.default_rng(0))
    assert np.isnan(one["rho"]) and np.isnan(many["bin_entropy"]["rho"])
    assert np.isfinite(many["crowding"]["rho"])


def test_bootstrap_with_constant_draws_does_not_invent_zero_correlations():
    d = make_df(23)
    d["collateral_raw"] = 0.
    d.loc[0, "collateral_raw"] = 1.
    r = partial_spearman(d, "crowding", "collateral_raw", [], 100, np.random.default_rng(0))
    assert np.isfinite(r["rho"])
    assert 0 < r["n_boot_valid"] < 100
    assert np.isnan(r["ci_lo"]) and np.isnan(r["ci_hi"])
