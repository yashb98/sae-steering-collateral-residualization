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
        assert not df[sorted(required)].isna().any().any(), p
        assert df.feature.is_unique, p
