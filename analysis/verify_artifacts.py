#!/usr/bin/env python
"""Verify delivered results against saved inputs and an optional earlier commit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ORDER = ['gpt2_small', 'pythia_70m_deduped', 'gemma_2_2b', 'llama_3_1_8b']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', help='optional commit whose shared per-feature columns must be unchanged')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root/'results'
    part = pd.read_csv(results/'analysis/partial_correlations.csv')
    variants = json.loads((results/'analysis/variants.json').read_text())
    record = {'settings': {}, 'reference': args.reference, 'checks': []}
    assert set(part.setting) == set(ORDER)
    assert (part.n_boot == 10000).all()
    assert not part.duplicated(['setting','target','control','predictor']).any()
    for field in ['p_holm', 'q_bh']:
        valid = part.p.notna()
        assert part.loc[valid, field].notna().all()
        assert (part.loc[valid, field] >= part.loc[valid, 'p']-1e-14).all()
        assert part.loc[~valid, field].isna().all()
    for setting in ORDER:
        path = results/setting/'per_feature.csv'
        d = pd.read_csv(path)
        r = pd.read_csv(results/(setting+'_random_paired')/'per_feature.csv')
        a = json.loads((results/setting/'selection.json').read_text())
        b = json.loads((results/(setting+'_random_paired')/'selection.json').read_text())
        assert len(d) == len(r) == 300 and d.feature.is_unique and r.feature.is_unique
        assert a['contexts'] == b['contexts'] and a['panel'] == b['panel']
        assert all(len(c) == len(set(c)) == 48 for c in a['contexts'])
        assert r.matched_feature.tolist() == d.feature.tolist()
        assert r[['frequency','act_mag','enc_norm']].isna().all().all()
        vectors = np.load(results/(setting+'_random_paired')/'directions.npz')['directions']
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), d.dec_norm, rtol=3e-6)
        assert (d.collateral_raw_nodense <= d.collateral_raw+1e-12).all()
        np.testing.assert_allclose(d.collateral_ctilde, d.collateral_raw/(d.effect_l2+1e-8), rtol=1e-12)
        assert np.isfinite(d[['resid_change_explained_energy', 'resid_error_delta_norm']]).all().all()
        checked = 0
        if args.reference:
            old_text = subprocess.check_output(['git','show',f'{args.reference}:results/{setting}/per_feature.csv'], cwd=root, text=True)
            old = pd.read_csv(io.StringIO(old_text)).rename(columns={'resid_sae_frac':'resid_reconstruction_norm_ratio'})
            shared = list(old.columns.intersection(d.columns))
            np.testing.assert_allclose(old[shared], d[shared], rtol=0, atol=1e-12)
            checked = len(shared)
        for target in ['collateral_raw', 'collateral_ctilde']:
            for control, names in [('none',[]), ('robust',['frequency','act_mag']), ('primary',['effect_l2','act_mag','frequency'])]:
                x, y = rankdata(d.crowding), rankdata(d[target])
                Z = np.column_stack([np.ones(len(d))] + [rankdata(d[c]) for c in names])
                rx = x - Z @ np.linalg.lstsq(Z,x,rcond=None)[0]
                ry = y - Z @ np.linalg.lstsq(Z,y,rcond=None)[0]
                ref = np.corrcoef(rx,ry)[0,1]
                row = part[(part.setting==setting)&(part.target==target)&(part.control==control)&(part.predictor=='crowding')].iloc[0]
                np.testing.assert_allclose(ref,row.rho,atol=1e-9,rtol=0)
            A = pd.read_csv(results/(setting+'_ctxA')/'per_feature.csv')
            B = pd.read_csv(results/(setting+'_ctxB')/'per_feature.csv')
            paired = A.merge(B,on='feature',suffixes=('_A','_B'),validate='one_to_one')
            rel = spearmanr(paired[target+'_A'],paired[target+'_B'])[0]
            np.testing.assert_allclose(rel,variants['reliability'][setting]['labels'][target]['rho'],atol=1e-10)
        record['settings'][setting] = {'n':len(d), 'shared_reference_columns':checked,
                                       'per_feature_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                       'paired_contexts_and_norms_verified':True,
                                       'headline_partials_match_independent_lstsq':True,
                                       'reliability_matches_independent_spearman':True}
    record['checks'] = ['four complete settings', '10000 bootstrap draws', 'unique analysis keys',
                        'multiplicity corrections preserve undefined tests', 'matched random controls',
                        'effect normalization', 'dense counts bounded by full counts']
    (results/'analysis/artifact_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
