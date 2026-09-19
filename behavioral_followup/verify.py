"""Read-only audit of completed generations, score provenance and split integrity."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
import re

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=Path(os.environ.get('DUAN_REPO_ROOT',ROOT.parent)).resolve()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()



def verify_rows(rows, expected):
    lookup={r['job_id']:r for r in rows}
    assert len(lookup)==len(rows)==len(expected), 'Incomplete or duplicate output grid'
    assert set(lookup)=={j['job_id'] for j in expected}, 'Unexpected output jobs'
    for job in expected:
        row=lookup[job['job_id']]
        assert all(row[k]==v for k,v in job.items()), 'Output job metadata changed'
        assert all(type(t) is int and t>=0 for t in row['token_ids'])
        assert len(row['token_ids'])<=(32 if row['kind']=='review' else 8)
        words=re.findall(r"\b[\w']+\b",row['text'].lower())
        trigrams=list(zip(words,words[1:],words[2:]))
        repeated=1-len(set(trigrams))/len(trigrams) if trigrams else 0.
        degenerate=len(words)<8 or repeated>.2
        assert row['word_count']==len(words) and row['degenerate']==degenerate
        assert np.isclose(row['repeated_trigram_fraction'],repeated,atol=1e-12,rtol=0)
        if row['kind']=='review':
            probability=row['positive_probability']
            assert np.isfinite(probability) and 0<=probability<=1
            assert row['positive_label']==(probability>=.5)
            assert row['nondegenerate_positive']==(probability>=.5 and not degenerate)
        else:
            first=row['text'].lstrip().split('\n',1)[0].strip().lstrip(chr(34)+chr(39)+' ')
            correct=bool(re.match(re.escape(row['answer'])+r'\b',first,re.I))
            assert row['correct']==correct, 'Factual score does not match output'


def expected_jobs(prompts, features, alphas, seeds, split, arm):
    return [dict(job_id=f"{split}:{p['id']}:{int(f)}:{a:g}:{seed}",prompt_id=p['id'],
                 kind=p['kind'],prompt=p['prompt'],answer=p.get('answer'),feature=int(f),
                 alpha=float(a),seed=seed,arm=arm)
            for a in alphas for f in features for p in prompts
            for seed in (seeds if p['kind']=='review' else [0])]




def reconstruct_scores(candidates, training, predictor):
    from sklearn.preprocessing import StandardScaler

    columns=predictor['columns']
    scaler=StandardScaler().fit(training[columns])
    X=scaler.transform(candidates[columns].astype(np.float32))
    return {name:X@np.array(coefs['coef'])+coefs['intercept']
            for name,coefs in predictor['coefficients'].items()}

def verify_predictor_fit(out, candidates, training):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    base=BASE/'results'
    setting=out.name
    original=pd.read_csv(base/setting/'per_feature.csv')
    pd.testing.assert_frame_equal(training,original,check_exact=False,rtol=1e-12,atol=1e-12)
    excluded=set(original.feature)
    for seed in [1,2]:
        excluded.update(pd.read_csv(base/f'{setting}_seed{seed}'/'per_feature.csv').feature)
    assert not set(candidates.feature)&excluded
    saved=json.loads((out/'score_model.json').read_text())
    scaler=StandardScaler().fit(training[saved['columns']])
    np.testing.assert_allclose(scaler.mean_,saved['mean'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(scaler.scale_,saved['scale'],rtol=1e-12,atol=1e-12)
    X=scaler.transform(training[saved['columns']])
    y=training.collateral_raw.to_numpy()
    Z=np.c_[np.ones(len(training)),training[['effect_l2','act_mag','frequency']]]
    residual=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    errors={}
    for name,target in [('predicted_collateral',y),('predicted_residual_collateral',residual)]:
        fitted=Ridge(alpha=1).fit(X,target)
        coeff=saved['coefficients'][name]
        np.testing.assert_allclose(fitted.coef_,coeff['coef'],rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(fitted.intercept_,coeff['intercept'],rtol=1e-9,atol=1e-9)
        errors[name]=float(np.max(np.abs(fitted.coef_-coeff['coef'])))
    return {'passed':True,'excluded_source_features':len(excluded),'candidate_overlap':0,
            'canonical_training_sha256':sha(base/setting/'per_feature.csv'),
            'refit_max_coefficient_errors':errors}

def verify_secondary(out, rows):
    from blind_judge import CRITERIA, prompt
    result=json.loads((out/'blind_judge.json').read_text())
    assert result['test_sha256']==sha(out/'test.jsonl')
    assert result['protocol_sha256']==sha(ROOT/'SECONDARY_JUDGE_PROTOCOL.md')
    assert result['prompts_sha256']==sha(ROOT/'prompts.json')
    source=ROOT/'results'/'source_snapshots'/f"blind_judge_{result['source_sha256']}.py"
    assert sha(source)==result['source_sha256']
    assert result['criteria']==CRITERIA
    for name,digest in result['output_sha256'].items():
        assert sha(out/name)==digest
    fixed_ids={'test_review_00','test_review_08','test_review_16','test_review_24'}
    subset=[r for r in rows if r['kind']=='review' and r['prompt_id'] in fixed_ids and r['seed']==20260918]
    raw=[json.loads(x) for x in (out/'blind_judge_raw.jsonl').read_text().splitlines()]
    lookup={(r['opaque_id'],r['criterion']):r for r in raw}
    assert len(raw)==len(lookup)==3*len(subset)
    assert len(subset)==result['n_outputs']
    scored=[]
    for row in subset:
        opaque=hashlib.sha256(row['job_id'].encode()).hexdigest()
        record={'arm':row['arm']}
        for criterion in CRITERIA:
            r=lookup[(opaque,criterion)]
            assert set(r)=={'opaque_id','criterion','query','probability_A','pass_criterion'}
            assert r['query']==prompt(row['prompt'],row['text'],criterion)
            assert 0<=r['probability_A']<=1
            assert r['pass_criterion']==(r['probability_A']>=.5)
            record[criterion]=r['pass_criterion']
        record['positive_coherent_on_topic']=all(record[k] for k in CRITERIA)
        record['sentiment_agrees']=record['positive']==row['positive_label']
        scored.append(record)
    frame=pd.DataFrame(scored)
    for arm,group in frame.groupby('arm'):
        assert result['arms'][arm]['n_outputs']==len(group)
        for metric in [*CRITERIA,'positive_coherent_on_topic','sentiment_agrees']:
            assert np.isclose(result['arms'][arm][metric],group[metric].mean(),atol=1e-9,rtol=0)
    fixed=json.loads((ROOT/'prompts.json').read_text())['calibration']
    cal=result['calibration_rows']
    assert len(cal)==len(fixed)==result['calibration_n']
    for observed,reference in zip(cal,fixed):
        assert observed['query']==prompt('A personal review.',reference['text'],'positive')
        assert observed['expected']==bool(reference['label'])
        assert observed['pass_criterion']==(observed['probability_A']>=.5)
    accuracy=float(np.mean([r['expected']==r['pass_criterion'] for r in cal]))
    assert result['calibration_accuracy']==accuracy
    assert result['calibration_passed']==(accuracy>=.9)
    return {'passed':True,'n_outputs':len(subset),'n_judgments':len(raw),
            'calibration_accuracy':accuracy,'calibration_passed':bool(accuracy>=.9),
            'source_sha256':result['source_sha256']}

def main():
    prompts=json.loads((ROOT/'prompts.json').read_text())
    dev={p['id']:p for p in prompts['development']}
    test={p['id']:p for p in prompts['test']}
    assert not {p['prompt'] for p in dev.values()} & {p['prompt'] for p in test.values()}
    record={'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'settings':{}}
    for setting in ['gpt2_small','gemma_2_2b']:
        out=ROOT/'results'/setting
        if not (out/'analysis.json').exists():
            continue
        provenance=json.loads((out/'manifest.json').read_text())
        assert provenance['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
        assert provenance['prompts_sha256']==sha(ROOT/'prompts.json')
        assert provenance['candidates_sha256']==sha(out/'candidates.csv')
        assert all(version(package)==expected for package,expected in provenance['versions'].items())
        candidates=pd.read_csv(out/'candidates.csv')
        training=pd.read_csv(out/'score_training.csv')
        assert candidates.feature.is_unique and not set(candidates.feature)&set(training.feature)
        assert len(training)==300
        fit_check=verify_predictor_fit(out,candidates,training)
        predictor=json.loads((out/'score_model.json').read_text())
        reconstructed_scores=reconstruct_scores(candidates,training,predictor)
        score_errors={}
        for name,coefs in predictor['coefficients'].items():
            reconstructed=reconstructed_scores[name]
            score_errors[name]=float(np.max(np.abs(reconstructed-candidates[name])))
            np.testing.assert_allclose(reconstructed,candidates[name],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(np.argsort(reconstructed),np.argsort(candidates[name]))
        selection=json.loads((out/'selection.json').read_text())
        manifest=json.loads((out/'test_manifest.json').read_text())
        assert selection['prompts_sha256']==sha(ROOT/'prompts.json')
        assert selection['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
        assert selection['development_sha256']==sha(out/'development.jsonl')
        assert selection['candidates_sha256']==sha(out/'candidates.csv')
        assert manifest['selection_sha256']==sha(out/'selection.json')
        source=ROOT/'results'/'source_snapshots'/f"run_{manifest['runner_sha256']}.py"
        assert source.exists() and sha(source)==manifest['runner_sha256']
        rows=[json.loads(x) for x in (out/'test.jsonl').read_text().splitlines()]
        assert len(rows)==manifest['n_jobs']==len({r['job_id'] for r in rows})
        expected=expected_jobs(prompts['test'],[-1],[0],prompts['review_seeds'],'test','unsteered')
        for arm in ['low','high']:
            expected+=expected_jobs(prompts['test'],[p[arm] for p in selection['pairs']],
                                    [selection['alpha']],prompts['review_seeds'],'test',arm)
        expected+=expected_jobs(prompts['test'],selection['random'],[selection['alpha']],
                                prompts['review_seeds'],'test','random')
        job_digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        assert manifest['job_sha256']==job_digest
        verify_rows(rows,expected)
        dev_expected=expected_jobs(prompts['development'],[-1],[0],[prompts['development_seed']],'dev','unsteered')
        dev_expected+=expected_jobs([p for p in prompts['development'] if p['kind']=='review'],
                                    candidates.feature.tolist(),[1,2,4,8],[prompts['development_seed']],'dev','candidate')
        dev_rows=[json.loads(x) for x in (out/'development.jsonl').read_text().splitlines()]
        verify_rows(dev_rows,dev_expected)
        analysis=json.loads((out/'analysis.json').read_text())
        assert analysis['test_sha256']==sha(out/'test.jsonl')
        assert analysis['selection_sha256']==sha(out/'selection.json')
        assert analysis['selection']==selection and analysis['n_outputs']==len(rows)
        feature_arm={-1:'unsteered',**{p['low']:'low' for p in selection['pairs']},
                     **{p['high']:'high' for p in selection['pairs']},**{i:'random' for i in selection['random']}}
        assert len(feature_arm)==1+2*len(selection['pairs'])+len(selection['random'])
        for row in rows:
            p=test[row['prompt_id']]
            assert row['prompt']==p['prompt'] and row['answer']==p.get('answer')
            assert row['arm']==feature_arm[row['feature']]
            assert row['alpha']==(0 if row['feature']==-1 else selection['alpha'])
            assert len(row['token_ids'])<=(32 if row['kind']=='review' else 8)
        d=pd.DataFrame(rows)
        for feature in feature_arm:
            r=d[d.feature==feature]
            assert len(r[r.kind=='review'])==128 and len(r[r.kind=='qa'])==64
            rev=r[r.kind=='review']; facts=r[r.kind=='qa']
            expected_reviews={(pid,seed) for pid,prompt in test.items() if prompt['kind']=='review' for seed in prompts['review_seeds']}
            assert set(zip(rev.prompt_id,rev.seed))==expected_reviews
            assert set(facts.prompt_id)=={pid for pid,prompt in test.items() if prompt['kind']=='qa'}
        pdv=pd.read_csv(out/'development_features.csv')
        for pair in selection['pairs']:
            assert abs(pair['low_dev_gain']-pair['high_dev_gain'])<=.1
            assert pair['low_dev_gain']>=.05 and pair['high_dev_gain']>=.05
            for arm in ['low','high']:
                row=pdv[(pdv.feature==pair[arm])&(pdv.alpha==selection['alpha'])].iloc[0]
                assert row.degenerate_rate<=.25
                assert (row.predicted_collateral<=selection['candidate_median_score'])==(arm=='low')
        record['settings'][setting]={'passed':True,'n_candidates':len(candidates),'n_pairs':len(selection['pairs']),
            'n_test_outputs':len(rows),'candidate_training_overlap':0,
            'score_reconstruction_max_error':score_errors,'predictor_fit':fit_check,
            'package_versions_match_manifest':True,
            'score_precision_note':'Candidate predictors were computed and standardized as float32. Restore that dtype after CSV loading and use the original StandardScaler transformation before applying saved coefficients. Absolute reconstruction error must be at most 1e-10 and every candidate rank must agree exactly.',
            'checks':['independently refitted predictor coefficients','independently reconstructed predictor scores',
                      'disjoint prompts and all three source feature samples',
                      'complete development and test job metadata and score reconstruction',
                      'complete output grid including failures','development-only matched selection',
                      'exact saved source for generation','unchanged frozen prompt/selection/protocol hashes'],
            'hashes':{name:sha(out/name) for name in ['candidates.csv','selection.json','development.jsonl','test.jsonl','analysis.json']}}
        if (out/'blind_judge.json').exists():
            record['settings'][setting]['secondary_judge']=verify_secondary(out,rows)
    resume=json.loads((ROOT/'RESUME_2026-09-18.json').read_text())
    for name,digest in resume['frozen_sha256'].items():
        path=ROOT/'results'/'source_snapshots'/f'run_{digest}.py' if name=='run.py' else ROOT/name
        assert sha(path)==digest
    prefix=(ROOT/'results/gemma_2_2b/development.jsonl').read_bytes().splitlines(keepends=True)[:resume['initial_gemma_development_rows']]
    assert hashlib.sha256(b''.join(prefix)).hexdigest()==resume['initial_gemma_development_prefix_sha256']
    record['source_sha256']={name:sha(ROOT/name) for name in
        ['run.py','analyze.py','verify.py','blind_judge.py','prompts.py','test_followup.py']}
    record['resume_preserved']=True
    record['complete']=set(record['settings'])=={'gpt2_small','gemma_2_2b'} and all(
        'secondary_judge' in s for s in record['settings'].values())
    assert record['settings']
    (ROOT/'VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
