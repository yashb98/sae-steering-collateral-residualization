"""Paired feature/prompt bootstrap and complete behavioral reporting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent


def bootstrap_pairs(values,n_boot=5000,seed=180926):
    values=np.asarray(values,float)
    if not values.size:
        return {'difference':None,'ci_low':None,'ci_high':None,'n_pairs':0,'n_prompts':0}
    assert values.ndim==2 and np.isfinite(values).all()
    rng=np.random.default_rng(seed)
    pairs=rng.integers(values.shape[0],size=(n_boot,values.shape[0]))
    prompts=rng.integers(values.shape[1],size=(n_boot,values.shape[1]))
    draws=values[pairs[:,:,None],prompts[:,None,:]].mean(axis=(1,2))
    lo,hi=np.quantile(draws,[.025,.975])
    return {'difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi),
            'n_pairs':values.shape[0],'n_prompts':values.shape[1],'n_boot':n_boot}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(setting):
    out=ROOT/'results'/setting
    if not (out/'test.jsonl').exists():
        return None
    selection=json.loads((out/'selection.json').read_text())
    manifest=json.loads((out/'test_manifest.json').read_text())
    assert manifest['selection_sha256']==sha(out/'selection.json')
    rows=[json.loads(x) for x in (out/'test.jsonl').read_text().splitlines()]
    assert len(rows)==manifest['n_jobs'],f'{setting}: incomplete test generations'
    d=pd.DataFrame(rows)
    assert d.job_id.is_unique
    assert len(d[d.kind=='review'].prompt_id.unique())==64
    assert len(d[d.kind=='qa'].prompt_id.unique())==64
    candidates=pd.read_csv(out/'candidates.csv')
    reviews=d[d.kind=='review'].copy()
    qa=d[d.kind=='qa'].copy()
    base_qa=qa[qa.feature==-1].set_index('prompt_id').correct.astype(float)
    base_review=reviews[reviews.feature==-1].groupby('prompt_id').positive_probability.mean()
    per_feature=[]
    for feature,f in d.groupby('feature'):
        rev=f[f.kind=='review']
        answers=f[f.kind=='qa'].set_index('prompt_id').correct.astype(float).reindex(base_qa.index)
        assert len(rev)==128 and len(answers)==64 and answers.notna().all()
        wrong=(1-answers)*base_qa
        repairs=answers*(1-base_qa)
        row={'feature':int(feature),'arm':f.arm.iloc[0],'alpha':float(f.alpha.iloc[0]),
             'positive_probability':float(rev.positive_probability.mean()),
             'positive_rate':float(rev.positive_label.astype(float).mean()),
             'nondegenerate_positive_rate':float(rev.nondegenerate_positive.astype(float).mean()),
             'degenerate_rate':float(rev.degenerate.mean()),'mean_words':float(rev.word_count.mean()),
             'repeated_trigram_fraction':float(rev.repeated_trigram_fraction.mean()),
             'qa_accuracy':float(answers.mean()),'qa_accuracy_loss':float(base_qa.mean()-answers.mean()),
             'broken_baseline_correct_rate':float(wrong.sum()/base_qa.sum()) if base_qa.sum() else None,
             'repaired_baseline_wrong_rate':float(repairs.sum()/(1-base_qa).sum()) if (1-base_qa).sum() else None,
             'sentiment_gain':float(rev.positive_probability.mean()-base_review.mean())}
        per_feature.append(row)
    features=pd.DataFrame(per_feature).merge(candidates[['feature','predicted_collateral','predicted_residual_collateral',
             'crowding','discovery_selectivity']],on='feature',how='left',validate='one_to_one')
    features.to_csv(out/'test_features.csv',index=False)
    metric_names=['positive_probability','positive_rate','nondegenerate_positive_rate','degenerate_rate',
                  'repeated_trigram_fraction','qa_accuracy','qa_accuracy_loss','sentiment_gain','broken_baseline_correct_rate']
    arms=features.groupby('arm')[metric_names].mean()
    arms['n_features']=features.groupby('arm').size()
    arms.to_csv(out/'arm_summary.csv')
    contrasts={}
    for metric,frame in [('positive_probability',reviews),('positive_label',reviews),
                         ('nondegenerate_positive',reviews),('degenerate',reviews),('correct',qa)]:
        pivot=frame.groupby(['feature','prompt_id'])[metric].mean().unstack('prompt_id')
        differences=np.array([(pivot.loc[p['low']]-pivot.loc[p['high']]).to_numpy(float) for p in selection['pairs']])
        contrasts[metric]=bootstrap_pairs(differences)
    primary=contrasts['positive_probability']
    preservation=contrasts['correct']
    gates=selection['feature_gate'] and selection['qa_sensitivity_gate']
    if not gates:
        verdict='Selection or benchmark gate failed; selector transfer is not established.'
    elif primary['ci_low']>-.05 and preservation['ci_low']>0:
        verdict='Promising exploratory transfer: the success noninferiority and factual-preservation criteria both pass.'
    elif primary['ci_high']<-.05:
        verdict='Low predicted collateral loses intended steering success; a cleaner-steering benefit is not established.'
    elif preservation['ci_high']<0:
        verdict='Low predicted collateral harms factual accuracy relative to high-score features on this test.'
    else:
        verdict='The joint behavioral-transfer criterion is not met; report uncertainty and the observed tradeoff.'
    valid=features[features.feature>=0]
    correlations={}
    for score in ['predicted_collateral','predicted_residual_collateral','crowding']:
        correlations[score]={}
        for target in ['qa_accuracy_loss','sentiment_gain','degenerate_rate']:
            x,y=valid[score],valid[target]
            correlations[score][target]=float(spearmanr(x,y).statistic) if x.nunique()>1 and y.nunique()>1 else None
    report={'setting':setting,'n_outputs':len(d),'selection':selection,'arms':json.loads(arms.to_json(orient='index')),
            'low_minus_high':contrasts,'exploratory_feature_correlations':correlations,'verdict':verdict,
            'test_sha256':sha(out/'test.jsonl'),'selection_sha256':sha(out/'selection.json')}
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    sample=[]
    for arm in ['unsteered','low','high','random']:
        f=features[features.arm==arm]
        if f.empty:
            continue
        chosen=int(f.sort_values('feature').iloc[0].feature)
        subset=d[(d.feature==chosen)&d.prompt_id.isin(['test_review_00','test_review_08','test_review_16','test_review_24','test_qa_00','test_qa_08'])]
        subset=subset[(subset.kind=='qa')|(subset.seed==20260918)]
        sample+=json.loads(subset.to_json(orient='records'))
    (out/'fixed_output_examples.json').write_text(json.dumps(sample,indent=2,allow_nan=False)+'\n')
    return report


def pct(x):
    return f'{100*x:.1f}%'


def ci(x):
    if x['difference'] is None:
        return 'undefined'
    return f"{100*x['difference']:+.1f} pp [{100*x['ci_low']:+.1f}, {100*x['ci_high']:+.1f}]"



def plot_contrasts(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharey=True,sharex=True)
    labels={'gpt2_small':'GPT-2-small','gemma_2_2b':'Gemma-2-2B'}
    for ax,metric,title in zip(axes,['positive_probability','correct'],
                             ['Positive-sentiment probability','Factual-answer accuracy']):
        for i,r in enumerate(reports):
            c=r['low_minus_high'][metric]
            if c['difference'] is None:
                continue
            x=100*c['difference']
            ax.errorbar(x,i,xerr=[[max(0,x-100*c['ci_low'])],[max(0,100*c['ci_high']-x)]],
                        fmt='o',color='#245C8E',capsize=4,markersize=6)
        ax.axvline(0,color='0.6',linewidth=1)
        if metric=='positive_probability':
            ax.axvline(-5,color='#A34E24',linestyle='--',linewidth=1,
                       label='Noninferiority threshold')
            ax.legend(loc='lower right',fontsize=8,frameon=False)
        else:
            ax.text(.5,.06,'No factual correctness changes\non these test prompts',
                    ha='center',transform=ax.transAxes,fontsize=8,color='0.35')
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('Low-score minus high-score (percentage points)')
        ax.set_yticks(range(len(reports)),[labels[r['setting']] for r in reports])
        ax.set_ylim(-.65,len(reports)-.35)
        ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Behavioral transfer of collateral scores',fontsize=13)
    fig.text(.5,.02,'95% pointwise intervals; matched feature pairs and prompts resampled; review seeds kept together.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.95))
    out=ROOT/'results'/'figures'
    out.mkdir(exist_ok=True)
    for extension in ['png','pdf','svg']:
        fig.savefig(out/f'behavioral_contrasts.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def write_report(reports):
    lines=['# Behavioral transfer of collateral scores','',
        'Completed results from the locally frozen September 18 protocol. This is an exploratory two-setting study of positive-sentiment steering and factual-retrieval preservation. No generated-output outcome was used to train the original collateral predictor. Development generations selected strength and matched feature pairs; all numbers below use held-out prompts.','']
    for r in reports:
        setting=r['setting']; s=r['selection']; arms=r['arms']
        lines += [f'## {setting}','',r['verdict'],'',
            f"Coefficient {s['alpha']}; {len(s['pairs'])} low/high feature pairs; {len(s['random'])} random candidate features; {r['n_outputs']:,} held-out outputs. Each feature has 64 review prompts with two seeds and 64 factual prompts. Development factual accuracy: {pct(s['baseline_qa_accuracy'])}.",'',
            '| Selector | Features | Positive probability | Positive rate | Positive and nondegenerate | Factual accuracy | Degenerate reviews |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ['unsteered','low','high','random']:
            if arm not in arms:
                continue
            a=arms[arm]
            lines.append(f"| {arm} | {int(a['n_features'])} | {pct(a['positive_probability'])} | {pct(a['positive_rate'])} | {pct(a['nondegenerate_positive_rate'])} | {pct(a['qa_accuracy'])} | {pct(a['degenerate_rate'])} |")
        lines += ['', 'Low-score minus high-score differences, with 95% pointwise feature/prompt bootstrap intervals:','',
            f"- Mean positive probability: {ci(r['low_minus_high']['positive_probability'])}.",
            f"- Factual accuracy: {ci(r['low_minus_high']['correct'])}.",
            f"- Positive and nondegenerate output rate: {ci(r['low_minus_high']['nondegenerate_positive'])}.",
            f"- Degenerate review rate: {ci(r['low_minus_high']['degenerate'])}.",'',
            'Intervals resample matched feature pairs and prompt identities, retaining both review seeds together. A promising result requires the lower interval bound for sentiment difference to exceed -5 percentage points and the lower bound for factual accuracy to exceed zero.','',
            f"Evidence: [all generated outputs](results/{setting}/test.jsonl), [feature results](results/{setting}/test_features.csv), [frozen selection](results/{setting}/selection.json), [complete analysis](results/{setting}/analysis.json), [fixed output examples](results/{setting}/fixed_output_examples.json).",'']
    lines += ['## Matched comparisons','',
        '![Low-minus-high behavioral differences with paired bootstrap intervals](results/figures/behavioral_contrasts.png)','',
        '[PDF](results/figures/behavioral_contrasts.pdf) | [SVG](results/figures/behavioral_contrasts.svg)','',
        '## Secondary blinded assessment','',
        'Qwen3-4B independently assessed four fixed product-review prompts per evaluated feature, using the first generation seed. These four prompts share one prefix template. Arm labels, feature IDs, intervention strengths and original scores were hidden from the judge. Rates below are descriptive checks on this small fixed subset, not additional success criteria.','']
    for r in reports:
        setting=r['setting']
        path=ROOT/'results'/setting/'blind_judge.json'
        if not path.exists():
            lines += [f'{setting}: secondary assessment pending.','']
            continue
        judge=json.loads(path.read_text())
        assert judge['test_sha256']==r['test_sha256']
        lines += [f"### {setting}",'',
            f"Sentiment calibration: {pct(judge['calibration_accuracy'])} on {judge['calibration_n']} separate sentences. Assessed {judge['n_outputs']} held-out outputs.",'']
        if not judge['calibration_passed']:
            lines += ['The sentiment calibration gate failed. These judge ratings are diagnostic and cannot corroborate primary sentiment results.','']
        lines += ['| Selector | Outputs | Positive | Coherent | On topic | All three | Sentiment agreement |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ['unsteered','low','high','random']:
            if arm not in judge['arms']:
                continue
            a=judge['arms'][arm]
            lines.append(f"| {arm} | {int(a['n_outputs'])} | {pct(a['positive'])} | {pct(a['coherent'])} | {pct(a['on_topic'])} | {pct(a['positive_coherent_on_topic'])} | {pct(a['sentiment_agrees'])} |")
        lines += ['',f'[Judge scores and provenance](results/{setting}/blind_judge.json).','']
    lines += ['## Interpretation limits','',
        '- Sentiment labels come from one external pretrained classifier. Calibration sentences validate obvious sentiment examples, not semantic adequacy of all generated text. Repetition/length filters are limited quality checks, not human evaluations.',
        '- The factual benchmark is synthetic retrieval from facts in the prompt; it does not establish preservation of general knowledge, reasoning, safety or all unrelated behavior. Identical factual outcomes yield a zero-width empirical difference interval; this does not prove equivalence or preservation outside these prompts.',
        '- The secondary judge checks only four product prompts sharing one template. It is another automated model, has no human validation, and cannot establish broad semantic quality.',
        '- Repeated generation-time steering differs from the original final-token intervention. The transferred label is behavioral correctness, not the original activation-count target.',
        '- Development matching can fail to equalize held-out sentiment success. Both success and collateral must be interpreted together.',
        '- Random controls are other sentiment-associated SAE features, not isotropic random vectors.',
        '- Models use fixed pretrained dictionaries, one behavior and a small selected feature sample. Bootstrap intervals are conditional on this setting and are not corrected across all exploratory comparisons.',
        '- A null or adverse result is retained; no final-test result triggers retuning or re-selection.','',
        '## Reproduction','',
        'Use the original duan environment and existing model cache. The source repository path is configured in run.py. Read PROTOCOL.md before running.','',
        '```bash','python run.py --setting gpt2_small --stage all','python run.py --setting gemma_2_2b --stage all','python blind_judge.py','python analyze.py','python verify.py','python -m pytest -q test_followup.py','```','',
        'The runner saves progress per generation and resumes completed jobs. Remove results only in a separate copy to obtain an independent rerun. Prompt generation, score coefficients, source hashes and package versions are saved alongside the results.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    reports=[r for s in ['gpt2_small','gemma_2_2b'] if (r:=analyze(s)) is not None]
    assert reports,'No completed test results'
    plot_contrasts(reports)
    write_report(reports)
    for r in reports:
        print(r['setting'],r['verdict'])
