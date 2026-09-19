"""Run the frozen behavioral transfer pilot, retaining every generated output."""
import argparse
from collections import Counter
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import yaml

from prompts import build, digest

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('DUAN_REPO_ROOT', ROOT.parent)).resolve()
sys.path.insert(0, str(BASE / 'src'))
from run_setting import sae_field
sys.path.insert(0, str(BASE / 'analysis'))
from residualize import PRED_SETS

START = time.time()


def log(message):
    print(f'[{time.time()-START:.1f}s] {message}', flush=True)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    path = ROOT / 'prompts.json'
    data = build()
    if path.exists():
        assert json.loads(path.read_text()) == data, 'Frozen prompts differ from generator'
    else:
        save(path, data)
    return data


def load_setting(setting):
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    cfg = yaml.safe_load((BASE / 'configs' / f'{setting}.yaml').read_text())
    loaded = SAE.from_pretrained(release=cfg['sae_release'], sae_id=cfg['primary_sae_id'], device='cuda')
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    sae = sae.eval()
    kwargs = dict(sae_field(sae, 'model_from_pretrained_kwargs', {}) or {})
    model = HookedTransformer.from_pretrained(cfg['model_name'], device='cuda', dtype=torch.float32, **kwargs).eval()
    assert sae_field(sae, 'hook_name') == cfg['primary_hook']
    log(f'Loaded {setting} and primary SAE')
    return model, sae, cfg


def tokens(model, texts, prepend_bos=True):
    tok = model.tokenizer
    rows = [tok.encode(t, add_special_tokens=False) for t in texts]
    if prepend_bos:
        rows = [[tok.bos_token_id] + row for row in rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device='cuda')
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, -len(row):] = torch.tensor(row, device='cuda')
        mask[i, -len(row):] = 1
    return ids, mask


def activations(model, sae, hook, texts, batch_size=16):
    out = []
    for start in range(0, len(texts), batch_size):
        ids, mask = tokens(model, texts[start:start+batch_size], bool(sae_field(sae, 'prepend_bos', True)))
        _, cache = model.run_with_cache(ids, attention_mask=mask, names_filter=[hook], return_type=None,
                                        stop_at_layer=int(hook.split('.')[1]) + 1)
        out.append(sae.encode(cache[hook][:, -1].float()).cpu())
    return torch.cat(out)


def corpus_activations(model, sae, cfg, out):
    cache_path = out / 'unsteered_activations.npz'
    if cache_path.exists():
        log('Loading unsteered activation cache')
        return torch.from_numpy(np.load(cache_path)['activations'])
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
    seen, rows, texts = set(), [], []
    prepend_bos = bool(sae_field(sae, 'prepend_bos', True))
    for row in ds:
        t = row['text'].strip()
        key = re.sub(r'\s+', ' ', t)
        if len(t) <= 200 or key in seen:
            continue
        seen.add(key)
        ids = model.to_tokens(t, prepend_bos=prepend_bos)[0]
        if len(ids) < 48:
            continue
        rows.append(ids[:48].cpu())
        texts.append(t)
        if len(rows) == 2048:
            break
    assert len(rows) == 2048
    all_ids = torch.stack(rows)
    acts = []
    for start in range(0, 2048, cfg['model_batch']):
        ids = all_ids[start:start+cfg['model_batch']].cuda()
        _, cache = model.run_with_cache(ids, names_filter=[cfg['primary_hook']], return_type=None,
                                        stop_at_layer=int(cfg['primary_hook'].split('.')[1]) + 1)
        acts.append(sae.encode(cache[cfg['primary_hook']][:, -1].float()).cpu())
        if start % (cfg['model_batch'] * 16) == 0:
            log(f'Unsteered corpus: {start}/2048')
    A = torch.cat(acts)
    np.savez_compressed(cache_path, activations=A.numpy(), token_ids=all_ids.numpy())
    save(out / 'corpus_manifest.json', {'n_contexts': 2048, 'text_sha256': digest(texts),
         'token_sha256': hashlib.sha256(all_ids.numpy().tobytes()).hexdigest()})
    return A


def predictors(model, sae, prim, ids):
    from scipy.stats import kurtosis
    F = torch.tensor(ids, device='cuda')
    dec, enc = sae.W_dec.detach().float(), sae.W_enc.detach().float()
    vec = dec[F]
    wn = torch.nn.functional.normalize(dec, dim=-1)
    sim = wn[F] @ wn.T
    sim = sim.abs()
    sim[torch.arange(len(F)), F] = 0
    top = sim.topk(20, dim=-1).values
    a = prim[:, ids].float()
    fires = (a > 1e-6).float()
    p = fires.mean(0).numpy()
    eps = 1e-12
    r = a / (a.sum(0, keepdim=True) + eps)
    B = (prim > 1e-6).float().cuda()
    Bf = B[:, F]
    co = (Bf.T @ B) / (Bf.sum(0)[:, None] + eps)
    co[torch.arange(len(F)), F] = 0
    pi = co / (co.sum(1, keepdim=True) + eps)
    rf = vec @ model.W_U.detach().float()
    mass = rf.abs() / (rf.abs().sum(-1, keepdim=True) + eps)
    encvec = enc[:, F].T
    data = dict(feature=ids, crowding=top.mean(-1).cpu().numpy(), crowd_max=top[:, 0].cpu().numpy(),
        dec_norm=vec.norm(dim=-1).cpu().numpy(), enc_norm=encvec.norm(dim=-1).cpu().numpy(),
        enc_dec_cos=torch.nn.functional.cosine_similarity(encvec, vec, dim=-1).cpu().numpy(),
        frequency=p, act_mag=a.mean(0).numpy(), act_mean_firing=((a*fires).sum(0)/fires.sum(0).clamp(min=1)).numpy(),
        act_std=a.std(0).numpy(), act_max=a.max(0).values.numpy(), act_kurtosis=kurtosis(a.numpy(),axis=0),
        bin_entropy=-(p*np.log(p+eps)+(1-p)*np.log(1-p+eps)),
        act_entropy=(-(r*torch.log(r+eps)).sum(0)/np.log(len(prim))).numpy(),
        coact_entropy=(-(pi*torch.log(pi+eps)).sum(-1)).cpu().numpy(),
        coact_count=((Bf.T@B.sum(1,keepdim=True)).squeeze(1)/(Bf.sum(0)+eps)-1).cpu().numpy(),
        logit_l2=rf.norm(dim=-1).cpu().numpy(), logit_linf=rf.abs().max(-1).values.cpu().numpy(),
        logit_entropy=(-(mass*torch.log(mass+eps)).sum(-1)).cpu().numpy(),
        logit_top10_mass=mass.topk(10,dim=-1).values.sum(-1).cpu().numpy())
    result = pd.DataFrame(data)
    assert np.isfinite(result[PRED_SETS['full_no_magnitude']]).all().all()
    return result


def prepare(setting, model, sae, cfg, data, out):
    if (out / 'candidates.csv').exists():
        return pd.read_csv(out / 'candidates.csv')
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    training_path = BASE / 'results' / setting / 'per_feature.csv'
    train = pd.read_csv(training_path)
    prim = corpus_activations(model, sae, cfg, out)
    discovery = data['discovery']
    pos = activations(model, sae, cfg['primary_hook'], [d['positive'] for d in discovery])
    neg = activations(model, sae, cfg['primary_hook'], [d['negative'] for d in discovery])
    diff = pos - neg
    relevance = (diff.mean(0) / (torch.cat([pos,neg]).std(0) + 1e-6)).numpy()
    freq = (prim > 1e-6).float().mean(0).numpy()
    excluded = set(train.feature)
    for seed in [1,2]:
        excluded.update(pd.read_csv(BASE/'results'/f'{setting}_seed{seed}'/'per_feature.csv').feature)
    eligible = np.where((freq >= .002)&(freq <= .5)&(relevance > 0)&((pos>1e-6).float().mean(0).numpy() >= .03))[0]
    eligible = np.array([i for i in eligible if i not in excluded])
    log(f'Discovery candidates after exclusions: {len(eligible)}')
    np.savez_compressed(out/'discovery_activations.npz', positive=pos.numpy(), negative=neg.numpy(), relevance=relevance)
    assert len(eligible) >= 12
    chosen = eligible[np.argsort(-relevance[eligible], kind='stable')[:48]]
    ids = np.r_[train.feature.to_numpy(), chosen]
    pred = predictors(model, sae, prim, ids)
    check = pred.iloc[:len(train)]
    errors = {}
    for col in PRED_SETS['full_no_magnitude']:
        errors[col] = float(np.max(np.abs(check[col].to_numpy()-train[col].to_numpy())))
        np.testing.assert_allclose(check[col], train[col], rtol=3e-4, atol=3e-5, err_msg=col)
    save(out/'predictor_verification.json', {'passed': True, 'max_absolute_error':errors,
        'training_csv_sha256':sha(training_path), 'excluded_feature_count':len(excluded),
        'candidate_overlap_with_training':len(set(chosen)&set(train.feature))})
    candidates = pred.iloc[len(train):].copy()
    candidates['discovery_selectivity'] = relevance[chosen]
    columns = PRED_SETS['full_no_magnitude']
    scaler = StandardScaler().fit(train[columns])
    X = scaler.transform(train[columns])
    Z = np.c_[np.ones(len(train)), train[['effect_l2','act_mag','frequency']]]
    y = train.collateral_raw.to_numpy()
    residual = y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    coefficients = {}
    for name, target in [('predicted_collateral',y), ('predicted_residual_collateral',residual)]:
        ridge = Ridge(alpha=1).fit(X,target)
        candidates[name] = ridge.predict(scaler.transform(candidates[columns]))
        coefficients[name] = {'coef':ridge.coef_.tolist(), 'intercept':float(ridge.intercept_)}
    candidates.to_csv(out/'candidates.csv', index=False)
    train.to_csv(out/'score_training.csv', index=False)
    save(out/'score_model.json', {'columns':columns,'mean':scaler.mean_.tolist(), 'scale':scaler.scale_.tolist(),
                                'coefficients':coefficients})
    from huggingface_hub import try_to_load_from_cache
    model_config_file=try_to_load_from_cache(cfg['model_name'],'config.json')
    model_revision=str(model_config_file).split('/snapshots/')[-1].split('/')[0] if '/snapshots/' in str(model_config_file) else None
    save(out/'manifest.json', {'model_revision':model_revision, 'setting':setting, 'config':cfg, 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'protocol_sha256':sha(ROOT/'PROTOCOL.md'), 'prompts_sha256':sha(ROOT/'prompts.json'),
        'runner_sha256':sha(__file__), 'candidates_sha256':sha(out/'candidates.csv'),
        'versions':{p:version(p) for p in ['torch','transformer-lens','sae-lens','transformers','numpy','scipy','scikit-learn']},
        'model_config':{k:str(getattr(model.cfg,k,None)) for k in ['model_name','d_model','n_layers','n_heads','dtype','normalization_type']}})
    log(f'Prepared {len(candidates)} held-out features; all training predictors verified')
    return candidates


def degeneration(text):
    words = re.findall(r"\b[\w']+\b",text.lower())
    trigrams = list(zip(words,words[1:],words[2:]))
    repeated = 1-len(set(trigrams))/len(trigrams) if trigrams else 0.
    return {'word_count':len(words), 'repeated_trigram_fraction':repeated,
            'degenerate':len(words)<8 or repeated>.2}


def exact_answer(text, answer):
    first = text.lstrip().split('\n',1)[0].strip().lstrip('"\' ')
    return bool(re.match(re.escape(answer)+r'\b',first,re.I))


class Sentiment:
    def __init__(self, data, out):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        name = 'siebert/sentiment-roberta-large-english'
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).cuda().eval()
        labels = self.model.config.id2label
        self.positive = next(int(k) for k,v in labels.items() if v.upper() == 'POSITIVE')
        p = self.score([x['text'] for x in data['calibration']])
        accuracy = np.mean((p>=.5)==np.array([x['label'] for x in data['calibration']]))
        save(out/'classifier_calibration.json', {'model':name,'revision':getattr(self.model.config,'_commit_hash',None),
             'label_mapping':labels,'accuracy':float(accuracy), 'n':len(p),
             'rows':[dict(x,positive_probability=float(y)) for x,y in zip(data['calibration'],p)]})
        assert accuracy>=.9, f'Classifier calibration failed: {accuracy}'
        log(f'Sentiment calibration accuracy {accuracy:.3f}')

    def score(self, texts):
        result=[]
        for i in range(0,len(texts),32):
            batch=self.tokenizer(texts[i:i+32],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
            result.extend(self.model(**batch).logits.softmax(-1)[:,self.positive].cpu().numpy().tolist())
        return np.array(result)


def generate_batch(model, hook, sae, jobs, max_tokens, sample):
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
    texts=[job['prompt'] for job in jobs]
    ids, mask=tokens(model,texts,bool(sae_field(sae,'prepend_bos',True)))
    vectors=torch.stack([sae.W_dec[j['feature']].detach().float()*j['alpha'] if j['feature']>=0
                         else torch.zeros(sae.cfg.d_in,device='cuda') for j in jobs])
    cache=TransformerLensKeyValueCache.init_cache(model.cfg,'cuda',len(jobs))
    uniforms=[]
    for j in jobs:
        seed=int(hashlib.sha256(f"{j['prompt_id']}:{j['seed']}".encode()).hexdigest()[:16],16)
        uniforms.append(np.random.default_rng(seed).random(max_tokens))
    uniforms=torch.tensor(np.array(uniforms),device='cuda',dtype=torch.float32)
    output=[]
    ended=torch.zeros(len(jobs),dtype=torch.bool,device='cuda')
    eos=model.tokenizer.eos_token_id

    def steer(resid, hook):
        resid[:,-1,:] += vectors.to(resid.dtype)
        return resid

    with model.hooks(fwd_hooks=[(hook,steer)]):
        for step in range(max_tokens):
            logits=model(ids,attention_mask=mask,past_kv_cache=cache)[:,-1,:].float()
            if sample:
                values,indices=torch.topk(logits/.8,50,dim=-1)
                cdf=values.softmax(-1).cumsum(-1)
                k=(cdf<uniforms[:,step,None]).sum(-1).clamp(max=49)
                nxt=indices.gather(1,k[:,None]).squeeze(1)
            else:
                nxt=logits.argmax(-1)
            nxt=torch.where(ended,torch.full_like(nxt,eos),nxt)
            output.append(nxt.cpu())
            ended|=nxt==eos
            ids=nxt[:,None]
            mask=torch.ones_like(ids)
            if ended.all():
                break
    rows=torch.stack(output,dim=1).tolist()
    result=[]
    for row in rows:
        if eos in row:
            row=row[:row.index(eos)]
        result.append((row,model.tokenizer.decode(row,skip_special_tokens=True)))
    return result


def run_jobs(model, sae, cfg, judge, jobs, path, batch_size):
    path=Path(path)
    done={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            assert row['job_id'] not in done
            done[row['job_id']]=row
    pending=[j for j in jobs if j['job_id'] not in done]
    log(f'{path.name}: {len(done)} saved, {len(pending)} remaining')
    with path.open('a') as f:
        for kind in ['review','qa']:
            subset=[j for j in pending if j['kind']==kind]
            for start in range(0,len(subset),batch_size):
                batch=subset[start:start+batch_size]
                generated=generate_batch(model,cfg['primary_hook'],sae,batch,32 if kind=='review' else 8,kind=='review')
                sentiment=judge.score([g[1] for g in generated]) if kind=='review' else [None]*len(batch)
                for job,(tok,text),prob in zip(batch,generated,sentiment):
                    row=dict(job,token_ids=tok,text=text,**degeneration(text))
                    if kind=='review':
                        row.update(positive_probability=float(prob),positive_label=bool(prob>=.5),
                                   nondegenerate_positive=bool(prob>=.5 and not row['degenerate']))
                    else:
                        row['correct']=exact_answer(text,job['answer'])
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                    done[row['job_id']]=row
                f.flush()
                if start%(batch_size*4)==0:
                    log(f'{path.name} {kind}: {start+len(batch)}/{len(subset)}')
    assert all(j['job_id'] in done for j in jobs)
    return [done[j['job_id']] for j in jobs]


def verify_generation(model,sae,cfg,candidates,out):
    jobs=[{'prompt':'The color of the sky is','prompt_id':'verify_a','seed':0,'feature':-1,'alpha':0.},
          {'prompt':'Review: I tried it yesterday and I think it was','prompt_id':'verify_b','seed':0,
           'feature':int(candidates.iloc[0].feature),'alpha':2.}]
    together=generate_batch(model,cfg['primary_hook'],sae,jobs,6,False)
    apart=[generate_batch(model,cfg['primary_hook'],sae,[j],6,False)[0] for j in jobs]
    assert [x[0] for x in together]==[x[0] for x in apart], 'Padded batched generation differs from individual generation'
    save(out/'generation_verification.json', {'passed':True,'check':'batched versus individual greedy token IDs',
                                             'cases':[dict(j,text=g[1],token_ids=g[0]) for j,g in zip(jobs,together)]})
    log('Batched cached generation verification passed')


def jobs_for(prompts, features, alphas, seeds, split, arm='candidate'):
    result=[]
    for alpha in alphas:
        for feature in features:
            for p in prompts:
                for seed in (seeds if p['kind']=='review' else [0]):
                    key=f"{split}:{p['id']}:{feature}:{alpha:g}:{seed}"
                    result.append({'job_id':key,'prompt_id':p['id'],'kind':p['kind'],'prompt':p['prompt'],
                                   'answer':p.get('answer'),'feature':int(feature),'alpha':float(alpha),'seed':seed,'arm':arm})
    return result


def develop(model,sae,cfg,judge,data,out,candidates,batch_size):
    if (out/'selection.json').exists():
        return json.loads((out/'selection.json').read_text())
    prompts=data['development']
    base_jobs=jobs_for(prompts,[-1],[0],[data['development_seed']],'dev',arm='unsteered')
    review=[p for p in prompts if p['kind']=='review']
    jobs=base_jobs+jobs_for(review,candidates.feature.tolist(),[1,2,4,8],[data['development_seed']],'dev')
    rows=run_jobs(model,sae,cfg,judge,jobs,out/'development.jsonl',batch_size)
    frame=pd.DataFrame(rows)
    baseline=frame[(frame.feature==-1)&(frame.kind=='review')].positive_probability.mean()
    qa_base=frame[(frame.feature==-1)&(frame.kind=='qa')].correct.mean()
    measure=frame[(frame.feature>=0)&(frame.kind=='review')].groupby(['alpha','feature']).agg(
        positive_probability=('positive_probability','mean'),degenerate_rate=('degenerate','mean')).reset_index()
    measure['gain']=measure.positive_probability-baseline
    measure=measure.merge(candidates,on='feature',validate='many_to_one')
    measure.to_csv(out/'development_features.csv',index=False)
    median=float(candidates.predicted_collateral.median())
    options=[]
    for alpha in [1,2,4,8]:
        d=measure[(measure.alpha==alpha)&(measure.gain>=.05)&(measure.degenerate_rate<=.25)].copy()
        lo=d[d.predicted_collateral<=median].reset_index(drop=True)
        hi=d[d.predicted_collateral>median].reset_index(drop=True)
        pairs=[]
        if len(lo) and len(hi):
            gain_distance=np.abs(lo.gain.to_numpy()[:,None]-hi.gain.to_numpy()[None,:])
            rel=np.abs(lo.discovery_selectivity.to_numpy()[:,None]-hi.discovery_selectivity.to_numpy()[None,:])
            norm=np.abs(lo.dec_norm.to_numpy()[:,None]-hi.dec_norm.to_numpy()[None,:])
            cost=gain_distance+.02*rel+.02*norm
            cost[gain_distance>.10]=1e6
            left,right=linear_sum_assignment(cost)
            ranked=sorted(zip(left,right),key=lambda pair:cost[pair])
            for l,h in ranked:
                if cost[l,h]>=1e6:
                    continue
                pairs.append({'low':int(lo.iloc[l].feature),'high':int(hi.iloc[h].feature),
                    'low_dev_gain':float(lo.iloc[l].gain),'high_dev_gain':float(hi.iloc[h].gain),
                    'cost':float(cost[l,h])})
                if len(pairs)==6:
                    break
        options.append({'alpha':alpha,'pairs':pairs,'n_effective':len(d),
                        'n_low_effective':len(lo),'n_high_effective':len(hi)})
    chosen=next((o for o in options if len(o['pairs'])==6), max(options,key=lambda o:(len(o['pairs']),-o['alpha'])))
    used={f for p in chosen['pairs'] for f in [p['low'],p['high']]}
    remaining=[int(f) for f in candidates.feature if f not in used]
    random=np.random.default_rng(180926).choice(remaining,size=min(6,len(remaining)),replace=False).tolist()
    selection=dict(chosen,random=random,all_strength_options=options,baseline_positive_probability=float(baseline),
        baseline_qa_accuracy=float(qa_base),qa_sensitivity_gate=bool(qa_base>=.7),
        feature_gate=bool(len(chosen['pairs'])>=3),candidate_median_score=median,
        development_sha256=sha(out/'development.jsonl'),candidates_sha256=sha(out/'candidates.csv'),
        prompts_sha256=sha(ROOT/'prompts.json'),protocol_sha256=sha(ROOT/'PROTOCOL.md'),
        selection_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(out/'selection.json',selection)
    log(f"Selection frozen: alpha={selection['alpha']}, {len(selection['pairs'])} pairs, baseline QA={qa_base:.3f}")
    return selection


def test(model,sae,cfg,judge,data,out,selection,batch_size):
    assert selection['prompts_sha256']==sha(ROOT/'prompts.json')
    jobs=jobs_for(data['test'],[-1],[0],data['review_seeds'],'test','unsteered')
    for arm in ['low','high']:
        ids=[p[arm] for p in selection['pairs']]
        jobs+=jobs_for(data['test'],ids,[selection['alpha']],data['review_seeds'],'test',arm)
    jobs+=jobs_for(data['test'],selection['random'],[selection['alpha']],data['review_seeds'],'test','random')
    save(out/'test_manifest.json',{'selection_sha256':sha(out/'selection.json'),'n_jobs':len(jobs),
                                 'job_sha256':digest(jobs),'runner_sha256':sha(__file__)})
    run_jobs(model,sae,cfg,judge,jobs,out/'test.jsonl',batch_size)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--setting',choices=['gpt2_small','gemma_2_2b'],required=True)
    ap.add_argument('--stage',choices=['prepare','develop','test','all'],default='all')
    ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.manual_seed(180926)
    source_path=ROOT/'results'/'source_snapshots'/f'run_{sha(__file__)}.py'
    source_path.parent.mkdir(parents=True,exist_ok=True)
    source_path.write_bytes(Path(__file__).read_bytes())
    data=inputs()
    out=ROOT/'results'/args.setting
    out.mkdir(parents=True,exist_ok=True)
    model,sae,cfg=load_setting(args.setting)
    candidates=prepare(args.setting,model,sae,cfg,data,out)
    if args.stage=='prepare':
        return
    verify_generation(model,sae,cfg,candidates,out)
    judge=Sentiment(data,out)
    selection=develop(model,sae,cfg,judge,data,out,candidates,args.batch_size)
    if args.stage=='develop':
        return
    test(model,sae,cfg,judge,data,out,selection,args.batch_size)
    log('Generation complete')


if __name__=='__main__':
    main()
