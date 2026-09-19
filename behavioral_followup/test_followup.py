import numpy as np

from analyze import bootstrap_pairs
from prompts import build
from run import degeneration, exact_answer


def test_prompt_splits_and_known_answers():
    p=build()
    dev={x['prompt'] for x in p['development']}
    test={x['prompt'] for x in p['test']}
    assert not dev & test
    assert len([x for x in p['test'] if x['kind']=='review'])==64
    assert len([x for x in p['test'] if x['kind']=='qa'])==64
    for row in p['test']:
        if row['kind']=='qa':
            assert row['answer'] in row['prompt'].split('Fact:')[-1].split('Question:')[0]


def test_factual_scoring_rejects_late_and_partial_answers():
    assert exact_answer(' blue. It is a blue box.','blue')
    assert exact_answer('\n"Paris" is the answer.','Paris')
    assert not exact_answer('bluebird','blue')
    assert not exact_answer('red, or perhaps blue','blue')
    assert not exact_answer('I do not know.\nblue','blue')


def test_degenerate_text_is_not_quality_success():
    assert degeneration('')['degenerate']
    assert degeneration('very good indeed '*15)['degenerate']
    assert not degeneration('The instructions were clear and the device worked well throughout the entire weekend.')['degenerate']


def test_bootstrap_respects_feature_uncertainty():
    # Many identical prompts cannot erase disagreement between feature pairs.
    values=np.repeat(np.array([[-1.],[1.]]),64,axis=1)
    result=bootstrap_pairs(values,n_boot=2000)
    assert result['difference']==0
    assert result['ci_low']==-1 and result['ci_high']==1
    constant=bootstrap_pairs(np.full((6,64),.25),n_boot=100)
    assert constant['ci_low']==constant['ci_high']==constant['difference']==.25


def test_output_audit_rejects_corrupted_jobs_and_scores():
    from copy import deepcopy
    import pytest
    from verify import expected_jobs, verify_rows

    prompts=[{'id':'test_qa_00','kind':'qa','prompt':'What color?','answer':'blue'}]
    expected=expected_jobs(prompts,[-1],[0],[20260918],'test','unsteered')
    row=dict(expected[0],text='blue',token_ids=[42],correct=True,**degeneration('blue'))
    verify_rows([row],expected)
    for field,value in [('kind','review'),('seed',1),('correct',False),('word_count',99)]:
        changed=deepcopy(row)
        changed[field]=value
        with pytest.raises(AssertionError):
            verify_rows([changed],expected)
    with pytest.raises(AssertionError):
        verify_rows([row,row],expected)
    with pytest.raises(AssertionError):
        verify_rows([],expected)


def test_output_audit_rejects_invalid_sentiment_scores():
    import pytest
    from verify import expected_jobs, verify_rows

    prompts=[{'id':'test_review_00','kind':'review','prompt':'My review:'}]
    expected=expected_jobs(prompts,[17],[2],[20260918],'test','low')
    text='It was a wonderful experience that I would happily repeat.'
    row=dict(expected[0],text=text,token_ids=[42],positive_probability=.8,
             positive_label=True,nondegenerate_positive=True,**degeneration(text))
    verify_rows([row],expected)
    for changed in [dict(row,positive_probability=float('nan')),dict(row,positive_probability=1.2),
                    dict(row,positive_label=False),dict(row,nondegenerate_positive=False)]:
        with pytest.raises(AssertionError):
            verify_rows([changed],expected)


def test_score_reconstruction_restores_float32_after_csv_roundtrip():
    from io import StringIO
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from verify import reconstruct_scores

    training=pd.DataFrame({'x':[.9999998,.9999999,1.,1.0000001]})
    candidates=pd.DataFrame({'x':np.array([.99999994,1.],dtype=np.float32)})
    scaler=StandardScaler().fit(training)
    expected=scaler.transform(candidates)[:,0]
    reloaded=pd.read_csv(StringIO(candidates.to_csv(index=False)))
    wrong=(reloaded.x.to_numpy()-scaler.mean_[0])/scaler.scale_[0]
    assert np.max(np.abs(wrong-expected))>.01
    model={'columns':['x'],'coefficients':{'score':{'coef':[1.],'intercept':0.}}}
    actual=reconstruct_scores(reloaded,training,model)['score']
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
