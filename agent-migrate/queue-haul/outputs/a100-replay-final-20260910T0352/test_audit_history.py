import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('audit_history',Path(__file__).with_name('audit-history.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_actual_output_retained_across_source_destination_and_reset():
    base={'phase':'service','cohort':'incoming','episode':'x','session':0,'done':True,'reset':False}
    rows=[{**base,'turn':0,'serving_role':'source','full_prompt_token_ids':[1,2],'token_ids':[3],
        'prompt_tokens':2,'end_ns':10,'client_dispatch_ns':0},
        {**base,'turn':1,'serving_role':'destination','full_prompt_token_ids':[1,2,3,4],'token_ids':[5],
        'prompt_tokens':4,'end_ns':20,'client_dispatch_ns':10},
        {**base,'turn':2,'reset':True,'full_prompt_token_ids':[7],'token_ids':[8],
        'prompt_tokens':1,'end_ns':30,'client_dispatch_ns':20}]
    result=module.audit(rows)
    assert result['nonreset_links']==1 and result['reset_links']==1
    assert result['source_to_destination_links']==1
    assert result['causal_violations']==result['retained_history_violations']==0
    rows[1]['full_prompt_token_ids'][2]=99;rows[1]['client_dispatch_ns']=9
    result=module.audit(rows)
    assert result['causal_violations']==result['retained_history_violations']==1
