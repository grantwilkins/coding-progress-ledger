from pool_replay_measure import cache_integrity


def test_missing_cold_usage_requires_independent_engine_evidence():
    cold, warm = {'prompt_tokens':2048,'cached_tokens':None}, {'cached_tokens':2032}
    before = {'vllm:prefix_cache_queries_total':0,'vllm:prefix_cache_hits_total':0,
              'vllm:external_prefix_cache_hits_total':0}
    after = {**before,'vllm:prefix_cache_queries_total':2048}
    result = cache_integrity(cold,warm,before,after)
    assert result['passed'] and result['cold_usage_cached_tokens'] is None
    assert not cache_integrity(cold,warm,before,{**after,'vllm:prefix_cache_hits_total':16})['passed']
    assert not cache_integrity(cold,warm,before,before)['passed']
